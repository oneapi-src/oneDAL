/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

//#include <daal/src/algorithms/svm/oneapi/svm_predict_kernel_oneapi.h>

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/svm/backend/gpu/svm_predict.hpp"
#include "oneapi/dal/algo/svm/backend/model_conversion.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/transfer.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"


namespace oneapi::dal::svm::backend {

using dal::backend::context_gpu;
using model_t = model<task::classification>;
using input_t = infer_input<task::classification>;
using result_t = infer_result<task::classification>;
using descriptor_t = detail::descriptor_base<task::classification>;

namespace daal_svm = daal::algorithms::svm;
namespace daal_kernel_function = daal::algorithms::kernel_function;
namespace interop = dal::backend::interop;
namespace pr = dal::backend::primitives;

template <typename Float>
static result_t infer(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
//    return call_daal_kernel<Float>(ctx, desc, input.get_model(), input.get_data());
    auto& queue = ctx.get_queue();

    const std::uint64_t class_count = desc.get_class_count();
    if (class_count > 2) {
        throw unimplemented(dal::detail::error_messages::svm_multiclass_not_implemented_for_gpu());
    }

    const auto data = input.get_data();
    const auto data_nd = pr::table2ndarray<Float>(queue, data, sycl::usm::alloc::device);
    const auto trained_model = input.get_model();

    const auto kernel_ptr = detail::get_kernel_ptr(desc);
    if (!kernel_ptr) {
        throw internal_error{ dal::detail::error_messages::unknown_kernel_function_type() };
    }

    const std::int64_t row_count  = data.get_row_count();

// DAAL_ASSERT(result.getNumberOfRows() == nVectors)
// DAAL_ASSERT(result.getNumberOfColumns() == 1)

//    BlockDescriptor<algorithmFPType> resultBlock;
//    DAAL_CHECK_STATUS(status, result.getBlockOfRows(0, nVectors, ReadWriteMode::writeOnly, resultBlock));
//    auto distanceBuff = resultBlock.getBuffer();

     auto distance_nd = pr::ndarray<Float, 1>::empty(queue, { row_count }, sycl::usm::alloc::device);

    auto sv_coeff_table = trained_model.get_coeffs();
    const std::int64_t n_sv  = trained_model.get_support_vector_count();

    if (n_sv == 0) {
        auto fill_event = distance_nd.fill(queue, Float(0.0));
        fill_event.wait_and_throw();
    } else {
        const auto sv_coeff_buff = pr::table2ndarray_1d<Float>(queue, sv_coeff_table, sycl::usm::alloc::device);

        const auto bieses = pr::table2ndarray_1d<Float>(trained_model.get_biases());
        const auto bias = *(bieses.get_data());
        std::cout << "bias=" << bias << "\n";
        auto fill_event = distance_nd.fill(queue, bias);
        fill_event.wait_and_throw();

        auto svTable = trained_model.get_support_vectors();

        // const std::int64_t n_rows_per_block = xTable->getDataLayout() == NumericTableIface::csrArray ? nVectors : 1024;
        // const std::int64_t nblocks       = nVectors / n_rows_per_block + !!(nVectors % n_rows_per_block);

        const std::int64_t n_rows_per_block = 1024;
        const std::int64_t nblocks       = row_count / n_rows_per_block + !!(row_count % n_rows_per_block);

//        std::cout << "NBlocks" << nblocks << "\n";

        std::shared_ptr<predict_task<Float>> predict_task =
            std::make_shared<predict_task_dense<Float>>(queue, n_rows_per_block, data, svTable, kernel_ptr);

        for (std::int64_t iblock = 0; iblock < nblocks; ++iblock)
        {
//            std::cout << "IBlock:" << iblock << "\n nblocks: " << nblocks << "\n";
            const std::int64_t start_row          = iblock * n_rows_per_block;

            // std::cout << "startrow=" << start_row << "\n";
            const std::int64_t n_rows_per_block_real = (iblock != nblocks - 1) ? n_rows_per_block : row_count - iblock * n_rows_per_block;
                        // std::cout << "n_rows_per_block_real=" << n_rows_per_block_real << "\n";

            auto distance_view = pr::ndarray<Float, 2>::wrap(distance_nd.get_mutable_data() + start_row, {n_rows_per_block_real, 1});
            
            auto kernelResBuff = predict_task->kernel_compute(queue, start_row, n_rows_per_block_real);

            // std::cout << "****kernelResBuff****\n";

            // auto print1 = kernelResBuff.to_host(queue).get_data();
            // for(int i = 1079000; i < 1084000; ++i) {
            //     std::cout << print1[i] << " ";
            // }
            
            // std::cout << "\n****kernelResBuff****\n";

            auto reshape_sv_coeff = sv_coeff_buff.reshape(pr::ndshape<2>{ n_sv, 1 });

            // std::cout << "****reshape_sv_coeff****\n";

            // auto print2 = sv_coeff_buff.to_host(queue).get_data();
            // for(int i = 0; i <  sv_coeff_buff.to_host(queue).get_count() && i < 1000; ++i) {
            //     std::cout << print2[i] << " ";
            // }
            
            // std::cout << "\n****reshape_sv_coeff****\n";

            auto gemm_event = pr::gemm(queue, kernelResBuff, reshape_sv_coeff, distance_view, Float(1), Float(1));
            /// m = n_rows_per_block_real n = 1 k = n_sv alpha =  algorithmFPType(1.0) a_buffer = kernelResBuff lda = n_sv offsetA = 0 b_buffer = sv_coeff_buff ldb = 1 offsetB = 0 beta = 1.0 c_buffer = distaceBuff  ldc = 1 offsetC = start_row
            // A = m x k ; B = k x n
            gemm_event.wait_and_throw();
            // std::cout << "****distance_nd****\n" ;

            // auto print3 = distance_nd.to_host(queue).get_data();
            // for(int i = 0; i < distance_nd.to_host(queue).get_count() && i < 1000; ++i) {
            //     std::cout << print3[i] << " ";
            // }

            // std::cout << "\n****distance_nd****\n";
        }
        }
///
    // std::cout << "********\n";

    // auto print = distance_nd.to_host(queue).get_data();
    // for(int i = 0; i < distance_nd.get_count(); ++i) {
    //     std::cout << print[i] << " ";
    // }

    // std::cout << "\n********\n";

    const auto arr_decision_func_host = distance_nd.to_host(queue);
    const auto arr_decision_data = arr_decision_func_host.get_data();

    // TODO: rework with help dpcpp code
    auto arr_response = pr::ndarray<Float, 1>::empty(row_count * 1);
    auto response_data = arr_response.get_mutable_data();
    for (std::int64_t i = 0; i < row_count; ++i) {
        response_data[i] = arr_decision_data[i] >= 0
                               ? trained_model.get_second_class_response()
                               : trained_model.get_first_class_response();
    }

    return result_t()
        .set_decision_function(homogen_table::wrap(arr_decision_func_host.flatten(), row_count, 1))
        .set_responses(homogen_table::wrap(arr_response.flatten(), row_count, 1));
}

template <typename Float>
struct infer_kernel_gpu<Float, method::by_default, task::classification> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template <typename Float>
struct infer_kernel_gpu<Float, method::by_default, task::nu_classification> {
    infer_result<task::nu_classification> operator()(
        const context_gpu& ctx,
        const detail::descriptor_base<task::nu_classification>& desc,
        const infer_input<task::nu_classification>& input) const {
        throw unimplemented(
            dal::detail::error_messages::svm_nu_classification_task_is_not_implemented_for_gpu());
    }
};

template struct infer_kernel_gpu<float, method::by_default, task::classification>;
template struct infer_kernel_gpu<double, method::by_default, task::classification>;
template struct infer_kernel_gpu<float, method::by_default, task::nu_classification>;
template struct infer_kernel_gpu<double, method::by_default, task::nu_classification>;

} // namespace oneapi::dal::svm::backend
