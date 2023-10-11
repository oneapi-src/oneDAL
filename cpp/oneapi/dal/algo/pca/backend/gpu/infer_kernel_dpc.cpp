/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/pca/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

namespace oneapi::dal::pca::backend {

namespace pr = oneapi::dal::backend::primitives;

using dal::backend::context_gpu;
using model_t = model<task::dim_reduction>;
using input_t = infer_input<task::dim_reduction>;
using result_t = infer_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;

template <typename Float>
static result_t infer(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& queue = ctx.get_queue();
    const auto data = input.get_data();
    auto model = input.get_model();
    auto eigenvectors = model.get_eigenvectors();
    auto means = model.get_means();
    auto eigenvalues = model.get_eigenvalues();

    const std::int64_t row_count = data.get_row_count();
    ONEDAL_ASSERT(row_count > 0);
    const std::int64_t col_count = data.get_column_count();
    ONEDAL_ASSERT(col_count > 0);
    const std::int64_t component_count = get_component_count(desc, data);
    ONEDAL_ASSERT(component_count > 0);
    dal::detail::check_mul_overflow(row_count, component_count);

    const auto data_nd = pr::table2ndarray<Float>(queue, data, sycl::usm::alloc::device);
    const auto means_nd = pr::table2ndarray_1d<Float>(queue, means, sycl::usm::alloc::device);
    auto mean_centered_data_nd =
        pr::ndarray<Float, 2>::empty(queue, { row_count, col_count }, sycl::usm::alloc::device);
    const auto eigenvectors_nd =
        pr::table2ndarray<Float>(queue, eigenvectors, sycl::usm::alloc::device);

    
    const auto kernel_minus = [=](const Float a, const Float b) -> Float {
        return a - b;
    };

    sycl::event mean_center_event;
    {
        auto mean_center_event =
            pr::element_wise(queue, kernel_minus, data_nd, means_nd, mean_centered_data_nd, {});
    }

    auto res_nd = pr::ndarray<Float, 2>::empty(queue,
                                               { row_count, component_count },
                                               sycl::usm::alloc::device);
    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, queue);
        gemm_event = pr::gemm(queue,
                              mean_centered_data_nd,
                              eigenvectors_nd.t(),
                              res_nd,
                              Float(1.0),
                              Float(0.0),
                              { });
        gemm_event.wait_and_throw();
    }

    if (eigenvalues.has_data()) {
        auto eigenvalues_nd =
            pr::table2ndarray_1d<Float>(queue, eigenvalues, sycl::usm::alloc::device);
        auto sqrt_eigenvalues_nd =
            pr::ndarray<Float, 1>::empty(queue, { component_count }, sycl::usm::alloc::device);
        sqrt_eigenvalues_nd = sqrt(eigenvalues_nd);
        const auto kernel_sqrt = [=](const Float a) -> Float {
            return std::sqrt(a);
        };
        element_wise(queue, kernel_sqrt, eigenvalues_nd, sqrt_eigenvalues_nd);

        auto result_whitened_nd = pr::ndarray<Float, 2>::empty(queue,
                                                               { row_count, component_count },
                                                               sycl::usm::alloc::device);
        sycl::event whiten_event;
        {
            const auto kernel_division = [=](const Float a, const Float b) -> Float {
                return a/b;
            };
            auto whiten_event = pr::element_wise(queue, kernel_division, res_nd, eigenvalues_nd, result_whitened_nd, {});
        }

        const auto res_array_whitened = result_whitened_nd.flatten(queue, { gemm_event });
        const auto res_table = homogen_table::wrap(res_array_whitened, row_count, component_count);

        return result_t{}.set_transformed_data(res_table);
    }

    const auto res_array = res_nd.flatten(queue, { gemm_event });
    const auto res_table = homogen_table::wrap(res_array, row_count, component_count);
    return result_t{}.set_transformed_data(res_table);
}

template <typename Float>
struct infer_kernel_gpu<Float, task::dim_reduction> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, task::dim_reduction>;
template struct infer_kernel_gpu<double, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
