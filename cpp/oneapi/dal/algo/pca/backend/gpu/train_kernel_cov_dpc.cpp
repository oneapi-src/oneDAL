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

#include <daal/src/algorithms/pca/pca_dense_correlation_batch_kernel.h>
#include <daal/src/algorithms/pca/oneapi/pca_dense_correlation_batch_kernel_ucapi.h>

#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

namespace oneapi::dal::pca::backend {

namespace interop = dal::backend::interop;
namespace daal_pca = daal::algorithms::pca;
namespace daal_cov = daal::algorithms::covariance;

using dal::backend::context_cpu;
using dal::backend::context_gpu;
using model_t = model<task::dim_reduction>;
using input_t = train_input<task::dim_reduction>;
using result_t = train_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;

template <typename Float>
using daal_pca_cor_cpu_kernel_iface_ptr =
    daal::services::SharedPtr<daal_pca::internal::PCACorrelationBaseIface<Float>>;

template <typename Float, daal::CpuType Cpu>
using daal_pca_cor_cpu_kernel_t = daal_pca::internal::PCACorrelationKernel<daal::batch, Float, Cpu>;

template <typename Float>
using daal_pca_cor_gpu_kernel_t = daal_pca::internal::PCACorrelationKernelBatchUCAPI<Float>;

template <typename Float, typename Cpu>
daal_pca_cor_cpu_kernel_iface_ptr<Float> make_daal_cpu_kernel(Cpu cpu) {
    using Kernel = daal_pca_cor_cpu_kernel_t<Float, interop::to_daal_cpu_type<Cpu>::value>;
    return daal::services::SharedPtr<Kernel>{ new Kernel{} };
}

template <typename Float, typename... Args>
static void call_daal_gpu_kernel(Args&&... args) {
    // GPU kernel depends on CPU ISA-specific kernel, so create this kernels
    // first using CPU dispatching mechanism and then pass to GPU one
    const auto daal_cpu_kernel = dal::backend::dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
        return make_daal_cpu_kernel<Float>(cpu);
    });

    daal_pca_cor_gpu_kernel_t<Float> daal_gpu_kernel{ daal_cpu_kernel };
    const auto status = daal_gpu_kernel.compute(std::forward<Args>(args)...);
    interop::status_to_exception(status);
}

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = get_component_count(desc, data);

    auto arr_data = row_accessor<const Float>{ data }.pull(queue);

    dal::detail::check_mul_overflow(column_count, component_count);
    auto arr_eigvec = array<Float>::empty(queue, column_count * component_count);
    auto arr_eigval = array<Float>::empty(queue, 1 * component_count);
    auto arr_means = array<Float>::empty(queue, 1 * column_count);
    auto arr_vars = array<Float>::empty(queue, 1 * column_count);

    const auto daal_data =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_data, row_count, column_count);
    const auto daal_eigvec = interop::convert_to_daal_sycl_homogen_table(queue,
                                                                         arr_eigvec,
                                                                         component_count,
                                                                         column_count);
    const auto daal_eigval =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_eigval, 1, component_count);
    const auto daal_means =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_means, 1, column_count);
    const auto daal_variances =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_vars, 1, column_count);

    daal_cov::Batch<Float, daal_cov::defaultDense> covariance_alg;
    covariance_alg.input.set(daal_cov::data, daal_data);

    constexpr bool is_correlation = false;
    constexpr std::uint64_t results_to_compute =
        std::uint64_t(daal_pca::mean | daal_pca::variance | daal_pca::eigenvalue);
    const bool is_deterministic = desc.get_deterministic();

    call_daal_gpu_kernel<Float>(is_correlation,
                                is_deterministic,
                                *daal_data,
                                &covariance_alg,
                                static_cast<DAAL_UINT64>(results_to_compute),
                                *daal_eigvec,
                                *daal_eigval,
                                *daal_means,
                                *daal_variances);

    // clang-format off
    const auto model = model_t{}
        .set_eigenvectors(
            dal::detail::homogen_table_builder{}
                .reset(arr_eigvec, component_count, column_count)
                .build()
        );

    return result_t{}
        .set_model(model)
        .set_eigenvalues(
            dal::detail::homogen_table_builder{}
                .reset(arr_eigval, 1, component_count)
                .build()
        )
        .set_variances(
            dal::detail::homogen_table_builder{}
                .reset(arr_vars, 1, column_count)
                .build()
        )
        .set_means(
            dal::detail::homogen_table_builder{}
                .reset(arr_means, 1, column_count)
                .build()
        );
    // clang-format on
}

template <typename Float>
static result_t train(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& queue = ctx.get_queue();



    return call_daal_kernel<Float>(ctx, desc, input.get_data());
}

template <typename Float>
struct train_kernel_gpu<Float, method::cov, task::dim_reduction> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, method::cov, task::dim_reduction>;
template struct train_kernel_gpu<double, method::cov, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
