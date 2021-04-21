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

#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"

#define ONEDAL_ENABLE_PROFILING
#include "oneapi/dal/backend/profiling.hpp"

#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include <daal/src/algorithms/pca/pca_dense_correlation_batch_kernel.h>

namespace oneapi::dal::pca::backend {

namespace interop = dal::backend::interop;
namespace daal_pca = daal::algorithms::pca;
namespace pr = oneapi::dal::backend::primitives;

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

template <typename Float, typename Cpu>
inline daal_pca_cor_cpu_kernel_iface_ptr<Float> make_daal_cpu_kernel(Cpu cpu) {
    using Kernel = daal_pca_cor_cpu_kernel_t<Float, interop::to_daal_cpu_type<Cpu>::value>;
    return daal::services::SharedPtr<Kernel>{ new Kernel{} };
}

template <typename Float>
auto compute_sums(sycl::queue& q,
                  const pr::ndview<Float, 2>& data,
                  const dal::backend::event_vector& deps = {}) {
    const std::int64_t column_count = data.get_dimension(1);
    auto sums = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto reduce_event =
        pr::reduce_by_columns(q, data, sums, pr::sum<Float>{}, pr::identity<Float>{}, deps);
    return std::make_tuple(sums, reduce_event);
}

template <typename Float>
auto compute_correlation(sycl::queue& q,
                         const pr::ndview<Float, 2>& data,
                         const pr::ndview<Float, 1>& sums,
                         const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.get_dimension(1) == sums.get_dimension(0));

    const std::int64_t column_count = data.get_dimension(1);
    auto corr =
        pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, sycl::usm::alloc::device);
    auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto vars = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto tmp = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);

    auto corr_event = pr::correlation(q, data, sums, corr, means, vars, tmp, deps);

    auto smart_event = dal::backend::smart_event{ corr_event }.attach(tmp);
    return std::make_tuple(corr, means, vars, smart_event);
}

template <typename Float>
inline void write_binary_file(const std::string& filename, const Float* data, size_t count) {
    std::fstream file(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(data), count * sizeof(Float));
}

template <typename Float>
auto compute_eigenvectors_on_host(sycl::queue& q,
                                  pr::ndarray<Float, 2>&& corr,
                                  std::int64_t component_count,
                                  const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1));
    const std::int64_t column_count = corr.get_dimension(0);

    // ONEDAL_TIMER_BEGIN(pca_cov_training, corr_to_host)
    auto host_corr = corr.to_host(q, deps);
    // ONEDAL_TIMER_END(corr_to_host)

    auto eigvecs = pr::ndarray<Float, 2>::empty({ component_count, column_count });
    auto eigvals = pr::ndarray<Float, 1>::empty(component_count);

    const auto daal_cpu_kernel = dal::backend::dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
        return make_daal_cpu_kernel<Float>(cpu);
    });

    auto corr_fake = pr::ndarray<Float, 2>::empty({ column_count, column_count });
    Float* corr_fake_ptr = corr_fake.get_mutable_data();
    // const Float* host_corr_ptr = host_corr.get_data();
    for (std::int64_t i = 0; i < column_count; i++) {
        for (std::int64_t j = 0; j < column_count; j++) {
            // corr_fake_ptr[i * column_count + j] = host_corr_ptr[i * column_count + j];
            // const Float x = host_corr_ptr[i * column_count + j];
            corr_fake_ptr[i * column_count + j] = 0.0f;
        }
        corr_fake_ptr[i * column_count + i] = 1.0f;
    }

    // auto host_corr_flat = host_corr.flatten();
    auto host_corr_flat = corr_fake.flatten();
    auto eigvecs_flat = eigvecs.flatten();
    auto eigvals_flat = eigvals.flatten();

    auto corr_nt =
        interop::convert_to_daal_homogen_table(host_corr_flat, column_count, column_count);
    auto eigvecs_nt =
        interop::convert_to_daal_homogen_table(eigvecs_flat, component_count, column_count);
    auto eigvals_nt = interop::convert_to_daal_homogen_table(eigvals_flat, 1, component_count);

    ONEDAL_TIMER_BEGIN(pca_cov_training, sym_eigvals_descending)
    daal_cpu_kernel->computeCorrelationEigenvalues(*corr_nt, *eigvecs_nt, *eigvals_nt);
    ONEDAL_TIMER_END(sym_eigvals_descending)

    // write_binary_file("correlation_dal.bin", host_corr.get_data(), host_corr.get_count());

    // ONEDAL_TIMER_BEGIN(pca_cov_training, sym_eigvals_descending)
    // pr::sym_eigvals_descending(host_corr, component_count, eigvecs, eigvals);
    // ONEDAL_TIMER_END(sym_eigvals_descending)

    return std::make_tuple(eigvecs, eigvals);
}

template <typename Float>
static result_t train(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& q = ctx.get_queue();
    const auto data = input.get_data();

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = get_component_count(desc, data);
    dal::detail::check_mul_overflow(row_count, column_count);
    dal::detail::check_mul_overflow(column_count, column_count);
    dal::detail::check_mul_overflow(component_count, column_count);

    // ONEDAL_TIMER_BEGIN(pca_cov_training, flatten_table)
    const auto data_nd = pr::flatten_table<Float, row_accessor>(q, data, sycl::usm::alloc::device);
    // ONEDAL_TIMER_END(flatten_table)

    // ONEDAL_TIMER_BEGIN(pca_cov_training, compute_sums)
    auto [sums, sums_event] = compute_sums(q, data_nd);
    // ONEDAL_TIMER_END(compute_sums, sums_event)

    // ONEDAL_TIMER_BEGIN(pca_cov_training, compute_correlation)
    auto [corr, means, vars, corr_event] = compute_correlation(q, data_nd, sums, { sums_event });
    // ONEDAL_TIMER_END(compute_correlation, corr_event)

    q.wait_and_throw();
    auto [eigvecs, eigvals] =
        compute_eigenvectors_on_host(q, std::move(corr), component_count, { corr_event });

    const auto model = model_t{}.set_eigenvectors(
        homogen_table::wrap(eigvecs.flatten(), component_count, column_count));

    return result_t{}
        .set_model(model)
        .set_eigenvalues(homogen_table::wrap(eigvals.flatten(), 1, component_count))
        .set_means(homogen_table::wrap(means.flatten(q), 1, column_count))
        .set_variances(homogen_table::wrap(vars.flatten(q), 1, column_count));
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
