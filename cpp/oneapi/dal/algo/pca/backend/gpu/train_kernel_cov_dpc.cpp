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

#include "oneapi/dal/algo/pca/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/sign_flip.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::pca::backend {

namespace pr = oneapi::dal::backend::primitives;

using dal::backend::context_gpu;
using model_t = model<task::dim_reduction>;
using input_t = train_input<task::dim_reduction>;
using result_t = train_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;

template <typename Float>
auto compute_sums(sycl::queue& q,
                  const pr::ndview<Float, 2>& data,
                  const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_sums, q);
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
    ONEDAL_PROFILER_TASK(compute_correlation, q);
    ONEDAL_ASSERT(data.get_dimension(1) == sums.get_dimension(0));

    const std::int64_t column_count = data.get_dimension(1);
    auto corr =
        pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, sycl::usm::alloc::device);
    auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto vars = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto tmp = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto gemm_event = gemm(q, data.t(), data, corr, Float(1), Float(0), deps);
    auto corr_event = pr::correlation(q, data, sums, means, corr, vars, tmp, { gemm_event });

    auto smart_event = dal::backend::smart_event{ corr_event }.attach(tmp);
    return std::make_tuple(corr, means, vars, smart_event);
}

template <typename Float>
auto compute_eigenvectors_on_host(sycl::queue& q,
                                  pr::ndarray<Float, 2>&& corr,
                                  std::int64_t component_count,
                                  const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_eigenvectors_on_host);
    ONEDAL_ASSERT(corr.get_dimension(0) == corr.get_dimension(1));
    const std::int64_t column_count = corr.get_dimension(0);

    auto eigvecs = pr::ndarray<Float, 2>::empty({ component_count, column_count });
    auto eigvals = pr::ndarray<Float, 1>::empty(component_count);

    auto host_corr = corr.to_host(q, deps);
    pr::sym_eigvals_descending(host_corr, component_count, eigvecs, eigvals);

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

    const auto data_nd = pr::table2ndarray<Float>(q, data, sycl::usm::alloc::device);

    auto [sums, sums_event] = compute_sums(q, data_nd);
    auto [corr, means, vars, corr_event] = compute_correlation(q, data_nd, sums, { sums_event });

    auto [eigvecs, eigvals] =
        compute_eigenvectors_on_host(q, std::move(corr), component_count, { corr_event });

    if (desc.get_deterministic()) {
        sign_flip(eigvecs);
    }

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
