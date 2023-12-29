/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/pca/backend/gpu/train_kernel_svd_impl.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/sign_flip.hpp"

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
namespace mkl = oneapi::fpk;
using alloc = sycl::usm::alloc;

using bk::context_gpu;

using task_t = task::dim_reduction;
using input_t = train_input<task_t>;
using result_t = train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
auto slice_data(sycl::queue& q,
                const pr::ndview<Float, 2>& data,
                std::int64_t component_count,
                std::int64_t column_count,
                const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_means, q);
    const std::int64_t column_count_local = data.get_dimension(1);
    auto data_to_compute =
        pr::ndarray<Float, 2>::empty(q, { component_count, column_count }, alloc::device);
    auto data_to_compute_ptr = data_to_compute.get_mutable_data();
    auto data_ptr = data.get_data();
    auto event = q.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(component_count, column_count);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t i = id[0];
            const std::int64_t j = id[1];
            data_to_compute_ptr[i * column_count + j] = data_ptr[i * column_count_local + j];
        });
    });
    return std::make_tuple(data_to_compute, event);
}

template <typename Float>
auto compute_singular_values(sycl::queue& q,
                             const pr::ndview<Float, 1>& data,
                             std::int64_t row_count,
                             std::int64_t column_count,
                             const bk::event_vector& deps = {}) {
    auto data_to_compute = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
    auto data_to_compute_ptr = data_to_compute.get_mutable_data();
    auto data_ptr = data.get_data();
    auto event = q.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_1d(column_count);
        h.parallel_for(range, [=](sycl::id<1> id) {
            data_to_compute_ptr[id] = sqrt((row_count - 1) * data_ptr[id]);
        });
    });
    return std::make_tuple(data_to_compute, event);
}

template <typename Float>
auto compute_explained_variances_on_host(sycl::queue& q,
                                         pr::ndarray<Float, 1> eigenvalues,
                                         pr::ndarray<Float, 1> vars,
                                         const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_explained_variances_on_host);
    ONEDAL_ASSERT(eigenvalues.has_mutable_data());

    const std::int64_t component_count = eigenvalues.get_dimension(0);
    const std::int64_t column_count = vars.get_dimension(0);
    auto explained_variances_ratio = pr::ndarray<Float, 1>::empty(component_count);

    auto eigvals_ptr = eigenvalues.get_data();
    auto vars_ptr = vars.get_data();
    auto explained_variances_ratio_ptr = explained_variances_ratio.get_mutable_data();

    Float sum = 0;
    for (std::int64_t i = 0; i < column_count; ++i) {
        sum += vars_ptr[i];
    }
    ONEDAL_ASSERT(sum > 0);
    const Float inverse_sum = 1.0 / sum;
    for (std::int64_t i = 0; i < component_count; ++i) {
        explained_variances_ratio_ptr[i] = eigvals_ptr[i] * inverse_sum;
    }
    return explained_variances_ratio;
}

template <typename Float, pr::ndorder order>
auto svd_decomposition(sycl::queue& queue,
                       pr::ndview<Float, 2, order>& data,
                       std::int64_t component_count,
                       const bk::event_vector& deps = {}) {
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto U = pr::ndarray<Float, 2>::empty(queue, { column_count, column_count }, alloc::device);

    auto S = pr::ndarray<Float, 1>::empty(queue, { component_count }, alloc::device);

    auto V_T = pr::ndarray<Float, 2>::empty(queue, { column_count, column_count }, alloc::device);

    Float* data_ptr = data.get_mutable_data();
    Float* U_ptr = U.get_mutable_data();
    Float* S_ptr = S.get_mutable_data();
    Float* V_T_ptr = V_T.get_mutable_data();
    std::int64_t lda = column_count;
    std::int64_t ldu = column_count;
    std::int64_t ldvt = column_count;
    sycl::event gesvd_event;
    {
        ONEDAL_PROFILER_TASK(gesvd, queue);
        gesvd_event = pr::gesvd<mkl::jobsvd::somevec, mkl::jobsvd::novec>(queue,
                                                                          row_count,
                                                                          column_count,
                                                                          data_ptr,
                                                                          lda,
                                                                          S_ptr,
                                                                          U_ptr,
                                                                          ldu,
                                                                          V_T_ptr,
                                                                          ldvt,
                                                                          { deps });
    }
    return std::make_tuple(U, S, V_T, gesvd_event);
}

template <typename Float>
result_t train_kernel_svd_impl<Float>::operator()(const descriptor_t& desc, const input_t& input) {
    ONEDAL_ASSERT(input.get_data().has_data());
    const auto data = input.get_data();
    const std::int64_t row_count = data.get_row_count();
    ONEDAL_ASSERT(data.get_column_count() > 0);

    const std::int64_t column_count = data.get_column_count();
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t component_count = get_component_count(desc, data);
    ONEDAL_ASSERT(component_count > 0);
    dal::detail::check_mul_overflow(column_count, component_count);
    constexpr bool bias = false; // Currently we use only unbiased covariance for PCA computation.

    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());
    pr::ndview<Float, 2> data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);

    auto xtx = pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);
    auto sums = pr::ndarray<Float, 1>::empty(q_, { column_count }, alloc::device);
    sycl::event sums_event;
    {
        ONEDAL_PROFILER_TASK(compute_sums, q_);
        sums_event =
            pr::reduce_by_columns(q_, data_nd, sums, pr::sum<Float>{}, pr::identity<Float>{}, {});
    }
    sycl::event means_event;
    if (desc.get_result_options().test(result_options::means)) {
        ONEDAL_PROFILER_TASK(compute_means, q_);
        auto means = pr::ndarray<Float, 1>::empty(q_, { column_count }, alloc::device);
        means_event = pr::means(q_, row_count, sums, means, { sums_event });
        result.set_means(homogen_table::wrap(means.flatten(q_, { means_event }), 1, column_count));
    }
    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, q_);
        gemm_event = gemm(q_, data_nd.t(), data_nd, xtx, Float(1.0), Float(0.0), { sums_event });
        gemm_event.wait_and_throw();
    }

    auto cov = pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);
    sycl::event copy_event;
    {
        ONEDAL_PROFILER_TASK(copy_xtx, q_);
        copy_event = copy(q_, cov, xtx, { means_event, gemm_event });
    }

    sycl::event cov_event;
    {
        ONEDAL_PROFILER_TASK(compute_covariance_matrix, q_);
        cov_event = pr::covariance(q_, row_count, sums, cov, bias, { copy_event });
    }

    auto vars = pr::ndarray<Float, 1>::empty(q_, { column_count }, alloc::device);
    sycl::event vars_event;
    {
        ONEDAL_PROFILER_TASK(compute_means, q_);
        vars_event = pr::variances(q_, cov, vars, { cov_event, means_event });
    }
    if (desc.get_result_options().test(result_options::vars)) {
        result.set_variances(
            homogen_table::wrap(vars.flatten(q_, { vars_event }), 1, column_count));
    }
    auto data_to_compute = cov;

    sycl::event corr_event;
    if (desc.get_normalization_mode() == normalization::zscore) {
        auto corr = pr::ndarray<Float, 2>::empty(q_, { column_count, column_count }, alloc::device);

        corr_event = pr::correlation_from_covariance(q_, row_count, cov, corr, bias, { cov_event });

        data_to_compute = corr;
    }

    auto [U, S, V_T, gesvd_event] =
        svd_decomposition(q_, data_to_compute, component_count, { vars_event });
    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(
            homogen_table::wrap(S.flatten(q_, { gesvd_event }), 1, component_count));
    }
    if (desc.get_result_options().test(result_options::singular_values)) {
        auto [singular_values, sv_event] =
            compute_singular_values(q_, S, row_count, column_count, { gesvd_event });
        if (desc.get_normalization_mode() == normalization::zscore) {
            result.set_singular_values(
                homogen_table::wrap(S.flatten(q_, { gesvd_event }), 1, component_count));
        }
        else {
            result.set_singular_values(
                homogen_table::wrap(singular_values.flatten(q_, { gesvd_event }), 1, column_count));
        }
    }
    if (desc.get_result_options().test(result_options::explained_variances_ratio)) {
        auto vars_host = vars.to_host(q_);
        auto eigvals_host = S.to_host(q_);
        auto explained_variances_ratio =
            compute_explained_variances_on_host(q_,
                                                eigvals_host,
                                                vars_host,
                                                { gesvd_event, vars_event });
        result.set_explained_variances_ratio(
            homogen_table::wrap(explained_variances_ratio.flatten(), 1, component_count));
    }
    //todo: after fix an MKL issue with gesvd, it should be correctly replaced by VT
    auto V_T_host = U.to_host(q_);
    if (desc.get_deterministic()) {
        //todo: after fix an MKL issue with gesvd, signs should be computed with U matrix
        sign_flip(V_T_host);
    }
    auto V_T_sign_flipped = V_T_host.to_device(q_);
    if (desc.get_result_options().test(result_options::eigenvectors)) {
        auto [sliced_data, event] =
            slice_data(q_, V_T_sign_flipped, component_count, column_count, { cov_event });
        result.set_eigenvectors(homogen_table::wrap(sliced_data.flatten(q_, { event }),
                                                    sliced_data.get_dimension(0),
                                                    sliced_data.get_dimension(1)));
    }

    return result;
}

template class train_kernel_svd_impl<float>;
template class train_kernel_svd_impl<double>;

} // namespace oneapi::dal::pca::backend
