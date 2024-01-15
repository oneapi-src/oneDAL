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
#include "oneapi/dal/algo/pca/backend/gpu/misc.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/algo/pca/backend/sign_flip.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include <iostream>
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"

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
            const std::int64_t row = id[0];
            const std::int64_t column = id[1];
            data_to_compute_ptr[row * column_count + column] =
                data_ptr[row * column_count_local + column];
        });
    });
    return std::make_tuple(data_to_compute, event);
}

template <typename Float>
auto compute_mean_centered_data(sycl::queue& q,
                                const pr::ndview<Float, 2>& data,
                                const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_means, q);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto [sums, sums_event] = compute_sums(q, data);
    auto [means, means_event] = compute_means(q, sums, row_count, { sums_event });

    auto data_to_compute =
        pr::ndarray<Float, 2>::empty(q, { row_count, column_count }, alloc::device);
    auto copy_event = copy(q, data_to_compute, data, { deps });
    auto data_to_compute_ptr = data_to_compute.get_mutable_data();
    auto means_ptr = means.get_data();
    auto event = q.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(row_count, column_count);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t i = id[0];
            const std::int64_t j = id[1];
            data_to_compute_ptr[i * column_count + j] =
                data_to_compute_ptr[i * column_count + j] - means_ptr[j];
        });
    });
    // auto data_to_compute_squared =
    //     pr::ndarray<Float, 2>::empty(q, { row_count, column_count }, alloc::device);
    // auto data_to_compute_squared_ptr = data_to_compute_squared.get_mutable_data();
    // auto event_ = q.submit([&](sycl::handler& h) {
    //     h.depends_on(event);
    //     const auto range = bk::make_range_2d(row_count, column_count);
    //     h.parallel_for(range, [=](sycl::id<2> id) {
    //         const std::int64_t i = id[0];
    //         const std::int64_t j = id[1];
    //         data_to_compute_squared_ptr[i * column_count + j] =
    //             data_to_compute_ptr[i * column_count + j] *
    //             data_to_compute_ptr[i * column_count + j] / (row_count - 1);
    //     });
    // });
    auto vars = pr::ndarray<Float, 1>::empty(q, { column_count }, alloc::device);
    // constexpr pr::sum<Float> binary{};
    // constexpr pr::identity<Float> unary{};
    // auto vars_event =
    //     pr::reduce_by_columns(q, data_to_compute_squared, vars, binary, unary, { event_ });
    return std::make_tuple(data_to_compute, means, vars, event);
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

    auto V_T = pr::ndarray<Float, 2>::empty(queue, { row_count, row_count }, alloc::device);

    Float* data_ptr = data.get_mutable_data();
    Float* U_ptr = U.get_mutable_data();
    Float* S_ptr = S.get_mutable_data();
    Float* V_T_ptr = V_T.get_mutable_data();
    std::int64_t lda = column_count;
    std::int64_t ldu = column_count;
    std::int64_t ldvt = row_count;
    sycl::event gesvd_event;
    {
        ONEDAL_PROFILER_TASK(gesvd, queue);
        gesvd_event = pr::gesvd<mkl::jobsvd::vectors, mkl::jobsvd::vectors>(queue,
                                                                            column_count,
                                                                            row_count,
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

    auto result = train_result<task_t>{}.set_result_options(desc.get_result_options());
    pr::ndview<Float, 2> data_nd = pr::table2ndarray<Float>(q_, data, alloc::device);

    auto [data_to_compute, means, vars, compute_event] =
        compute_mean_centered_data(q_, data_nd, {});

    auto [U, S, V_T, gesvd_event] =
        svd_decomposition(q_, data_to_compute, component_count, { compute_event });

    if (desc.get_result_options().test(result_options::singular_values)) {
        result.set_singular_values(
            homogen_table::wrap(S.flatten(q_, { gesvd_event }), 1, component_count));
    }
    result.set_means(homogen_table::wrap(means.flatten(q_, { gesvd_event }), 1, column_count));
    auto S_host = S.to_host(q_);
    auto eigenvalues = compute_eigenvalues_on_device(q_, S_host, row_count, { gesvd_event });
    if (desc.get_result_options().test(result_options::eigenvalues)) {
        result.set_eigenvalues(homogen_table::wrap(eigenvalues.flatten(), 1, component_count));
    }
    // if (desc.get_result_options().test(result_options::vars)) {
    //     result.set_variances(
    //         homogen_table::wrap(vars.flatten(q_, { gesvd_event }), 1, column_count));
    // }
    // if (desc.get_result_options().test(result_options::explained_variances_ratio)) {
    //     auto vars_host = vars.to_host(q_);
    //     auto explained_variances_ratio =
    //         compute_explained_variances_on_host(q_, eigenvalues, vars_host, { gesvd_event });
    //     result.set_explained_variances_ratio(
    //         homogen_table::wrap(explained_variances_ratio.flatten(), 1, component_count));
    // }

    auto U_host = U.to_host(q_);

    if (desc.get_deterministic()) {
        sign_flip(U_host);
    }

    auto U_host_sign_flipped = U_host.to_device(q_);
    if (desc.get_result_options().test(result_options::eigenvectors)) {
        auto [sliced_data, event] =
            slice_data(q_, U_host_sign_flipped, component_count, column_count, { gesvd_event });
        result.set_eigenvectors(homogen_table::wrap(sliced_data.flatten(q_, { event }),
                                                    sliced_data.get_dimension(0),
                                                    sliced_data.get_dimension(1)));
    }

    return result;
}

template class train_kernel_svd_impl<float>;
template class train_kernel_svd_impl<double>;

} // namespace oneapi::dal::pca::backend
