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

#include "oneapi/dal/algo/pca/backend/gpu/partial_train_kernel.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/memory.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::dim_reduction;
using input_t = partial_train_input<task_t>;
using result_t = partial_train_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
auto compute_sums(sycl::queue& q,
                  const pr::ndview<Float, 2>& data,
                  const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_sums, q);
    ONEDAL_ASSERT(data.has_data());

    const std::int64_t column_count = data.get_dimension(1);
    auto sums = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto reduce_event =
        pr::reduce_by_columns(q, data, sums, pr::sum<Float>{}, pr::identity<Float>{}, deps);

    return std::make_tuple(sums, reduce_event);
}

template <typename Float>
auto compute_crossproduct(sycl::queue& q,
                          const pr::ndview<Float, 2>& data,
                          const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_crossproduct, q);
    ONEDAL_ASSERT(data.has_data());

    const std::int64_t column_count = data.get_dimension(1);
    auto xtx = pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);

    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, q);
        gemm_event = gemm(q, data.t(), data, xtx, Float(1.0), Float(0.0));
        gemm_event.wait_and_throw();
    }

    return std::make_tuple(xtx, gemm_event);
}

template <typename Float>
auto init(sycl::queue& q,
          const std::int64_t row_count,
          const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(init_partial_results, q);

    auto result_nobs = pr::ndarray<Float, 1>::empty(q, 1);

    auto result_nobs_ptr = result_nobs.get_mutable_data();

    auto init_event = q.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range<1>(1);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<1> id) {
            result_nobs_ptr[0] = row_count;
        });
    });

    return std::make_tuple(result_nobs, init_event);
}

template <typename Float>
auto update_partial_results(sycl::queue& q,
                            const pr::ndview<Float, 2>& crossproducts,
                            const pr::ndview<Float, 1>& sums,
                            const pr::ndview<Float, 2>& current_crossproducts,
                            const pr::ndview<Float, 1>& current_sums,
                            const pr::ndview<Float, 1>& current_nobs,
                            const std::int64_t row_count,
                            const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(update_partial_results, q);

    auto column_count = crossproducts.get_dimension(1);
    auto result_sums = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_crossproducts =
        pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);
    auto result_nobs = pr::ndarray<Float, 1>::empty(q, 1);

    auto result_sums_ptr = result_sums.get_mutable_data();
    auto result_crossproducts_ptr = result_crossproducts.get_mutable_data();
    auto result_nobs_ptr = result_nobs.get_mutable_data();

    auto current_crossproducts_data = current_crossproducts.get_data();
    auto current_sums_data = current_sums.get_data();
    auto current_nobs_data = current_nobs.get_data();

    auto crossproducts_data = crossproducts.get_data();
    auto sums_data = sums.get_data();

    auto update_event = q.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range<2>(column_count, column_count);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<2> id) {
            const std::int64_t row = id[0], col = id[1];

            result_crossproducts_ptr[row * column_count + col] =
                crossproducts_data[row * column_count + col] +
                current_crossproducts_data[row * column_count + col];

            if (static_cast<std::int64_t>(col) < column_count) {
                result_sums_ptr[col] = sums_data[col] + current_sums_data[col];
            }

            if (row == 0 && col == 0) {
                result_nobs_ptr[0] = row_count + current_nobs_data[0];
            }
        });
    });

    return std::make_tuple(result_sums, result_crossproducts, result_nobs, update_event);
}

template <typename Float, typename Task>
static partial_train_result<Task> partial_train(const context_gpu& ctx,
                                                const descriptor_t& desc,
                                                const partial_train_input<Task>& input) {
    auto& q = ctx.get_queue();

    const auto data = input.get_data();
    auto result = partial_train_result();
    const auto input_ = input.get_prev();
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = data.get_column_count();
    dal::detail::check_mul_overflow(row_count, column_count);
    dal::detail::check_mul_overflow(column_count, column_count);
    dal::detail::check_mul_overflow(component_count, column_count);

    const auto data_nd = pr::table2ndarray<Float>(q, data, sycl::usm::alloc::device);

    auto [sums, sums_event] = compute_sums(q, data_nd);

    auto [crossproduct, crossproduct_event] = compute_crossproduct(q, data_nd, { sums_event });
    const bool has_nobs_data = input_.get_partial_n_rows().has_data();

    if (has_nobs_data) {
        const auto sums_nd =
            pr::table2ndarray_1d<Float>(q, input_.get_partial_sum(), sycl::usm::alloc::device);
        const auto nobs_nd = pr::table2ndarray_1d<Float>(q, input_.get_partial_n_rows());

        const auto crossproducts_nd = pr::table2ndarray<Float>(q,
                                                               input_.get_partial_crossproduct(),
                                                               sycl::usm::alloc::device);

        auto [result_sums, result_crossproducts, result_nobs, update_event] =
            update_partial_results(q,
                                   crossproduct,
                                   sums,
                                   crossproducts_nd,
                                   sums_nd,
                                   nobs_nd,
                                   row_count,
                                   { crossproduct_event });
        result.set_partial_sum(
            homogen_table::wrap(result_sums.flatten(q, { update_event }), 1, column_count));
        result.set_partial_crossproduct(
            homogen_table::wrap(result_crossproducts.flatten(q, { update_event }),
                                column_count,
                                column_count));
        result.set_partial_n_rows(
            homogen_table::wrap(result_nobs.flatten(q, { update_event }), 1, 1));
    }
    else {
        auto [result_nobs, init_event] = init<Float>(q, row_count, { crossproduct_event });

        result.set_partial_sum(
            homogen_table::wrap(sums.flatten(q, { init_event }), 1, column_count));
        result.set_partial_crossproduct(homogen_table::wrap(crossproduct.flatten(q, { init_event }),
                                                            column_count,
                                                            column_count));
        result.set_partial_n_rows(
            homogen_table::wrap(result_nobs.flatten(q, { init_event }), 1, 1));
    }
    return result;
}

template <typename Float>
struct partial_train_kernel_gpu<Float, method::cov, task::dim_reduction> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return partial_train<Float, task::dim_reduction>(ctx, desc, input);
    }
};

template struct partial_train_kernel_gpu<float, method::cov, task::dim_reduction>;
template struct partial_train_kernel_gpu<double, method::cov, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
