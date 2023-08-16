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

#include "oneapi/dal/algo/covariance/backend/gpu/partial_compute_kernel.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/memory.hpp"

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"

namespace oneapi::dal::covariance::backend {

namespace bk = dal::backend;
namespace pr = oneapi::dal::backend::primitives;

using alloc = sycl::usm::alloc;

using bk::context_gpu;
using task_t = task::compute;
using input_t = partial_compute_input<task_t>;
using result_t = partial_compute_input<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
auto compute_sums(sycl::queue& q,
                  const pr::ndview<Float, 2>& data,
                  const dal::backend::event_vector& deps = {}) {
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
//TODO:optimize and rewrite this function. its temporary implementation
template <typename Float>
auto update_partial_results(sycl::queue& q,
                            const pr::ndview<Float, 2>& crossproducts,
                            const pr::ndview<Float, 1>& sums,
                            const pr::ndview<Float, 2>& current_crossproducts,
                            const pr::ndview<Float, 1>& current_sums,
                            const pr::ndview<Float, 1>& current_nobs,
                            std::int64_t row_count,
                            const dal::backend::event_vector& deps = {}) {
    auto column_count = crossproducts.get_dimension(1);
    auto result_sums = pr::ndarray<Float, 1>::empty(q, column_count, alloc::device);
    auto result_crossproducts =
        pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, alloc::device);
    auto result_nobs = pr::ndarray<Float, 1>::empty(q, 1, alloc::device);
    auto result_nobs_ptr = result_nobs.get_mutable_data();
    auto current_nobs_ptr = current_nobs.get_mutable_data();

    auto result_sums_ptr = result_sums.get_mutable_data();
    auto sums_ptr = sums.get_mutable_data();
    auto current_sums_ptr = current_sums.get_mutable_data();
    auto result_crossproducts_ptr = result_crossproducts.get_mutable_data();
    auto crossproducts_ptr = crossproducts.get_mutable_data();
    auto current_crossproducts_ptr = current_crossproducts.get_mutable_data();
    auto update_event = q.submit([&](sycl::handler& cgh) {
        const auto range = sycl::range<2>(column_count, column_count);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::item<2> id) {
            const std::int64_t i = id.get_id(0);
            const std::int64_t j = id.get_id(1);

            if (i < column_count && j < column_count) {
                result_crossproducts_ptr[i * column_count + j] =
                    crossproducts_ptr[i * column_count + j] +
                    current_crossproducts_ptr[i * column_count + j];
                result_sums_ptr[i] = current_sums_ptr[i] + sums_ptr[i];
            }
            result_nobs_ptr[0] = row_count + current_nobs_ptr[0];
        });
    });

    return std::make_tuple(result_sums, result_crossproducts, result_nobs, update_event);
}

template <typename Float, typename Task>
static partial_compute_input<Task> partial_compute(const context_gpu& ctx,
                                                   const descriptor_t& desc,
                                                   const partial_compute_input<Task>& input) {
    auto& q = ctx.get_queue();

    const auto data = input.get_data();
    auto result = partial_compute_input(input);

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = data.get_column_count();
    dal::detail::check_mul_overflow(row_count, column_count);
    dal::detail::check_mul_overflow(column_count, column_count);
    dal::detail::check_mul_overflow(component_count, column_count);

    const auto data_nd = pr::table2ndarray<Float>(q, data, sycl::usm::alloc::device);
    const auto sums_nd = pr::table2ndarray_1d<Float>(q, input.get_sums(), sycl::usm::alloc::device);
    const auto nobs_nd =
        pr::table2ndarray_1d<Float>(q, input.get_nobs_table(), sycl::usm::alloc::device);
    const auto crossproducts_nd =
        pr::table2ndarray<Float>(q, input.get_crossproduct_matrix(), sycl::usm::alloc::device);

    auto [sums, sums_event] = compute_sums(q, data_nd);

    auto [crossproduct, crossproduct_event] = compute_crossproduct(q, data_nd);

    auto [result_sums, result_crossproducts, result_nobs, update_event] =
        update_partial_results(q,
                               crossproduct,
                               sums,
                               crossproducts_nd,
                               sums_nd,
                               nobs_nd,
                               row_count,
                               { crossproduct_event });

    result.set_sums(
        (homogen_table::wrap(result_sums.flatten(q, { update_event }), 1, column_count)));
    result.set_crossproduct_matrix(
        (homogen_table::wrap(result_crossproducts.flatten(q, { update_event }),
                             column_count,
                             column_count)));
    result.set_nobs_table((homogen_table::wrap(result_nobs.flatten(q, { update_event }), 1, 1)));

    return result;
}

template <typename Float>
struct partial_compute_kernel_gpu<Float, method::by_default, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return partial_compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct partial_compute_kernel_gpu<float, method::dense, task::compute>;
template struct partial_compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::backend
