/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/covariance/backend/gpu/compute_kernel.hpp"

#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::covariance::backend {

namespace pr = oneapi::dal::backend::primitives;

using dal::backend::context_cpu;
using dal::backend::context_gpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

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
inline auto compute_means(sycl::queue& q,
                          const pr::ndview<Float, 2>& data,
                          const pr::ndview<Float, 1>& sums,
                          const dal::backend::event_vector& deps = {}) {
    const std::int64_t column_count = data.get_dimension(1);
    auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto means_event = pr::means(q, data, sums, means, deps);

    auto smart_event = dal::backend::smart_event{ means_event };
    return std::make_tuple(means, smart_event);
}

template <typename Float>
inline auto compute_covariance(sycl::queue& q,
                               const pr::ndview<Float, 2>& data,
                               const pr::ndview<Float, 1>& sums,
                               const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.get_dimension(1) == sums.get_dimension(0));

    const std::int64_t column_count = data.get_dimension(1);
    auto cov =
        pr::ndarray<Float, 2>::empty(q, { column_count, column_count }, sycl::usm::alloc::device);
    auto means = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);
    auto vars = pr::ndarray<Float, 1>::empty(q, { column_count }, sycl::usm::alloc::device);

    auto cov_event = pr::covariance(q, data, sums, cov, means, vars, deps);

    auto smart_event = dal::backend::smart_event{ cov_event };
    return std::make_tuple(cov, means, vars, smart_event);
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

template <typename Float, typename Task>
static compute_result<Task> compute(const context_gpu& ctx,
                                    const descriptor_t& desc,
                                    const input_t& input) {
    //bool is_mean_computed = false;
    auto result = compute_result<Task>{}.set_result_options(desc.get_result_options());
    auto& q = ctx.get_queue();
    const auto data = input.get_data();

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = data.get_column_count();
    dal::detail::check_mul_overflow(row_count, column_count);
    dal::detail::check_mul_overflow(column_count, column_count);
    dal::detail::check_mul_overflow(component_count, column_count);

    const auto data_nd = pr::table2ndarray<Float>(q, data, sycl::usm::alloc::device);

    auto [sums, sums_event] = compute_sums(q, data_nd);

    if (desc.get_result_options().test(result_options::cov_matrix)) {
        auto [cov, means, vars, cov_event] = compute_covariance(q, data_nd, sums, { sums_event });
        //is_mean_computed = true;

        result.set_cov_matrix((homogen_table::wrap(cov.flatten(q), column_count, column_count)));
    }
    if (desc.get_result_options().test(result_options::cor_matrix)) {
        auto [corr, means, vars, corr_event] =
            compute_correlation(q, data_nd, sums, { sums_event });

        //is_mean_computed = true;

        result.set_cor_matrix((homogen_table::wrap(corr.flatten(q), column_count, column_count)));
    }
    if (desc.get_result_options().test(result_options::means)) {
        //if (!is_mean_computed) {
        auto [means, means_event] = compute_means(q, data_nd, sums, { sums_event });
        //}
        result.set_means(homogen_table::wrap(means.flatten(q), 1, column_count));
    }
    return result;
}

template <typename Float>
struct compute_kernel_gpu<Float, method::dense, task::compute> {
    compute_result<task::compute> operator()(const context_gpu& ctx,
                                             const descriptor_t& desc,
                                             const input_t& input) const {
        return compute<Float, task::compute>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::covariance::backend
