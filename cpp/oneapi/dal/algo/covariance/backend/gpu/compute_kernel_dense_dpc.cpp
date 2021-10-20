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
#include "oneapi/dal/algo/covariance/backend/gpu/compute_kernel_dense_impl.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::covariance::backend {

namespace pr = oneapi::dal::backend::primitives;

using method_t = method::dense;
using task_t = task::compute;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

template <typename Float>
static result_t compute(const bk::context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) {
    //auto& q = ctx.get_queue();
    //const auto data = input.get_data();
    //bool is_corr_computed = false;
    //auto result = compute_result<Task>{}.set_result_options(desc.get_result_options());

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t component_count = data.get_column_count();
    dal::detail::check_mul_overflow(row_count, column_count);
    dal::detail::check_mul_overflow(column_count, column_count);
    dal::detail::check_mul_overflow(component_count, column_count);

    const auto data_nd = pr::table2ndarray<Float>(q, data, sycl::usm::alloc::device);

    auto [means, sums, means_event] = compute_means(q, data_nd);

    if (desc.get_result_options().test(result_options::cov_matrix)) {
        auto [cov, tmp, cov_event] = compute_covariance(q, data_nd, sums, means, { means_event });

        result.set_cov_matrix(
            (homogen_table::wrap(cov.flatten(q, { cov_event }), column_count, column_count)));

        if (desc.get_result_options().test(result_options::cor_matrix)) {
            is_corr_computed = true;

            auto [corr, corr_event] =
                compute_correlation_with_covariance(q, data_nd, cov, tmp, { cov_event });

            result.set_cor_matrix(
                (homogen_table::wrap(corr.flatten(q, { corr_event }), column_count, column_count)));
        }
    }
    if (desc.get_result_options().test(result_options::cor_matrix) && !is_corr_computed) {
        auto [corr, corr_event] = compute_correlation(q, data_nd, sums, means, { means_event });

        result.set_cor_matrix(
            (homogen_table::wrap(corr.flatten(q, { corr_event }), column_count, column_count)));
    }
    if (desc.get_result_options().test(result_options::means)) {
        result.set_means(homogen_table::wrap(means.flatten(q, { means_event }), 1, column_count));
    }
    return result;
}

template <typename Float>
struct compute_kernel_gpu<Float, method_t, task_t> {
    result_t operator()(const bk::context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method_t, task_t>;
template struct compute_kernel_gpu<double, method_t, task_t>;

} // namespace oneapi::dal::covariance::backend
