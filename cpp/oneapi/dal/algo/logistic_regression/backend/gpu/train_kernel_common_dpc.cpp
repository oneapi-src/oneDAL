/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/logistic_regression/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/gpu/train_kernel_common.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/optimizer_impl.hpp"
#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/algo/logistic_regression/train_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/objective_function.hpp"
#include "oneapi/dal/backend/primitives/optimizers.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::logistic_regression::backend {

using dal::backend::context_gpu;

namespace be = dal::backend;
namespace pr = be::primitives;

template <typename Float, typename Task>
train_result<Task> call_dal_kernel(const context_gpu& ctx,
                                   const detail::descriptor_base<Task>& desc,
                                   const detail::train_parameters<Task>& params,
                                   const table& data,
                                   const table& resp) {
    using dal::detail::check_mul_overflow;

    auto queue = ctx.get_queue();

    ONEDAL_PROFILER_TASK(log_reg_train_kernel, queue);

    using model_t = model<Task>;
    using model_impl_t = detail::model_impl<Task>;

    auto opt_impl = detail::get_optimizer_impl(desc);

    if (!opt_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_optimizer() };
    }

    const auto sample_count = data.get_row_count();
    const auto feature_count = data.get_column_count();
    ONEDAL_ASSERT(sample_count == resp.get_row_count());
    const auto responses_nd =
        pr::table2ndarray_1d<std::int32_t>(queue, resp, sycl::usm::alloc::device);

    const std::int64_t bsize = params.get_gpu_macro_block();

    const Float l2 = Float(1.0) / desc.get_inverse_regularization();
    const bool fit_intercept = desc.get_compute_intercept();

    auto& comm = ctx.get_communicator();

    pr::logloss_function<Float> loss_func =
        pr::logloss_function(queue, comm, data, responses_nd, l2, fit_intercept, bsize);

    auto [x, fill_event] =
        pr::ndarray<Float, 1>::zeros(queue, { feature_count + 1 }, sycl::usm::alloc::device);

    pr::ndview<Float, 1> x_suf = fit_intercept ? x : x.slice(1, feature_count);

    auto [train_event, iter_num] = opt_impl->minimize(queue, loss_func, x_suf, { fill_event });

    auto all_coefs = homogen_table::wrap(x.flatten(queue, { train_event }), 1, feature_count + 1);

    const auto model_impl = std::make_shared<model_impl_t>(all_coefs);
    const auto model = dal::detail::make_private<model_t>(model_impl);

    const auto options = desc.get_result_options();
    auto result = train_result<Task>().set_model(model).set_result_options(options);

    if (options.test(result_options::intercept)) {
        ONEDAL_ASSERT(fit_intercept);
        table intercept_table =
            homogen_table::wrap(x.slice(0, 1).flatten(queue, { train_event }), 1, 1);
        result.set_intercept(intercept_table);
    }

    if (options.test(result_options::coefficients)) {
        auto coefs_array = x.slice(1, feature_count).flatten(queue, { train_event });
        auto coefs_table = homogen_table::wrap(coefs_array, 1, feature_count);
        result.set_coefficients(coefs_table);
    }

    if (options.test(result_options::iterations_count)) {
        result.set_iterations_count(iter_num);
    }

    if (options.test(result_options::inner_iterations_count)) {
        result.set_inner_iterations_count(opt_impl->get_inner_iter());
    }

    return result;
}

template train_result<task::classification> call_dal_kernel<float, task::classification>(
    const context_gpu&,
    const detail::descriptor_base<task::classification>&,
    const detail::train_parameters<task::classification>&,
    const table&,
    const table&);
template train_result<task::classification> call_dal_kernel<double, task::classification>(
    const context_gpu&,
    const detail::descriptor_base<task::classification>&,
    const detail::train_parameters<task::classification>&,
    const table&,
    const table&);

} // namespace oneapi::dal::logistic_regression::backend
