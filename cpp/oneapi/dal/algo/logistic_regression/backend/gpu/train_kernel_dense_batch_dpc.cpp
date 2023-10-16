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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/algo/logistic_regression/train_types.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/gpu/optimizers.hpp"
#include "oneapi/dal/backend/primitives/objective_function.hpp"
#include "oneapi/dal/backend/primitives/optimizers.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/optimizer_impl.hpp"

namespace oneapi::dal::logistic_regression::backend {

using dal::backend::context_gpu;

namespace be = dal::backend;
namespace pr = be::primitives;

template <typename Float, typename Task>
static train_result<Task> call_dal_kernel(const context_gpu& ctx,
                                          const detail::descriptor_base<Task>& desc,
                                          const detail::train_parameters<Task>& params,
                                          const table& data,
                                          const table& resp) {
    using dal::detail::check_mul_overflow;

    using model_t = model<Task>;
    using model_impl_t = detail::model_impl<Task>;

    auto& queue = ctx.get_queue();

    ONEDAL_PROFILER_TASK(log_reg_train_kernel, queue);

    const auto sample_count = data.get_row_count();
    const auto feature_count = data.get_column_count();
    ONEDAL_ASSERT(sample_count == resp.get_row_count());
    const auto responses_nd =
        pr::table2ndarray_1d<std::int32_t>(queue, resp, sycl::usm::alloc::device);

    const std::int64_t bsize = params.get_gpu_macro_block();

    const Float L2 = desc.get_l2_coef();
    const bool fit_intercept = desc.get_compute_intercept();
    //const Float tol = desc.get_tol();
    //const std::int64_t maxiter = desc.get_max_iter();

    // TODO: add check if the dataset can be moved to gpu
    // Move data to gpu
    pr::ndarray<Float, 2> data_nd = pr::table2ndarray<Float>(queue, data, sycl::usm::alloc::device);
    table data_gpu = homogen_table::wrap(data_nd.flatten(queue, {}), sample_count, feature_count);

    pr::LogLossFunction<Float> loss_func =
        pr::LogLossFunction(queue, data_gpu, responses_nd, L2, fit_intercept, bsize);

    auto opt_impl = detail::get_optimizer_impl(desc);

    auto [x, fill_event] =
        pr::ndarray<Float, 1>::zeros(queue, { feature_count + 1 }, sycl::usm::alloc::device);

    pr::ndview<Float, 1> x_suf;

    x_suf = fit_intercept ? x : x.slice(1, feature_count);

    sycl::event train_event;

    if (opt_impl->get_optimizer_type() == detail::optimizer_type::newton_cg) {
        train_event = pr::newton_cg(queue,
                                    loss_func,
                                    x_suf,
                                    Float(opt_impl->get_tol()),
                                    opt_impl->get_max_iter(),
                                    { fill_event });
    }

    //sycl::event train_event = opt_impl->minimize(queue, loss_func, x_suf, { fill_event });

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
        auto coefs_table =
            homogen_table::wrap(x.slice(1, feature_count).flatten(queue, { train_event }),
                                1,
                                feature_count);
        result.set_coefficients(coefs_table);
    }

    return result;
}

template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const detail::train_parameters<Task>& params,
                                const train_input<Task>& input) {
    return call_dal_kernel<Float, Task>(ctx, desc, params, input.get_data(), input.get_responses());
}

template <typename Float, typename Task>
struct train_kernel_gpu<Float, method::dense_batch, Task> {
    train_result<Task> operator()(const context_gpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const detail::train_parameters<Task>& params,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, params, input);
    }
};

template struct train_kernel_gpu<float, method::dense_batch, task::classification>;
template struct train_kernel_gpu<double, method::dense_batch, task::classification>;

} // namespace oneapi::dal::logistic_regression::backend
