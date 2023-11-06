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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"
#include "oneapi/dal/algo/linear_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/linear_regression/backend/gpu/finalize_train_kernel.hpp"
#include "oneapi/dal/algo/linear_regression/backend/gpu/update_kernel.hpp"

namespace oneapi::dal::linear_regression::backend {

using dal::backend::context_gpu;

namespace be = dal::backend;
namespace pr = be::primitives;

template <typename Float, typename Task>
static train_result<Task> call_dal_kernel(const context_gpu& ctx,
                                          const detail::descriptor_base<Task>& desc,
                                          const detail::train_parameters<Task>& params,
                                          const partial_train_result<Task>& input) {
    using dal::detail::check_mul_overflow;

    using model_t = model<Task>;
    using model_impl_t = detail::model_impl<Task>;

    auto& queue = ctx.get_queue();

    const bool beta = desc.get_compute_intercept();
    ONEDAL_PROFILER_TASK(linreg_train_kernel, queue);

    constexpr auto uplo = pr::mkl::uplo::upper;
    constexpr auto alloc = sycl::usm::alloc::device;

    const auto response_count = input.get_partial_xty().get_row_count();
    const auto ext_feature_count = input.get_partial_xty().get_column_count();
    const auto feature_count = ext_feature_count - beta;

    const pr::ndshape<2> xtx_shape{ ext_feature_count, ext_feature_count };

    const auto xtx_nd =
        pr::table2ndarray<Float>(queue, input.get_partial_xtx(), sycl::usm::alloc::device);
    const auto xty_nd = pr::table2ndarray<Float, pr::ndorder::f>(queue,
                                                                 input.get_partial_xty(),
                                                                 sycl::usm::alloc::device);

    const pr::ndshape<2> betas_shape{ response_count, feature_count + 1 };

    const auto betas_size = check_mul_overflow(response_count, feature_count + 1);
    auto betas_arr = array<Float>::zeros(queue, betas_size, alloc);

    auto nxtx = pr::ndarray<Float, 2>::empty(queue, xtx_shape, alloc);
    auto nxty = pr::ndview<Float, 2>::wrap_mutable(betas_arr, betas_shape);
    auto solve_event = pr::solve_system<uplo>(queue, beta, xtx_nd, xty_nd, nxtx, nxty, {});
    sycl::event::wait_and_throw({ solve_event });

    auto betas = homogen_table::wrap(betas_arr, response_count, feature_count + 1);

    const auto model_impl = std::make_shared<model_impl_t>(betas);
    const auto model = dal::detail::make_private<model_t>(model_impl);

    const auto options = desc.get_result_options();
    auto result = train_result<Task>().set_model(model).set_result_options(options);

    if (options.test(result_options::intercept)) {
        auto arr = array<Float>::zeros(queue, response_count, alloc);
        auto dst = pr::ndview<Float, 2>::wrap_mutable(arr, { 1l, response_count });
        const auto src = nxty.get_col_slice(0l, 1l).t();

        pr::copy(queue, dst, src).wait_and_throw();

        auto intercept = homogen_table::wrap(arr, 1l, response_count);
        result.set_intercept(intercept);
    }

    if (options.test(result_options::coefficients)) {
        const auto size = check_mul_overflow(response_count, feature_count);

        auto arr = array<Float>::zeros(queue, size, alloc);
        const auto src = nxty.get_col_slice(1l, feature_count + 1);
        auto dst = pr::ndview<Float, 2>::wrap_mutable(arr, { response_count, feature_count });

        pr::copy(queue, dst, src).wait_and_throw();

        auto coefficients = homogen_table::wrap(arr, response_count, feature_count);
        result.set_coefficients(coefficients);
    }

    return result;
}

template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const detail::train_parameters<Task>& params,
                                const partial_train_result<Task>& input) {
    return call_dal_kernel<Float, Task>(ctx, desc, params, input);
}

template <typename Float, typename Task>
struct finalize_train_kernel_gpu<Float, method::norm_eq, Task> {
    train_result<Task> operator()(const context_gpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const detail::train_parameters<Task>& params,
                                  const partial_train_result<Task>& input) const {
        return train<Float, Task>(ctx, desc, params, input);
    }
};

template struct finalize_train_kernel_gpu<float, method::norm_eq, task::regression>;
template struct finalize_train_kernel_gpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend
