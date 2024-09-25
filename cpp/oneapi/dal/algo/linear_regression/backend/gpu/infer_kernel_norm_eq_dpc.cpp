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

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/ndindexer.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train_types.hpp"
#include "oneapi/dal/algo/linear_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/linear_regression/backend/gpu/infer_kernel.hpp"

namespace oneapi::dal::linear_regression::backend {

using daal::services::Status;
using dal::backend::context_gpu;

namespace be = dal::backend;
namespace pr = be::primitives;
namespace interop = dal::backend::interop;

template <typename Float>
std::int64_t propose_block_size(const sycl::queue& q, std::int64_t f, std::int64_t r) {
    constexpr std::int64_t fsize = sizeof(Float);
    return 0x10000l * (8 / fsize);
}

template <typename Float, pr::ndorder layout>
inline sycl::event apply_betas(sycl::queue& q,
                               bool beta,
                               pr::ndview<Float, 2, layout>& y,
                               const pr::ndview<Float, 2>& betas,
                               const be::event_vector& deps = {}) {
    if (beta) {
        ONEDAL_ASSERT(betas.has_data());
        ONEDAL_ASSERT(y.has_mutable_data());

        const auto shape = y.get_shape();
        ONEDAL_ASSERT(shape.at(1) == betas.get_dimension(0));
        ONEDAL_ASSERT(std::int64_t(1) == betas.get_dimension(1));

        return q.submit([&](sycl::handler& h) {
            h.depends_on(deps);

            auto y_idx = make_ndindexer(y);
            auto b_idx = make_ndindexer(betas);

            const auto range = shape.to_range();

            h.parallel_for(range, [=](sycl::id<2> idx) {
                const auto r = idx[0];
                const auto c = idx[1];

                y_idx.at(r, c) += b_idx.at(c, 0);
            });
        });
    }
    else {
        sycl::event::wait_and_throw(deps);
        return sycl::event{};
    }
}

template <typename Float, typename Task>
static infer_result<Task> call_dal_kernel(const context_gpu& ctx,
                                          const detail::descriptor_base<Task>& desc,
                                          const table& infer,
                                          const model<Task>& m) {
    using dal::detail::check_mul_overflow;

    auto& queue = ctx.get_queue();
    ONEDAL_PROFILER_TASK(linreg_infer_kernel, queue);

    constexpr auto alloc = sycl::usm::alloc::device;

    constexpr Float zero = 0, one = 1;

    const auto& betas = m.get_packed_coefficients();

    const auto sample_count = infer.get_row_count();
    const auto response_count = betas.get_row_count();
    const auto feature_count = infer.get_column_count();
    const auto beta = desc.get_compute_intercept();
    ONEDAL_ASSERT((feature_count + 1) == betas.get_column_count());

    const auto resps_size = check_mul_overflow(sample_count, response_count);
    auto resps_arr = array<Float>::zeros(queue, resps_size, alloc);

    auto y = pr::ndarray<Float, 2>::wrap_mutable(resps_arr, { sample_count, response_count });

    const auto b_count = propose_block_size<Float>(queue, feature_count, response_count);
    const be::uniform_blocking blocking(sample_count, b_count);

    sycl::event last_event;

    row_accessor<const Float> x_accessor(infer);

    auto betas_ndarr = pr::table2ndarray<Float>(queue, betas, alloc);
    const auto core = betas_ndarr.get_col_slice(1, feature_count + 1);
    const auto intp = betas_ndarr.get_col_slice(0, 1);

    for (std::int64_t b = 0; b < blocking.get_block_count(); ++b) {
        const auto last = blocking.get_block_end_index(b);
        const auto first = blocking.get_block_start_index(b);

        const auto length = last - first;
        auto y_sub = y.get_row_slice(first, last);
        auto x_arr = x_accessor.pull(queue, { first, last }, alloc);
        auto x_sub = pr::ndarray<Float, 2>::wrap(x_arr, { length, feature_count });

        sycl::event gemm_event;
        {
            gemm_event = pr::gemm(queue, x_sub, core.t(), y_sub, one, zero, { last_event });
            gemm_event.wait_and_throw();
        }
        last_event = apply_betas(queue, beta, y_sub, intp, { gemm_event });
    }

    sycl::event::wait({ last_event });

    auto responses = homogen_table::wrap(resps_arr, sample_count, response_count);

    auto result = infer_result<Task>().set_responses(responses);

    return result;
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_gpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const infer_input<Task>& input) {
    return call_dal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float, typename Task>
struct infer_kernel_gpu<Float, method::norm_eq, Task> {
    infer_result<Task> operator()(const context_gpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, method::norm_eq, task::regression>;
template struct infer_kernel_gpu<double, method::norm_eq, task::regression>;

} // namespace oneapi::dal::linear_regression::backend
