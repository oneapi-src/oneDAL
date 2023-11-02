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

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/objective_function.hpp"
#include "oneapi/dal/backend/primitives/ndindexer.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/algo/logistic_regression/train_types.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/model_impl.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/gpu/infer_kernel.hpp"

namespace oneapi::dal::logistic_regression::backend {

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

template <typename Float, typename Task>
static infer_result<Task> call_dal_kernel(const context_gpu& ctx,
                                          const detail::descriptor_base<Task>& desc,
                                          const table& infer,
                                          const model<Task>& m) {
    using dal::detail::check_mul_overflow;

    auto& queue = ctx.get_queue();
    ONEDAL_PROFILER_TASK(logreg_infer_kernel, queue);

    constexpr auto alloc = sycl::usm::alloc::device;

    const auto& betas = m.get_packed_coefficients();

    const auto sample_count = infer.get_row_count();
    const auto feature_count = infer.get_column_count();
    const bool fit_intercept = desc.get_compute_intercept();
    ONEDAL_ASSERT((feature_count + 1) == betas.get_column_count());
    ONEDAL_ASSERT(1 == betas.get_row_count());

    pr::ndarray<Float, 1> params = pr::table2ndarray_1d<Float>(queue, betas, alloc);
    pr::ndview<Float, 1> params_suf = fit_intercept ? params : params.slice(1, feature_count);

    pr::ndarray<Float, 1> probs = pr::ndarray<Float, 1>::empty(queue, { sample_count }, alloc);
    pr::ndarray<std::int32_t, 1> responses =
        pr::ndarray<std::int32_t, 1>::empty(queue, { sample_count }, alloc);

    const auto bsize = propose_block_size<Float>(queue, feature_count, 1);
    const be::uniform_blocking blocking(sample_count, bsize);

    row_accessor<const Float> x_accessor(infer);

    be::event_vector all_deps = {};

    for (std::int64_t b = 0; b < blocking.get_block_count(); ++b) {
        const auto last = blocking.get_block_end_index(b);
        const auto first = blocking.get_block_start_index(b);

        const auto length = last - first;

        auto probs_slice = probs.slice(first, length);
        auto resp_slice = responses.slice(first, length);
        auto x_rows = x_accessor.pull(queue, { first, last }, alloc);
        auto x_nd = pr::ndarray<Float, 2>::wrap(x_rows, { length, feature_count });

        sycl::event prob_event =
            pr::compute_probabilities(queue, params_suf, x_nd, probs_slice, fit_intercept, {});

        const auto* const prob_ptr = probs_slice.get_data();
        auto* const resp_ptr = resp_slice.get_mutable_data();

        sycl::event fill_resp_event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(prob_event);
            const auto range = be::make_range_1d(length);
            cgh.parallel_for(range, [=](sycl::id<1> idx) {
                resp_ptr[idx] = prob_ptr[idx] < 0.5 ? 0 : 1;
            });
        });
        all_deps.push_back(fill_resp_event);
    }
    be::wait_or_pass(all_deps).wait_and_throw();

    auto resp_table = homogen_table::wrap(responses.flatten(queue, {}), sample_count, 1);
    auto prob_table = homogen_table::wrap(probs.flatten(queue, {}), sample_count, 1);

    auto result = infer_result<Task>().set_responses(resp_table).set_probabilities(prob_table);

    return result;
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_gpu& ctx,
                                const detail::descriptor_base<Task>& desc,
                                const infer_input<Task>& input) {
    return call_dal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float, typename Task>
struct infer_kernel_gpu<Float, method::dense_batch, Task> {
    infer_result<Task> operator()(const context_gpu& ctx,
                                  const detail::descriptor_base<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, method::dense_batch, task::classification>;
template struct infer_kernel_gpu<double, method::dense_batch, task::classification>;

} // namespace oneapi::dal::logistic_regression::backend
