/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/linear_kernel/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/primitives/blas/gemm.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::linear_kernel::backend {

using dal::backend::context_gpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace pr = dal::backend::primitives;

template <typename Float>
auto compute_linear(sycl::queue& queue,
                    const pr::ndarray<Float, 2>& x_nd,
                    const pr::ndarray<Float, 2>& y_nd,
                    pr::ndarray<Float, 2>& res_nd,
                    const descriptor_t& desc) {
    ONEDAL_ASSERT(x_nd.get_dimension(0) == res_nd.get_dimension(0));
    ONEDAL_ASSERT(y_nd.get_dimension(0) == res_nd.get_dimension(1));
    ONEDAL_ASSERT(x_nd.get_dimension(1) == y_nd.get_dimension(1));

    const Float scale = desc.get_scale();
    const Float shift = desc.get_shift();

    sycl::event fill_res_event;
    if (shift != Float(0)) {
        fill_res_event = res_nd.fill(queue, Float(1));
    }

    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(linear_kernel.gemm, queue);
        gemm_event = gemm(queue, x_nd, y_nd.t(), res_nd, scale, shift, { fill_res_event });
    }
    return gemm_event;
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& queue = ctx.get_queue();
    const auto x = input.get_x();
    const auto y = input.get_y();

    const std::int64_t x_row_count = x.get_row_count();
    const std::int64_t y_row_count = y.get_row_count();

    ONEDAL_ASSERT(x.get_column_count() == y.get_column_count());
    dal::detail::check_mul_overflow(x_row_count, y_row_count);

    const auto x_nd = pr::table2ndarray<Float>(queue, x, sycl::usm::alloc::device);

    const auto y_nd = pr::table2ndarray<Float>(queue, y, sycl::usm::alloc::device);

    auto res_nd =
        pr::ndarray<Float, 2>::empty(queue, { x_row_count, y_row_count }, sycl::usm::alloc::device);

    auto compute_linear_event = compute_linear(queue, x_nd, y_nd, res_nd, desc);

    const auto res_array = res_nd.flatten(queue, { compute_linear_event });
    auto res_table = homogen_table::wrap(res_array, x_row_count, y_row_count);

    return result_t{}.set_values(res_table);
}

template <typename Float>
struct compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void operator()(const context_gpu& ctx,
                    const descriptor_t& desc,
                    const table& x,
                    const table& y,
                    homogen_table& res) {
        ONEDAL_ASSERT(x.get_row_count() == res.get_row_count());
        ONEDAL_ASSERT(y.get_row_count() == res.get_column_count());
        ONEDAL_ASSERT(x.get_column_count() == y.get_column_count());

        auto& queue = ctx.get_queue();
        const auto x_nd = pr::table2ndarray<Float>(queue, x, sycl::usm::alloc::device);
        const auto y_nd = pr::table2ndarray<Float>(queue, y, sycl::usm::alloc::device);

        auto res_ptr = res.get_data<Float>();

        // Temporary workaround until the table_builder approach is ready
        auto res_nd = pr::ndarray<Float, 2>::wrap(const_cast<Float*>(res_ptr),
                                                  { res.get_row_count(), res.get_column_count() });

        compute_linear(queue, x_nd, y_nd, res_nd, desc).wait_and_throw();
    }
#endif
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::linear_kernel::backend
