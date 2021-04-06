/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/rbf_kernel/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/math.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::rbf_kernel::backend {

using dal::backend::context_gpu;
using input_t = compute_input<task::compute>;
using result_t = compute_result<task::compute>;
using descriptor_t = detail::descriptor_base<task::compute>;

namespace pr = dal::backend::primitives;

template <typename Float>
sycl::event compute_rbf(sycl::queue& queue,
                        const pr::ndview<Float, 1>& sqr_x,
                        const pr::ndview<Float, 1>& sqr_y,
                        pr::ndview<Float, 2> res_rbf,
                        const Float coeff,
                        const std::int64_t ld,
                        const dal::backend::event_vector& deps) {
    const Float* sqr_x_ptr = sqr_x.get_data();
    const Float* sqr_y_ptr = sqr_y.get_data();
    Float* res_rbf_ptr = res_rbf.get_mutable_data();

    const Float threshold = dal::backend::exp_treshold<Float>();

    auto compute_rbf_event = queue.submit([&](sycl::handler& chg) {
        chg.depends_on(deps);

        const auto g_0 = dal::detail::integral_cast<std::size_t>(sqr_x.get_dimension(0));
        const auto g_1 = dal::detail::integral_cast<std::size_t>(sqr_y.get_dimension(0));
        const auto range = sycl::range<2>(g_0, g_1);

        chg.parallel_for(range, [=](sycl::id<2> id) {
            const int i = id[0];
            const int j = id[1];
            const Float sqr_x_i = sqr_x_ptr[i];
            const Float sqr_y_j = sqr_y_ptr[j];
            const Float res_rbf_ij = res_rbf_ptr[i * ld + j];
            const Float arg = sycl::fmax((sqr_x_i + sqr_y_j + res_rbf_ij) * coeff, threshold);

            res_rbf_ptr[i * ld + j] = sycl::exp(arg);
        });
    });

    return compute_rbf_event;
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    const auto x = input.get_x();
    const auto y = input.get_y();

    auto& queue = ctx.get_queue();

    const int64_t row_count_x = x.get_row_count();
    const int64_t col_count_x = x.get_column_count();
    const int64_t row_count_y = y.get_row_count();
    const int64_t col_count_y = y.get_column_count();

    ONEDAL_ASSERT(col_count_x == col_count_y);
    dal::detail::check_mul_overflow(row_count_x, row_count_y);

    const double sigma = desc.get_sigma();
    const Float coeff = static_cast<Float>(-0.5 / (sigma * sigma));

    auto arr_x = row_accessor<const Float>(x).pull(queue, { 0, -1 }, sycl::usm::alloc::device);
    const auto ndarray_x = pr::ndarray<Float, 2>::wrap(arr_x, { row_count_x, col_count_x });

    auto arr_y = row_accessor<const Float>(y).pull(queue, { 0, -1 }, sycl::usm::alloc::device);
    const auto ndarray_y = pr::ndarray<Float, 2>::wrap(arr_y, { row_count_y, col_count_y });

    auto ndarray_res =
        pr::ndarray<Float, 2>::empty(queue, { row_count_x, row_count_y }, sycl::usm::alloc::device);

    auto ndarray_sqr_x =
        pr::ndarray<Float, 1>::empty(queue, { row_count_x }, sycl::usm::alloc::device);
    auto ndarray_sqr_y =
        pr::ndarray<Float, 1>::empty(queue, { row_count_y }, sycl::usm::alloc::device);

    auto reduce_x_event =
        reduce_by_rows(queue, ndarray_x, ndarray_sqr_x, pr::sum<Float>{}, pr::square<Float>{});
    auto reduce_y_event =
        reduce_by_rows(queue, ndarray_y, ndarray_sqr_y, pr::sum<Float>{}, pr::square<Float>{});

    constexpr Float alpha = -2.0;
    constexpr Float beta = 0.0;
    auto gemm_event = gemm(queue, ndarray_x, ndarray_y.t(), ndarray_res, alpha, beta);

    compute_rbf(queue,
                ndarray_sqr_x,
                ndarray_sqr_y,
                ndarray_res,
                coeff,
                row_count_y,
                { reduce_x_event, reduce_y_event, gemm_event })
        .wait_and_throw();

    return result_t{}.set_values(dal::detail::homogen_table_builder{}
                                     .reset(ndarray_res.flatten(queue), row_count_x, row_count_y)
                                     .build());
}

template <typename Float>
struct compute_kernel_gpu<Float, method::dense, task::compute> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::dense, task::compute>;
template struct compute_kernel_gpu<double, method::dense, task::compute>;

} // namespace oneapi::dal::rbf_kernel::backend
