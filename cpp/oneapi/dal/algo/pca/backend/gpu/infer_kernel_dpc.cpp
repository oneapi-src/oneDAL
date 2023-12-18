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

#include "oneapi/dal/algo/pca/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/pca/backend/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/stat.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"

namespace oneapi::dal::pca::backend {

namespace pr = oneapi::dal::backend::primitives;
namespace bk = oneapi::dal::backend;
using dal::backend::context_gpu;
using alloc = sycl::usm::alloc;

using model_t = model<task::dim_reduction>;
using input_t = infer_input<task::dim_reduction>;
using result_t = infer_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;

template <typename Float>
auto get_centered(sycl::queue& q,
                  pr::ndview<Float, 2>& data,
                  const pr::ndview<Float, 1>& means,
                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_centered_data, q);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto centered_data_ptr = data.get_mutable_data();
    auto means_ptr = means.get_data();

    auto centered_event = q.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(row_count, column_count);
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t i = id[0];
            const std::int64_t j = id[1];
            centered_data_ptr[i * column_count + j] =
                centered_data_ptr[i * column_count + j] - means_ptr[j];
        });
    });
    return centered_event;
}

template <typename Float>
auto get_scaled(sycl::queue& q,
                pr::ndview<Float, 2>& data,
                const pr::ndview<Float, 1>& variances,
                const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_scaled_data, q);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto scaled_data_ptr = data.get_mutable_data();
    auto variances_ptr = variances.get_data();

    auto scaled_event = q.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(row_count, column_count);
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t i = id[0];
            const std::int64_t j = id[1];
            Float sqrt_var = sycl::sqrt(variances_ptr[j]);
            Float inv_var = sqrt_var == 0 ? 0 : 1 / sqrt_var;
            scaled_data_ptr[i * column_count + j] = scaled_data_ptr[i * column_count + j] * inv_var;
        });
    });
    return scaled_event;
}

template <typename Float>
auto get_whitened(sycl::queue& q,
                  pr::ndview<Float, 2>& data,
                  const pr::ndview<Float, 1>& eigenvalues,
                  const bk::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(compute_whitened_data, q);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t column_count = data.get_dimension(1);

    auto whitened_data_ptr = data.get_mutable_data();
    auto eigenvalues_ptr = eigenvalues.get_data();

    auto whitened_event = q.submit([&](sycl::handler& h) {
        const auto range = bk::make_range_2d(row_count, column_count);
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> id) {
            const std::int64_t i = id[0];
            const std::int64_t j = id[1];
            Float sqrt_eigenvalue = sycl::sqrt(eigenvalues_ptr[j]);
            Float inv_eigenvalue = sqrt_eigenvalue == 0 ? 0 : 1 / sqrt_eigenvalue;
            whitened_data_ptr[i * column_count + j] =
                whitened_data_ptr[i * column_count + j] * inv_eigenvalue;
        });
    });
    return whitened_event;
}

template <typename Float>
static result_t infer(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& queue = ctx.get_queue();

    const auto data = input.get_data();
    const auto model = input.get_model();
    const auto eigenvectors = model.get_eigenvectors();
    const auto eigenvalues = model.get_eigenvalues();

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t component_count = get_component_count(desc, data);
    const std::int64_t column_count = data.get_column_count();

    dal::detail::check_mul_overflow(row_count, component_count);

    const auto data_nd = pr::table2ndarray<Float>(queue, data, alloc::device);
    auto data_to_xtx =
        pr::ndarray<Float, 2>::empty(queue, { row_count, column_count }, alloc::device);

    auto copy_event = copy(queue, data_to_xtx, data_nd, {});
    copy_event.wait_and_throw();

    sycl::event mean_centered_event;
    if (desc.do_mean_centering() && model.get_means().has_data()) {
        const auto means = model.get_means();
        const auto means_nd = pr::table2ndarray_1d<Float>(queue, means, alloc::device);
        mean_centered_event = get_centered(queue, data_to_xtx, means_nd, { copy_event });
    }

    sycl::event scaled_event;
    if (desc.do_scale() && model.get_variances().has_data()) {
        const auto variances = model.get_variances();
        const auto variances_nd = pr::table2ndarray_1d<Float>(queue, variances, alloc::device);
        scaled_event = get_scaled(queue, data_to_xtx, variances_nd, { mean_centered_event });
    }

    const auto eigenvectors_nd = pr::table2ndarray<Float>(queue, eigenvectors, alloc::device);

    auto res_nd =
        pr::ndarray<Float, 2>::empty(queue, { row_count, component_count }, alloc::device);
    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, queue);
        gemm_event = pr::gemm(queue,
                              data_to_xtx,
                              eigenvectors_nd.t(),
                              res_nd,
                              Float(1.0),
                              Float(0.0),
                              { scaled_event });
        gemm_event.wait_and_throw();
    }

    sycl::event whiten_event;
    if (desc.whiten() && model.get_eigenvalues().has_data()) {
        const auto eigenvalues = model.get_eigenvalues();
        const auto eigenvalues_nd = pr::table2ndarray_1d<Float>(queue, eigenvalues, alloc::device);
        whiten_event = get_whitened(queue, res_nd, eigenvalues_nd, { gemm_event });
    }

    return result_t{}.set_transformed_data(
        homogen_table::wrap(res_nd.flatten(queue, { whiten_event }), row_count, component_count));
}

template <typename Float>
struct infer_kernel_gpu<Float, task::dim_reduction> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return infer<Float>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, task::dim_reduction>;
template struct infer_kernel_gpu<double, task::dim_reduction>;

} // namespace oneapi::dal::pca::backend
