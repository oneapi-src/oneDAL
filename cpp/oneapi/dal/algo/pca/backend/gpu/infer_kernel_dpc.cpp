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

using model_t = model<task::dim_reduction>;
using input_t = infer_input<task::dim_reduction>;
using result_t = infer_result<task::dim_reduction>;
using descriptor_t = detail::descriptor_base<task::dim_reduction>;
template <typename Float>
inline auto make_mutable(sycl::queue& q, pr::ndarray<Float, 2> data)
    -> std::tuple<sycl::event, pr::ndarray<Float, 2>> {
    constexpr auto alloc = sycl::usm::alloc::device;

    if (data.has_mutable_data()) {
        sycl::event event{};
        return std::make_tuple(std::move(event), std::move(data));
    }
    else {
        auto place = pr::ndarray<Float, 2>::empty(q,
                                                  { data.get_dimension(0), data.get_dimension(1) },
                                                  alloc);
        auto event = pr::copy(q, place, data, {});
        return std::make_tuple(std::move(event), std::move(place));
    }
}

template <typename Float>
inline auto get_centered(sycl::queue& q, const table& data, const table& means)
    -> std::tuple<sycl::event, pr::ndarray<Float, 2>> {
    constexpr auto alloc = sycl::usm::alloc::device;

    auto raw_data = pr::table2ndarray<Float>(q, data, alloc);

    if (means.has_data()) {
        auto [mut_event, mut_data] = make_mutable(q, std::move(raw_data));
        auto raw_means = pr::table2ndarray_1d<Float>(q, means, alloc);
        sycl::event new_mut_event = std::move(mut_event);
        pr::ndarray<Float, 2> new_mut_data = std::move(mut_data);
        auto event = q.submit([&](sycl::handler& h) {
            h.depends_on({ new_mut_event });

            const auto row_count = new_mut_data.get_dimension(0);
            const auto col_count = new_mut_data.get_dimension(1);
            const auto stride = new_mut_data.get_leading_stride();

            const auto* const mean_ptr = raw_means.get_data();
            auto* const data_ptr = new_mut_data.get_mutable_data();

            auto range = bk::make_range_2d(row_count, col_count);
            h.parallel_for(range, [=](sycl::id<2> id) -> void {
                auto* const ptr = data_ptr + id[0] * stride + id[1];
                (*ptr) -= *(mean_ptr + id[1]);
            });
        });

        return std::make_tuple(std::move(event), std::move(new_mut_data));
    }
    else {
        sycl::event event{};
        return std::make_tuple(std::move(event), std::move(raw_data));
    }
}

template <typename Float>
inline auto get_whitened(sycl::queue& q,
                         pr::ndarray<Float, 2>& mut_data,
                         const table& eigenvalues,
                         sycl::event deps) -> std::tuple<sycl::event, pr::ndarray<Float, 2>> {
    constexpr auto alloc = sycl::usm::alloc::device;

    if (eigenvalues.has_data()) {
        auto raw_ev = pr::table2ndarray_1d<Float>(q, eigenvalues, alloc);
        auto event = q.submit([&](sycl::handler& h) {
            h.depends_on({ deps });
            const auto row_count = mut_data.get_dimension(0);
            const auto col_count = mut_data.get_dimension(1);
            const auto stride = mut_data.get_leading_stride();

            const auto* const ev_ptr = raw_ev.get_data();
            auto* const data_ptr = mut_data.get_mutable_data();

            auto range = bk::make_range_2d(row_count, col_count);
            h.parallel_for(range, [=](sycl::id<2> id) -> void {
                auto* const ptr = data_ptr + id[0] * stride + id[1];
                (*ptr) /= sycl::sqrt(*(ev_ptr + id[1]));
            });
        });

        return std::make_tuple(std::move(event), std::move(mut_data));
    }
    else {
        sycl::event event{};
        return std::make_tuple(std::move(event), std::move(mut_data));
    }
}
template <typename Float>
static result_t infer(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& queue = ctx.get_queue();
    const auto data = input.get_data();
    const auto model = input.get_model();
    const auto eigenvectors = model.get_eigenvectors();
    const auto means = model.get_means();
    const auto eigenvalues = model.get_eigenvalues();

    const std::int64_t row_count = data.get_row_count();
    const std::int64_t component_count = get_component_count(desc, data);

    dal::detail::check_mul_overflow(row_count, component_count);

    const auto data_nd = pr::table2ndarray<Float>(queue, data, sycl::usm::alloc::device);
    auto [mean_center_event, mean_centered_data_nd] = get_centered<Float>(queue, data, means);
    const auto eigenvectors_nd =
        pr::table2ndarray<Float>(queue, eigenvectors, sycl::usm::alloc::device);

    auto res_nd = pr::ndarray<Float, 2>::empty(queue,
                                               { row_count, component_count },
                                               sycl::usm::alloc::device);
    sycl::event gemm_event;
    {
        ONEDAL_PROFILER_TASK(gemm, queue);
        gemm_event = pr::gemm(queue,
                              mean_centered_data_nd,
                              eigenvectors_nd.t(),
                              res_nd,
                              Float(1.0),
                              Float(0.0),
                              { mean_center_event });
        gemm_event.wait_and_throw();
    }

    if (desc.whiten()) {
        auto [whiten_event, whitened_data_nd] =
            get_whitened<Float>(queue, res_nd, eigenvalues, { gemm_event });
        const auto res_array = whitened_data_nd.flatten(queue, { whiten_event });
        const auto res_table = homogen_table::wrap(res_array, row_count, component_count);
        return result_t{}.set_transformed_data(res_table);
    }
    else {
        const auto res_array = res_nd.flatten(queue, { gemm_event });
        auto res_table = homogen_table::wrap(res_array, row_count, component_count);
        return result_t{}.set_transformed_data(res_table);
    }
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
