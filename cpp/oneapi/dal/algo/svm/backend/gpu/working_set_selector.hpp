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

#pragma once

#include "oneapi/dal/algo/svm/backend/gpu/misc.hpp"
#include "oneapi/dal/backend/primitives/selection/select_flagged.hpp"

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;

inline std::int64_t propose_working_set_size(const sycl::queue& queue,
                                             const std::int64_t row_count) {
    const std::int64_t max_wg_size = dal::backend::device_max_wg_size(queue);
    return std::min(dal::backend::down_pow2<std::uint32_t>(row_count),
                    dal::backend::down_pow2<std::uint32_t>(max_wg_size));
}

template <typename Float>
class working_set_selector {
public:
    working_set_selector(const sycl::queue& queue,
                         const pr::ndarray<Float, 1>& labels,
                         const Float C,
                         const std::int64_t row_count)
            : queue_(queue),
              row_count_(row_count),
              C_(C),
              labels_(labels) {
        ONEDAL_ASSERT(row_count > 0);
        ONEDAL_ASSERT(row_count <= dal::detail::limits<std::uint32_t>::max());
        auto [indicator, indicator_event] =
            pr::ndarray<std::uint8_t, 1>::zeros(queue_, { row_count_ }, sycl::usm::alloc::device);
        sorted_f_indices_ =
            pr::ndarray<std::uint32_t, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);

        ws_count_ = propose_working_set_size(queue_, row_count);

        values_sort_ =
            pr::ndarray<Float, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);
        buff_indices_ =
            pr::ndarray<std::uint32_t, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);

        indicator_event.wait_and_throw();
        indicator_ = indicator;
    }

    sycl::event select(const pr::ndview<Float, 1>& alpha,
                       const pr::ndview<Float, 1>& f,
                       pr::ndview<std::uint32_t, 1>& ws_indices,
                       const std::uint32_t iteration_index,
                       const dal::backend::event_vector& deps = {}) {
        ONEDAL_ASSERT(labels_.get_dimension(0) == alpha.get_dimension(0));
        ONEDAL_ASSERT(labels_.get_dimension(0) == f.get_dimension(0));
        ONEDAL_ASSERT(alpha.get_dimension(0) == f.get_dimension(0));
        ONEDAL_ASSERT(ws_indices.get_dimension(0) == ws_count_);
        ONEDAL_ASSERT(ws_indices.has_mutable_data());

        std::int64_t left_to_select = ws_count_;
        std::int64_t selected_count = 0;
        sycl::event event;

        sycl::event::wait_and_throw(deps);

        if (iteration_index > 0) {
            std::tie(event, selected_count) = copy_last_to_first(ws_indices);
            left_to_select -= selected_count;
        }

        event = arg_sort(queue_, f, values_sort_, sorted_f_indices_, row_count_, { event });

        const std::int64_t need_select_up = (ws_count_ - selected_count) / 2;
        std::tie(event, selected_count) = select_ws_edge(alpha,
                                                         ws_indices,
                                                         need_select_up,
                                                         left_to_select,
                                                         ws_edge::up,
                                                         { event });
        left_to_select -= selected_count;

        const std::int64_t need_select_count_low = ws_count_ - selected_count;
        std::tie(event, selected_count) = select_ws_edge(alpha,
                                                         ws_indices,
                                                         need_select_count_low,
                                                         left_to_select,
                                                         ws_edge::low,
                                                         { event });
        left_to_select -= selected_count;

        if (left_to_select > 0) {
            std::tie(event, selected_count) = select_ws_edge(alpha,
                                                             ws_indices,
                                                             left_to_select,
                                                             left_to_select,
                                                             ws_edge::up,
                                                             { event });
            left_to_select -= selected_count;
        }
        ONEDAL_ASSERT(left_to_select == 0);

        return event;
    }

private:
    sycl::event reset_indicator(const pr::ndview<std::uint32_t, 1>& idx,
                                pr::ndview<std::uint8_t, 1>& indicator,
                                const std::int64_t n,
                                const dal::backend::event_vector& deps = {}) {
        ONEDAL_ASSERT(idx.get_dimension(0) == ws_count_);
        ONEDAL_ASSERT(indicator.get_dimension(0) == row_count_);
        ONEDAL_ASSERT(indicator.has_mutable_data());

        const std::uint32_t* idx_ptr = idx.get_data();
        std::uint8_t* indicator_ptr = indicator.get_mutable_data();

        const auto wg_size = std::min(dal::backend::propose_wg_size(queue_), n);
        const auto range = dal::backend::make_multiple_nd_range_1d(n, wg_size);

        auto reset_indicator_event = queue_.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                const std::uint32_t i = item.get_global_id(0);
                indicator_ptr[idx_ptr[i]] = 0;
            });
        });

        return reset_indicator_event;
    }

    std::tuple<sycl::event, const std::int64_t> select_ws_edge(
        const pr::ndview<Float, 1>& alpha,
        pr::ndview<std::uint32_t, 1>& ws_indices,
        const std::int64_t need_select_count,
        const std::int64_t left_to_select,
        ws_edge edge,
        const dal::backend::event_vector& deps = {}) {
        auto select_ws_edge_event =
            check_ws_edge(queue_, labels_, alpha, indicator_, C_, row_count_, edge, deps);

        const std::int64_t already_selected = ws_count_ - left_to_select;

        /* Reset indicator for busy Indices */
        if (already_selected > 0) {
            select_ws_edge_event =
                reset_indicator(ws_indices, indicator_, already_selected, { select_ws_edge_event });
        }
        std::int64_t select_flagged_count = 0;
        auto select_flagged = pr::select_flagged_index<std::uint32_t, std::uint8_t>{ queue_ };
        select_ws_edge_event = select_flagged(indicator_,
                                              sorted_f_indices_,
                                              buff_indices_,
                                              select_flagged_count,
                                              { select_ws_edge_event });

        const std::int64_t select_count = std::min(select_flagged_count, need_select_count);

        std::uint32_t* ws_indices_ptr = ws_indices.get_mutable_data();
        const std::uint32_t* buff_indices_ptr = buff_indices_.get_data();

        if (select_count > 0) {
            std::int64_t offset = 0;
            if (edge == ws_edge::low) {
                offset = select_flagged_count - select_count;
            }
            select_ws_edge_event = dal::backend::copy(queue_,
                                                      ws_indices_ptr + already_selected,
                                                      buff_indices_ptr + offset,
                                                      select_count,
                                                      { select_ws_edge_event });
        }

        return { select_ws_edge_event, select_count };
    }

    sycl::event check_ws_edge(sycl::queue& queue,
                              const pr::ndview<Float, 1>& y,
                              const pr::ndview<Float, 1>& alpha,
                              pr::ndview<std::uint8_t, 1>& indicator,
                              const Float C,
                              const std::int64_t n,
                              ws_edge edge,
                              const dal::backend::event_vector& deps = {}) {
        ONEDAL_ASSERT(y.get_dimension(0) == n);
        ONEDAL_ASSERT(alpha.get_dimension(0) == n);
        ONEDAL_ASSERT(indicator.get_dimension(0) == n);
        ONEDAL_ASSERT(indicator.has_mutable_data());

        const Float* y_ptr = y.get_data();
        const Float* alpha_ptr = alpha.get_data();
        std::uint8_t* indicator_ptr = indicator.get_mutable_data();

        const auto wg_size = std::min(dal::backend::propose_wg_size(queue), n);
        const auto range = dal::backend::make_multiple_nd_range_1d(n, wg_size);

        sycl::event check_event;

        if (edge == ws_edge::up) {
            check_event = queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(deps);

                cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                    const std::uint32_t i = item.get_global_id(0);
                    indicator_ptr[i] = is_upper_edge<Float>(y_ptr[i], alpha_ptr[i], C);
                });
            });
        }
        else {
            check_event = queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on(deps);

                cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                    const std::uint32_t i = item.get_global_id(0);
                    indicator_ptr[i] = is_lower_edge<Float>(y_ptr[i], alpha_ptr[i], C);
                });
            });
        }
        return check_event;
    }

    std::tuple<sycl::event, const std::int64_t> copy_last_to_first(
        pr::ndview<std::uint32_t, 1>& ws_indices,
        const dal::backend::event_vector& deps = {}) {
        ONEDAL_ASSERT(ws_indices.get_dimension(0) == ws_count_);
        ONEDAL_ASSERT(ws_indices.has_mutable_data());
        const std::int64_t q = ws_count_ / 2;
        std::uint32_t* ws_indices_ptr = ws_indices.get_mutable_data();
        auto copy_event =
            dal::backend::copy(queue_, ws_indices_ptr, ws_indices_ptr + q, ws_count_ - q, deps);
        const std::int64_t selected_count = q;
        return { copy_event, selected_count };
    }

    sycl::queue queue_;

    std::int64_t row_count_;
    std::int64_t ws_count_;
    Float C_;

    pr::ndarray<std::uint32_t, 1> sorted_f_indices_;
    pr::ndarray<std::uint32_t, 1> buff_indices_;
    pr::ndarray<std::uint8_t, 1> indicator_;
    pr::ndarray<Float, 1> values_sort_;
    pr::ndarray<Float, 1> labels_;
};

#define INSTANTIATE_WORKING_SET(F) template class working_set_selector<F>;

INSTANTIATE_WORKING_SET(float);
INSTANTIATE_WORKING_SET(double);

} // namespace oneapi::dal::svm::backend
