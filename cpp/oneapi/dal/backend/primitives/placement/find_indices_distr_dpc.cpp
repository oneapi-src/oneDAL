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

#include "oneapi/dal/array.hpp"

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

/*template <typename Float, typename Index>
sycl::event fix_indices(sycl::queue& queue,
                        const std::int64_t rank,
                        const std::int64_t offset,
                        const ndview<Float, 1>& points,
                        const ndview<Float, 1>& bounds,
                        ndview<Index, 1>& indices,
                        const event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(fix_indices.kernel, queue);

    ONEDAL_ASSERT(points.has_data());
    ONEDAL_ASSERT(bounds.has_data());
    ONEDAL_ASSERT(indices.has_mutable_data());

    const auto count = points.get_count();
    const auto rank_count = bounds.get_count()
    ONEDAL_ASSERT(count == indices.get_count());

    const auto range = make_range_1d(count);
    const Float* const pts_ptr = points.get_data();
    const Float* const bds_ptr = bounds.get_data();
    Float* const ids_ptr = indices.get_mutable_data();

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        h.parallel_for(range, [=](sycl::id<1> idx) {
            const auto value = pts_ptr[idx];
            const auto index = ids_ptr[idx];

            const auto curr_upper = bds_ptr[rank];
            const auto curr_lower = (rank == std::int64_t(0)) ? 
                std::numeric_limits<Float>::lowest() : bds_ptr[rank - 1];

            const bool not_this_rank = (value < curr_lower) || (curr_upper <= value); 

            ids_ptr[idx] = not_this_rank ? (index + offset) : std::numeric_limits<Index>::lowest();
        });
    });
}

template <typename Float, typename Index>
sycl::event fix_indices(sycl::queue& queue,
                        communicator& comm,
                        const std::int64_t offset,
                        const ndview<Float, 1>& points,
                        const ndview<Float, 1>& bounds,
                        ndview<Index, 1>& indices,
                        const event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(fix_indices, queue);

    const auto rank = comm.get_rank();
    const auto rank_count = comm.get_rank_count();
    ONEDAL_ASSERT(bounds.get_count() == rank_count);
    auto event = fix_indices(queue, rank, offset, points, bounds, indices, deps);

    {
        ONEDAL_PROFILER_TASK(fix_indices, queue);
        sycl::event::wait_and_throw({ event });

        const auto ids_arr = array<Float>::wrap(queue, 
            indices.get_mutable_data(), indices.get_count());

        comm.allreduce(ids_arr, reduce_op::max).wait();

        return sycl::event{};
    }
}

template <typename Float, typename Index>
sycl::event find_indices(sycl::queue& queue,
                         communicator& comm,
                         const ndview<Float, 1>& points,
                         ndview<Float, 1>& values,
                         ndview<Float, 1>& bounds,
                         ndview<Index, 1>& indices,
                         const event_vector& deps) {
    const auto rank = comm.get_rank();
    const auto rank_count = comm.get_rank_count();

    ONEDAL_ASSERT(bounds.has_mutable_data());
    ONEDAL_ASSERT(rank_count + 1 == bounds.get_count());

    ONEDAL_ASSERT(values.has_mutable_data());
    const auto local_count = values.get_count();

    {
        sycl::event::wait_and_throw(deps);

        ONEDAL_PROFILER_TASK(find_indices.boundaries, queue);

        auto bnds_arr = array<Float>::wrap(queue, bounds.get_mutable_data(), rank_count);
        const Float* const last_val_ptr = values.get_data() + local_count - 1;
        auto last_val_arr = array<Float>::wrap(queue, last_val_ptr, 1);
        comm.allgather(last_val_arr, bnds_arr).wait();
    }

    constexpr auto alignment = search_alignment::left;
    auto search_event = search_sorted_1d<alignment>(queue, data, points);

    return fix_indices(queue, rank, points, bounds, indices, { search_event });
}*/

} // namespace oneapi::dal::backend::primitives
