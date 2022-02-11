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

#include "oneapi/dal/backend/primitives/search/copy_search_callback.hpp"

namespace oneapi::dal::backend::primitives {

template <typename F, bool i, bool d>
std::int64_t get_length(const ndview<F, 2>& dsts, ndview<std::int32_t, 2>& inds) {
    static_assert(i || d);
    return i ? inds.get_dimension(0) : dsts.get_dimension(0);
}

template <typename Float, bool indices, bool distances>
copy_callback<Float, indices, distances>::copy_callback(sycl::queue& queue,
                                                        std::int64_t qblock,
                                                        ndview<std::int32_t, 2> out_indices,
                                                        ndview<Float, 2> out_distances)
        : queue_(queue),
          out_distances_(out_distances),
          out_indices_(out_indices),
          query_blocking_(get_length<Float, indices, distances>(out_distances, out_indices),
                          qblock) {}

template <typename Float, bool indices, bool distances>
sycl::event copy_callback<Float, indices, distances>::run(
    std::int64_t qb_id,
    const ndview<std::int32_t, 2>& inp_indices,
    const ndview<Float, 2>& inp_distances,
    const event_vector& deps) {
    sycl::event ind_event, dst_event;

    const std::int64_t from = query_blocking_.get_block_start_index(qb_id);
    const std::int64_t to = query_blocking_.get_block_end_index(qb_id);

    if constexpr (indices) {
        auto out_block = out_indices_.get_row_slice(from, to);
        ind_event = copy(queue_, out_block, inp_indices, deps);
    }

    if constexpr (distances) {
        auto out_block = out_distances_.get_row_slice(from, to);
        dst_event = copy(queue_, out_block, inp_distances, deps);
    }

    sycl::event::wait_and_throw(deps + ind_event + dst_event);
    return sycl::event();
}

#define INSTANTIATE_FLOAT(F)                      \
    template class copy_callback<F, true, true>;  \
    template class copy_callback<F, true, false>; \
    template class copy_callback<F, false, true>;

INSTANTIATE_FLOAT(float);
INSTANTIATE_FLOAT(double);

#undef INSTANTIATE_FLOAT

} // namespace oneapi::dal::backend::primitives
