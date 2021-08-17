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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/search/search.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float, bool indices = true, bool distances = true>
class copy_callback : public callback_base<Float, copy_callback<Float, indices, distances>> {
    using this_t = copy_callback<Float, indices, distances>;
    friend class callback_base<Float, this_t>;

    static_assert(indices || distances);

public:
    using float_t = Float;

    copy_callback(sycl::queue& queue,
                  std::int64_t query_block,
                  ndview<std::int32_t, 2> out_indices = {},
                  ndview<Float, 2> out_distances = {});

protected:
    sycl::event run(std::int64_t qb_id,
                    const ndview<std::int32_t, 2>& inp_indices,
                    const ndview<Float, 2>& inp_distances,
                    const event_vector& deps = {});

private:
    sycl::queue& queue_;
    ndview<Float, 2> out_distances_;
    ndview<std::int32_t, 2> out_indices_;
    const uniform_blocking query_blocking_;
};

#endif

} // namespace oneapi::dal::backend::primitives
