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

#include <type_traits>

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
class kselect_by_rows_base {
public:
    virtual ~kselect_by_rows_base() {}
    virtual sycl::event operator()(sycl::queue& queue,
                                   const ndview<Float, 2>& data,
                                   std::int64_t k,
                                   ndview<Float, 2>& selection,
                                   ndview<std::int32_t, 2>& column_indices,
                                   const event_vector& deps = {}) = 0;

    virtual sycl::event operator()(sycl::queue& queue,
                                   const ndview<Float, 2>& data,
                                   std::int64_t k,
                                   ndview<Float, 2>& selection,
                                   const event_vector& deps = {}) = 0;

    virtual sycl::event operator()(sycl::queue& queue,
                                   const ndview<Float, 2>& data,
                                   std::int64_t k,
                                   ndview<std::int32_t, 2>& column_indices,
                                   const event_vector& deps = {}) = 0;
};
#endif

} // namespace oneapi::dal::backend::primitives
