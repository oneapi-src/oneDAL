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
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Type, typename Index, ndorder inp_order, ndorder out_order>
sycl::event select_indexed_rows(sycl::queue& q,
                                const ndview<Index, 1>& ids,
                                const ndview<Type, 2, inp_order>& src,
                                ndview<Type, 2, out_order>& dst,
                                const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
