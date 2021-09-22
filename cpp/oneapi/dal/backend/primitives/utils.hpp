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

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Type>
inline ndarray<Type, 2> table2ndarray(const table& table) {
    row_accessor<const Type> accessor{ table };
    const auto data = accessor.pull({ 0, -1 });
    return ndarray<Type, 2>::wrap(data, { table.get_row_count(), table.get_column_count() });
}

template <typename Type>
inline ndarray<Type, 1> table2ndarray_1d(const table& table) {
    row_accessor<const Type> accessor{ table };
    const auto data = accessor.pull({ 0, -1 });
    return ndarray<Type, 1>::wrap(data, { data.get_count() });
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename Type>
inline ndarray<Type, 2> table2ndarray(sycl::queue& q,
                                      const table& table,
                                      sycl::usm::alloc alloc = sycl::usm::alloc::shared) {
    row_accessor<const Type> accessor{ table };
    const auto data = accessor.pull(q, { 0, -1 }, alloc);
    return ndarray<Type, 2>::wrap(data, { table.get_row_count(), table.get_column_count() });
}

template <typename Type>
inline ndarray<Type, 1> table2ndarray_1d(sycl::queue& q,
                                         const table& table,
                                         sycl::usm::alloc alloc = sycl::usm::alloc::shared) {
    row_accessor<const Type> accessor{ table };
    const auto data = accessor.pull(q, { 0, -1 }, alloc);
    return ndarray<Type, 1>::wrap(data, { data.get_count() });
}
#endif

} // namespace oneapi::dal::backend::primitives
