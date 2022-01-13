/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
inline ndarray<Type, 2, ndorder::c> table2ndarray_rm(sycl::queue& q,
                                                     const table& table,
                                                     sycl::usm::alloc alloc) {
    constexpr auto order = ndorder::c;
    row_accessor<const Type> accessor{ table };
    const auto data = accessor.pull(q, { 0, -1 }, alloc);
    return ndarray<Type, 2, ndorder::c>::wrap(data, { table.get_row_count(), table.get_column_count() });
}

template <typename Type>
inline ndarray<Type, 2, ndorder::f> table2ndarray_cm(sycl::queue& q,
                                                     const table& table,
                                                     sycl::usm::alloc alloc) {
    constexpr auto order = ndorder::f;
    using arr_t = ndarray<Type, 2, order>;
    const auto row_count = table.get_row_count();
    const auto column_count = table.get_column_count();
    const Type* const ptr = dynamic_cast<const homogen_table&>(table).get_data();
    const auto init_arr = arr_t::wrap(ptr, { row_count, column_count });

    const auto context = q.get_context();
    const auto ptr_alloc = sycl::get_pointer_type(ptr, context);
    bool is_suitable_ptr = false;
    if(ptr_alloc == alloc) {
        if(ptr_alloc == sycl::usm::alloc::device) {
            const auto device = q.get_device();
            const auto ptr_device = sycl::get_pointer_device(ptr, context);
            if(ptr_device == device) is_suitable_ptr = true;
        } else {
            is_suitable_ptr = true;
        }
    }

    if(is_suitable_ptr) {
        return init_arr;
    } else {
        const auto resl_arr = arr_t::empty(q, {rc, cc}, alloc);
        ONEDAL_ASSERT(init_arr.get_count() == resl_arr.get_count());
        q.copy<Type>(resl_arr.get_mutable_data(),
                     init_arr.get_data(),
                     init_arr.get_count()).wait_and_throw();
        return resl_arr;
    }
}

/*template<ndorder out_order, typename Type, ndorder inp_order>
inline auto convert_to_layout(sycl::queue& q,
                              const ndview<Type, 2, inp_order>& inp,
                              sycl::usm::alloc alloc = sycl::usm::alloc::shared) {
    using out_t = ndview<Type, 2, out_order>;
    const auto rc = inp.get_dimension(0);
    const auto cc = inp.get_dimension(1);
    if constexpr (inp_order == out_order) {
        const auto calloc
        if()
    }

    auto result = out_t::empty(q, {rc, cc}, alloc);
    copy(q, out, inp).wait_
}*/

template <typename Type, ndorder order = ndorder::c>
inline ndarray<Type, 2, order> table2ndarray(sycl::queue& q,
                                             const table& table,
                                             sycl::usm::alloc alloc = sycl::usm::alloc::shared) {
    [[maybe_unused]] const auto layout = table.get_data_layout();
    if constexpr (order == ndorder::c) {
        ONEDAL_ASSERT(layout ==  decltype(layout)::row_major);
        return table2ndarray_rm(q, table, alloc);
    } else {
        ONEDAL_ASSERT(layout ==  decltype(layout)::column_major);
        return table2ndarray_cm(q, table, alloc);
    }
}

/*template <typename Type>
inline auto table2ndarray_variant(sycl::queue& q,
                                  const table& table,
                                  sycl::usm::alloc alloc) {
    ONEDAL_ASSERT(table.had_data());
    ONEDAL_ASSERT(table.get_kind() == homogen_table::kind());
    const auto data_layout = table.get_data_layout();
    using var1_t = ndarray<Type, 2, ndorder::c>;
    using var2_t = ndarray<Type, 2, ndorder::f>;
    std::variant<var1_t, var2_t> result;
    if(data_layout == decltype(data_layout)::row_major) {
        result = table2ndarray(q, table, alloc);
    }
    if(data_layout == decltype(data_layout)::col_major) {

    }
    return result;
}*/

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
