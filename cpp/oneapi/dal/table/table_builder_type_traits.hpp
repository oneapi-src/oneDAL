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

#pragma once

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal {

template <typename T>
struct is_table_builder_impl {
    ONEAPI_DAL_SIMPLE_HAS_METHOD_TRAIT(table, build, ())

    static constexpr bool value = has_method_build_v<T>;
};

template <typename T>
inline constexpr bool is_table_builder_impl_v = is_table_builder_impl<T>::value;

template <typename T>
struct is_homogen_table_builder_impl {
    ONEAPI_DAL_SIMPLE_HAS_METHOD_TRAIT(homogen_table, build, ())
    ONEAPI_DAL_HAS_METHOD_TRAIT(void, reset, (homogen_table && t), reset_from_table)
    ONEAPI_DAL_HAS_METHOD_TRAIT(void,
                                reset,
                                (const array<byte_t>& data,
                                 std::int64_t row_count,
                                 std::int64_t column_count),
                                reset_from_array)
    ONEAPI_DAL_SIMPLE_HAS_METHOD_TRAIT(void, set_data_type, (data_type dt))
    ONEAPI_DAL_SIMPLE_HAS_METHOD_TRAIT(void, set_feature_type, (feature_type ft))
    ONEAPI_DAL_SIMPLE_HAS_METHOD_TRAIT(void,
                                       allocate,
                                       (std::int64_t row_count, std::int64_t column_count))
    ONEAPI_DAL_SIMPLE_HAS_METHOD_TRAIT(void, set_layout, (homogen_data_layout layout))
    ONEAPI_DAL_SIMPLE_HAS_METHOD_TRAIT(void,
                                       copy_data,
                                       (const void* data,
                                        std::int64_t row_count,
                                        std::int64_t column_count))

    static constexpr bool value_host =
        has_method_build_v<T> && has_method_reset_from_table_v<T> &&
        has_method_reset_from_array_v<T> && has_method_set_data_type_v<T> &&
        has_method_set_feature_type_v<T> && has_method_allocate_v<T> &&
        has_method_set_layout_v<T> && has_method_copy_data_v<T>;

#ifdef ONEAPI_DAL_DATA_PARALLEL
    ONEAPI_DAL_HAS_METHOD_TRAIT(void,
                                allocate,
                                (const sycl::queue& queue,
                                 std::int64_t row_count,
                                 std::int64_t column_count,
                                 sycl::usm::alloc kind),
                                allocate_dpc)
    ONEAPI_DAL_HAS_METHOD_TRAIT(
        void,
        copy_data,
        (sycl::queue & queue, const void* data, std::int64_t row_count, std::int64_t column_count),
        copy_data_dpc)

    static constexpr bool value_dpc = has_method_allocate_dpc_v<T> && has_method_copy_data_dpc_v<T>;
    static constexpr bool value     = value_host && value_dpc;
#else
    static constexpr bool value = value_host;
#endif
};

template <typename T>
inline constexpr bool is_homogen_table_builder_impl_v = is_homogen_table_builder_impl<T>::value;

} // namespace oneapi::dal
