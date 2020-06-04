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

#include "oneapi/dal/util/detail/type_traits.hpp"

namespace dal {

template <typename T>
struct is_table_impl {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(std::int64_t, get_column_count, () const);
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(std::int64_t, get_row_count, () const);
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(const table_metadata&, get_metadata, () const);

    INSTANTIATE_HAS_METHOD_CHECKER(void, pull_rows, (array<float>&, const range&) const, pull_rows_float);
    INSTANTIATE_HAS_METHOD_CHECKER(void, pull_rows, (array<double>&, const range&) const, pull_rows_double);
    INSTANTIATE_HAS_METHOD_CHECKER(void, pull_rows, (array<std::int32_t>&, const range&) const, pull_rows_int32);

    INSTANTIATE_HAS_METHOD_CHECKER(void, push_back_rows, (const array<float>&, const range&), push_back_rows_float);
    INSTANTIATE_HAS_METHOD_CHECKER(void, push_back_rows, (const array<double>&, const range&), push_back_rows_double);
    INSTANTIATE_HAS_METHOD_CHECKER(void, push_back_rows, (const array<std::int32_t>&, const range&), push_back_rows_int32);

    INSTANTIATE_HAS_METHOD_CHECKER(void, pull_column, (array<float>&, std::int64_t, const range&) const, pull_column_float);
    INSTANTIATE_HAS_METHOD_CHECKER(void, pull_column, (array<double>&, std::int64_t, const range&) const, pull_column_double);
    INSTANTIATE_HAS_METHOD_CHECKER(void, pull_column, (array<std::int32_t>&, std::int64_t, const range&) const, pull_column_int32);

    INSTANTIATE_HAS_METHOD_CHECKER(void, push_back_column, (const array<float>&, std::int64_t, const range&), push_back_column_float);
    INSTANTIATE_HAS_METHOD_CHECKER(void, push_back_column, (const array<double>&, std::int64_t, const range&), push_back_column_double);
    INSTANTIATE_HAS_METHOD_CHECKER(void, push_back_column, (const array<std::int32_t>&, std::int64_t, const range&), push_back_column_int32);

    static constexpr bool can_pull_rows = has_method_pull_rows_float_v<T> &&
                                          has_method_pull_rows_double_v<T> &&
                                          has_method_pull_rows_int32_v<T>;

    static constexpr bool can_push_back_rows = has_method_push_back_rows_float_v<T> &&
                                               has_method_push_back_rows_double_v<T> &&
                                               has_method_push_back_rows_int32_v<T>;

    static constexpr bool can_pull_column = has_method_pull_column_float_v<T> &&
                                            has_method_pull_column_double_v<T> &&
                                            has_method_pull_column_int32_v<T>;

    static constexpr bool can_push_back_column = has_method_push_back_column_float_v<T> &&
                                                 has_method_push_back_column_double_v<T> &&
                                                 has_method_push_back_column_int32_v<T>;

    static constexpr bool value = has_method_get_column_count_v<T> &&
                                  has_method_get_row_count_v<T> &&
                                  has_method_get_metadata_v<T> &&
                                  can_pull_rows && can_push_back_rows &&
                                  can_pull_column && can_push_back_column;
};

template <typename T>
inline constexpr bool is_table_impl_v = is_table_impl<T>::value;

template <typename T>
struct is_homogen_table_impl {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(const void*, get_data, () const);

    static constexpr bool value = is_table_impl_v<T> && has_method_get_data_v<T>;
};

template <typename T>
inline constexpr bool is_homogen_table_impl_v = is_homogen_table_impl<T>::value;

} // namespace dal
