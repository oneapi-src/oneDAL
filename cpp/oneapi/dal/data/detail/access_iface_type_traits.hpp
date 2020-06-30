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

#include "oneapi/dal/data/array.hpp"
#include "oneapi/dal/util/detail/type_traits.hpp"

namespace oneapi::dal::detail {

template <typename T, typename DataType>
struct has_pull_rows_host {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(void, pull_rows, (array<DataType>&, const range&) const)
    static constexpr bool value = has_method_pull_rows_v<T>;
};

template <typename T, typename DataType>
struct has_push_rows_host {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(void, push_rows, (const array<DataType>&, const range&))
    static constexpr bool value = has_method_push_rows_v<T>;
};

template <typename T, typename DataType>
struct has_pull_column_host {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(void, pull_column, (array<DataType>&, std::int64_t, const range&) const)
    static constexpr bool value = has_method_pull_column_v<T>;
};

template <typename T, typename DataType>
struct has_push_column_host {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(void, push_column, (const array<DataType>&, std::int64_t, const range&))
    static constexpr bool value = has_method_push_column_v<T>;
};

#ifdef ONEAPI_DAL_DATA_PARALLEL

template <typename T, typename DataType>
struct has_pull_rows_dpc {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(void, pull_rows, (sycl::queue&, array<DataType>&, const range&) const)
    static constexpr bool value = has_method_pull_rows_v<T>;
};

template <typename T, typename DataType>
struct has_push_rows_dpc {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(void, push_rows, (sycl::queue&, const array<DataType>&, const range&))
    static constexpr bool value = has_method_push_rows_v<T>;
};

template <typename T, typename DataType>
struct has_pull_column_dpc {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(void, pull_column, (sycl::queue&, array<DataType>&, std::int64_t, const range&) const)
    static constexpr bool value = has_method_pull_column_v<T>;
};

template <typename T, typename DataType>
struct has_push_column_dpc {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(void, push_column, (sycl::queue&, const array<DataType>&, std::int64_t, const range&))
    static constexpr bool value = has_method_push_column_v<T>;
};

#endif

} // namespace oneapi::dal::detail
