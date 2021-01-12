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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/util/detail/type_traits.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T, typename Data>
struct has_pull_rows_host {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void, pull_rows, (array<Data>&, const range&)const)
    static constexpr bool value = has_method_pull_rows_v<T>;
};

template <typename T, typename Data>
struct has_push_rows_host {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void, push_rows, (const array<Data>&, const range&))
    static constexpr bool value = has_method_push_rows_v<T>;
};

template <typename T, typename Data>
struct has_pull_column_host {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void,
                                   pull_column,
                                   (array<Data>&, std::int64_t, const range&)const)
    static constexpr bool value = has_method_pull_column_v<T>;
};

template <typename T, typename Data>
struct has_push_column_host {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void,
                                   push_column,
                                   (const array<Data>&, std::int64_t, const range&))
    static constexpr bool value = has_method_push_column_v<T>;
};

#ifdef ONEDAL_DATA_PARALLEL

template <typename T, typename Data>
struct has_pull_rows_dpc {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(
        void,
        pull_rows,
        (sycl::queue&, array<Data>&, const range&, const sycl::usm::alloc&)const)
    static constexpr bool value = has_method_pull_rows_v<T>;
};

template <typename T, typename Data>
struct has_push_rows_dpc {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void,
                                   push_rows,
                                   (sycl::queue&, const array<Data>&, const range&))
    static constexpr bool value = has_method_push_rows_v<T>;
};

template <typename T, typename Data>
struct has_pull_column_dpc {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(
        void,
        pull_column,
        (sycl::queue&, array<Data>&, std::int64_t, const range&, const sycl::usm::alloc&)const)
    static constexpr bool value = has_method_pull_column_v<T>;
};

template <typename T, typename Data>
struct has_push_column_dpc {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void,
                                   push_column,
                                   (sycl::queue&, const array<Data>&, std::int64_t, const range&))
    static constexpr bool value = has_method_push_column_v<T>;
};

#endif // ONEDAL_DATA_PARALLEL

} // namespace v1

using v1::has_pull_rows_host;
using v1::has_push_rows_host;
using v1::has_pull_column_host;
using v1::has_push_column_host;

#ifdef ONEDAL_DATA_PARALLEL
using v1::has_pull_rows_dpc;
using v1::has_push_rows_dpc;
using v1::has_pull_column_dpc;
using v1::has_push_column_dpc;
#endif

} // namespace oneapi::dal::detail
