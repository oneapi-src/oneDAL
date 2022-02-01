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

#include <tuple>
#include <type_traits>

#include "oneapi/dal/test/engine/macro.hpp"

#define _TE_COMBINE_TYPES_TRAIT oneapi::dal::test::engine::combine_types_t

#define _TE_COMBINE_TYPES_1(_1) _TE_COMBINE_TYPES_TRAIT<std::tuple<_TE_UNPACK(_1)>>

#define _TE_COMBINE_TYPES_2(_1, _2) \
    _TE_COMBINE_TYPES_TRAIT<std::tuple<_TE_UNPACK(_1)>, std::tuple<_TE_UNPACK(_2)>>

#define _TE_COMBINE_TYPES_3(_1, _2, _3)                 \
    _TE_COMBINE_TYPES_TRAIT<std::tuple<_TE_UNPACK(_1)>, \
                            std::tuple<_TE_UNPACK(_2)>, \
                            std::tuple<_TE_UNPACK(_3)>>

#define _TE_COMBINE_TYPES_4(_1, _2, _3, _4)             \
    _TE_COMBINE_TYPES_TRAIT<std::tuple<_TE_UNPACK(_1)>, \
                            std::tuple<_TE_UNPACK(_2)>, \
                            std::tuple<_TE_UNPACK(_3)>, \
                            std::tuple<_TE_UNPACK(_4)>>

#define _TE_COMBINE_TYPES_5(_1, _2, _3, _4, _5)         \
    _TE_COMBINE_TYPES_TRAIT<std::tuple<_TE_UNPACK(_1)>, \
                            std::tuple<_TE_UNPACK(_2)>, \
                            std::tuple<_TE_UNPACK(_3)>, \
                            std::tuple<_TE_UNPACK(_4)>, \
                            std::tuple<_TE_UNPACK(_5)>>

#define _TE_COMBINE_TYPES_6(_1, _2, _3, _4, _5, _6)     \
    _TE_COMBINE_TYPES_TRAIT<std::tuple<_TE_UNPACK(_1)>, \
                            std::tuple<_TE_UNPACK(_2)>, \
                            std::tuple<_TE_UNPACK(_3)>, \
                            std::tuple<_TE_UNPACK(_4)>, \
                            std::tuple<_TE_UNPACK(_5)>, \
                            std::tuple<_TE_UNPACK(_6)>>

#define _TE_COMBINE_TYPES_(N, ...) _TE_CONCAT_2(_TE_COMBINE_TYPES_, N)(__VA_ARGS__)

#define COMBINE_TYPES(...) _TE_COMBINE_TYPES_(_TE_NARGS(__VA_ARGS__), __VA_ARGS__)

namespace oneapi::dal::test::engine {

template <std::size_t index, typename TupleX, typename TupleY>
struct combine_types_element_2d {
private:
    static constexpr std::size_t count_x = std::tuple_size_v<TupleX>;
    static constexpr std::size_t count_y = std::tuple_size_v<TupleY>;
    static constexpr std::size_t i = index / count_y;
    static constexpr std::size_t j = index % count_y;

    static_assert(i < count_x);
    static_assert(j < count_y);

public:
    using type = std::tuple<std::tuple_element_t<i, TupleX>, std::tuple_element_t<j, TupleY>>;
};

template <std::size_t index, typename TupleX, typename TupleY>
using combine_types_element_2d_t = typename combine_types_element_2d<index, TupleX, TupleY>::type;

template <typename Head, typename... Tail>
struct combine_types {
private:
    template <typename T, typename... Args>
    static constexpr auto compress_tuple(std::tuple<T, std::tuple<Args...>>)
        -> std::tuple<T, Args...>;

    // tuple<T, tuple<U, V>> -> tuple<T, U, V>
    template <typename Tuple>
    using compress_tuple_t = decltype(compress_tuple(std::declval<Tuple>()));

    template <typename... Tuples>
    static constexpr auto compress_tuples(std::tuple<Tuples...>)
        -> std::tuple<compress_tuple_t<Tuples>...>;

    // tuple< tuple<T, tuple<U, V>>... > -> tuple< tuple<T, U, V>, ... >
    template <typename Tuple>
    using compress_tuples_t = decltype(compress_tuples(std::declval<Tuple>()));

    using combined_tail_t = typename combine_types<Tail...>::type;
    using combined_head_t = typename combine_types<Head, combined_tail_t>::type;

public:
    using type = compress_tuples_t<combined_head_t>;
};

template <typename First, typename Second>
struct combine_types<First, Second> {
private:
    template <std::size_t... indices>
    static constexpr auto index_helper(std::index_sequence<indices...>)
        -> std::tuple<combine_types_element_2d_t<indices, First, Second>...>;

public:
    static constexpr std::size_t count = std::tuple_size_v<First> * std::tuple_size_v<Second>;
    using type = decltype(index_helper(std::make_index_sequence<count>{}));
};

template <typename Tuple>
struct combine_types<Tuple> {
    using type = Tuple;
};

template <typename... Tuples>
using combine_types_t = typename combine_types<Tuples...>::type;

} // namespace oneapi::dal::test::engine
