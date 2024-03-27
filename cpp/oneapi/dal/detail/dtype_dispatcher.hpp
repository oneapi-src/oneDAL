/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include <cstdint>

#include "oneapi/dal/common.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename Op, typename OnUnknown>
inline constexpr auto dispatch_by_data_type(data_type dtype, Op&& op, OnUnknown&& on_unknown) {
    switch (dtype) {
        case data_type::int8: return op(std::int8_t{});
        case data_type::uint8: return op(std::uint8_t{});
        case data_type::int16: return op(std::int16_t{});
        case data_type::uint16: return op(std::uint16_t{});
        case data_type::int32: return op(std::int32_t{});
        case data_type::uint32: return op(std::uint32_t{});
        case data_type::int64: return op(std::int64_t{});
        case data_type::uint64: return op(std::uint64_t{});
        case data_type::float32: return op(float{});
        case data_type::float64: return op(double{});
        default: return on_unknown(dtype);
    }
}

template <typename Op, typename ResultType = std::invoke_result_t<Op, float>>
inline constexpr ResultType dispatch_by_data_type(data_type dtype, Op&& op) {
    // Necessary to make the return type conformant with
    // other dispatch branches
    const auto on_unknown = [](data_type) -> ResultType {
        using msg = dal::detail::error_messages;
        throw unimplemented{ msg::unsupported_conversion_types() };
    };

    return dispatch_by_data_type(dtype, std::forward<Op>(op), on_unknown);
}

namespace impl {

template <typename Result, typename... Types>
struct type_holder {
    using result_t = Result;

    template <typename Tail>
    using add_tail = type_holder<Result, Types..., Tail>;

    template <typename Op>
    constexpr static inline Result evaluate(Op&& op) {
        return op(Types{}...);
    }
};

template <typename TypeHolder, typename Op>
inline constexpr auto multi_dispatch_by_data_type(Op&& op) {
    return TypeHolder::evaluate(std::forward<Op>(op));
}

template <typename TypeHolder, typename Op, typename Head, typename... Tail>
inline constexpr auto multi_dispatch_by_data_type(Op&& op, Head&& head, Tail&&... tail) {
    using result_t = typename TypeHolder::result_t;
    const auto functor = [&](auto arg) -> result_t {
        using type_t = std::decay_t<decltype(arg)>;
        using holder_t = typename TypeHolder::template add_tail<type_t>;
        return multi_dispatch_by_data_type<holder_t>( //
            std::forward<Op>(op),
            std::forward<Tail>(tail)...);
    };
    return dispatch_by_data_type(head, functor);
}

template <std::size_t n, typename DefaultType, typename Op, typename... Types>
struct invoke_result_multiple_impl {
    using next_t = invoke_result_multiple_impl<n - 1, DefaultType, Op, DefaultType, Types...>;
    using type = typename next_t::type;
};

template <typename DefaultType, typename Op, typename... Types>
struct invoke_result_multiple_impl<0ul, DefaultType, Op, Types...> {
    using type = std::invoke_result_t<Op, Types...>;
};

template <typename Op, std::size_t n, typename DefaultType = float>
using invoke_result_multiple_t = typename invoke_result_multiple_impl<n, DefaultType, Op>::type;

} // namespace impl

// Signature of this function is slightly different from
// a simple `dispatch_by_data_type` due to inconsistency
// with a `std::visit` which it heavily resembles
template <typename ResultType, typename Op, typename... Types>
inline constexpr ResultType multi_dispatch_by_data_type(Op&& op, Types&&... types) {
    using holder_t = impl::type_holder<ResultType>;
    return impl::multi_dispatch_by_data_type<holder_t, Op>( //
        std::forward<Op>(op),
        std::forward<Types>(types)...);
}

template <typename Op, typename... Types>
inline constexpr auto multi_dispatch_by_data_type(Op&& op, Types&&... types) {
    using result_t = impl::invoke_result_multiple_t<Op, sizeof...(Types), float>;
    return multi_dispatch_by_data_type<result_t, Op>( //
        std::forward<Op>(op),
        std::forward<Types>(types)...);
}

} // namespace v1

using v1::dispatch_by_data_type;
using v1::multi_dispatch_by_data_type;

} // namespace oneapi::dal::detail
