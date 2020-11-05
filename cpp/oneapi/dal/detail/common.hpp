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

#include <memory>
#include <limits>
#include <type_traits>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T, typename... Args>
using is_one_of = std::disjunction<std::is_same<T, Args>...>;

template <typename T, typename... Args>
constexpr bool is_one_of_v = is_one_of<T, Args...>::value;

template <class T, class U = void>
struct enable_if_type {
    using type = U;
};

template <typename T>
using enable_if_type_t = typename enable_if_type<T>::type;

template <typename T, typename Enable = void>
struct is_tagged : std::false_type {};

template <typename T>
struct is_tagged<T, enable_if_type_t<typename T::tag_t>> : std::true_type {};

template <typename T>
constexpr bool is_tagged_v = is_tagged<T>::value;

template <typename T, bool Enable = is_tagged_v<T>>
struct is_tag_one_of_impl {};

template <typename T>
struct is_tag_one_of_impl<T, true> {
    template <typename... Tags>
    static constexpr bool value = is_one_of_v<typename T::tag_t, Tags...>;
};

template <typename T>
struct is_tag_one_of_impl<T, false> {
    template <typename... Tags>
    static constexpr bool value = false;
};

template <typename T, typename... Tags>
struct is_tag_one_of {
    static constexpr bool value = is_tag_one_of_impl<T>::template value<Tags...>;
};

template <typename T, typename... Tags>
constexpr bool is_tag_one_of_v = is_tag_one_of<T, Tags...>::value;

template <typename T>
using shared = std::shared_ptr<T>;

template <typename T>
using unique = std::unique_ptr<T>;

template <typename T>
using pimpl = shared<T>;

struct pimpl_accessor {
    template <typename Object>
    auto& get_pimpl(Object&& object) const {
        return object.impl_;
    }

    template <typename Object, typename... Args>
    Object make(Args&&... args) const {
        return Object{ std::forward<Args>(args)... };
    }
};

template <typename Object>
inline auto& get_impl(Object&& object) {
    return *pimpl_accessor{}.get_pimpl(object);
}

template <typename Impl, typename Object>
inline Impl& cast_impl(Object&& object) {
    return static_cast<Impl&>(get_impl(object));
}

template <typename Object, typename... Args>
inline Object make_private(Args&&... args) {
    return pimpl_accessor{}.template make<Object>(std::forward<Args>(args)...);
}

inline constexpr std::int64_t get_data_type_size(data_type t) {
    if (t == data_type::float32) {
        return sizeof(float);
    }
    else if (t == data_type::float64) {
        return sizeof(double);
    }
    else if (t == data_type::int32) {
        return sizeof(int32_t);
    }
    else if (t == data_type::int64) {
        return sizeof(int64_t);
    }
    else if (t == data_type::uint32) {
        return sizeof(uint32_t);
    }
    else if (t == data_type::uint64) {
        return sizeof(uint64_t);
    }
    else {
        throw unimplemented{ dal::detail::error_messages::unsupported_data_type() };
    }
}

template <typename T>
constexpr data_type make_data_type_impl();

template <typename T>
constexpr data_type make_data_type() {
    return make_data_type_impl<std::decay_t<T>>();
}

template <typename T>
constexpr data_type make_data_type_impl() {
    if constexpr (std::is_same_v<std::int32_t, T>) {
        return data_type::int32;
    }
    else if constexpr (std::is_same_v<std::int64_t, T>) {
        return data_type::int64;
    }
    else if constexpr (std::is_same_v<std::uint32_t, T>) {
        return data_type::uint32;
    }
    else if constexpr (std::is_same_v<std::uint64_t, T>) {
        return data_type::uint64;
    }
    else if constexpr (std::is_same_v<float, T>) {
        return data_type::float32;
    }
    else if constexpr (std::is_same_v<double, T>) {
        return data_type::float64;
    }

    static_assert(
        is_one_of_v<T, std::int32_t, std::int64_t, std::uint32_t, std::uint64_t, float, double>,
        "unsupported data type");
    return data_type::float32; // shall never come here
}

constexpr bool is_floating_point(data_type t) {
    if (t == data_type::bfloat16 || t == data_type::float32 || t == data_type::float64) {
        return true;
    }
    else {
        return false;
    }
}

template <typename T>
constexpr bool is_floating_point() {
    return is_floating_point(make_data_type<T>());
}

template <typename Data>
struct integer_overflow_ops {
    void check_mul_overflow(const Data& first, const Data& second);
    void check_sum_overflow(const Data& first, const Data& second);
};

template <typename Data>
inline void check_sum_overflow(const Data& first, const Data& second) {
    static_assert(std::is_integral_v<Data>, "The check requires integral operands");
    integer_overflow_ops<Data>{}.check_sum_overflow(first, second);
}

template <typename Data>
inline void check_mul_overflow(const Data& first, const Data& second) {
    static_assert(std::is_integral_v<Data>, "The check requires integral operands");
    integer_overflow_ops<Data>{}.check_mul_overflow(first, second);
}

template <typename Data>
struct limits {
    static constexpr Data min() {
        return std::numeric_limits<Data>::min();
    }
    static constexpr Data max() {
        return std::numeric_limits<Data>::max();
    }
};

template <typename Out, typename In>
inline Out integral_cast(const In& value) {
    static_assert(std::is_integral_v<In> && std::is_integral_v<Out>,
                  "The cast requires integral operands");
    if constexpr (std::is_signed_v<Out> && std::is_signed_v<In>) {
        ONEDAL_ASSERT(value <= limits<Out>::max(), "Integral type conversion overflow");
        ONEDAL_ASSERT(value >= limits<Out>::min(), "Integral type conversion underflow");
    }
    else if constexpr (std::is_unsigned_v<Out> && std::is_unsigned_v<In>) {
        ONEDAL_ASSERT(value <= limits<Out>::max(), "Integral type conversion overflow");
    }
    else if constexpr (std::is_unsigned_v<Out> && std::is_signed_v<In>) {
        ONEDAL_ASSERT(value >= In(0), "Negative integral value conversion to unsigned");
        ONEDAL_ASSERT(static_cast<std::make_unsigned_t<In>>(value) <= limits<Out>::max(),
                      "Integral type conversion overflow");
    }
    else if constexpr (std::is_signed_v<Out> && std::is_unsigned_v<In>) {
        ONEDAL_ASSERT(value <= static_cast<std::make_unsigned_t<Out>>(limits<Out>::max()),
                      "Integral type conversion overflow");
    }
    return static_cast<Out>(value);
}

} // namespace v1

using v1::is_one_of;
using v1::is_one_of_v;
using v1::is_tagged;
using v1::is_tagged_v;
using v1::is_tag_one_of;
using v1::is_tag_one_of_v;

using v1::shared;
using v1::unique;
using v1::pimpl;
using v1::pimpl_accessor;
using v1::limits;

using v1::get_impl;
using v1::cast_impl;
using v1::make_private;
using v1::make_data_type;
using v1::get_data_type_size;
using v1::is_floating_point;
using v1::check_sum_overflow;
using v1::check_mul_overflow;
using v1::integral_cast;

} // namespace oneapi::dal::detail
