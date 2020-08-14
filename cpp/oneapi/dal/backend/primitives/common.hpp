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

#include <cstdint>

namespace oneapi::dal::backend::primitives {

enum class layout : int { row_major = 0, col_major = 1 };

template<int size>
struct unsigned_type { };

template<>
struct unsigned_type<1> {
    typedef std::uint8_t type;
    constexpr inline static type ones = 0xff;
    constexpr inline static type zeros = 0x00;
    constexpr inline static type vals[2] = {zeros, ones};
    constexpr inline static type val(bool condition) {
        return vals[int(condition)];
    }
};

template<>
struct unsigned_type<2> {
    typedef std::uint16_t type;
    constexpr inline static type ones = 0xffff;
    constexpr inline static type zeros = 0x0000;
    constexpr inline static type vals[2] = {zeros, ones};
    constexpr inline static type val(bool condition) {
        return vals[int(condition)];
    }
};

template<>
struct unsigned_type<4> {
    typedef std::uint32_t type;
    constexpr inline static type ones = 0xffffffff;
    constexpr inline static type zeros = 0x00000000;
    constexpr inline static type vals[2] = {zeros, ones};
    constexpr inline static type val(bool condition) {
        return vals[int(condition)];
    }
};

template<>
struct unsigned_type<8> {
    typedef std::uint64_t type;
    constexpr inline static type ones = 0xffffffffffffffff;
    constexpr inline static type zeros = 0x0000000000000000;
    constexpr inline static type vals[2] = {zeros, ones};
    constexpr inline static type val(bool condition) {
        return vals[int(condition)];
    }
};

template<typename T>
inline void conditional_swap(T& a, T& b, bool swap = true) {
    constexpr unsigned_type<sizeof(T)> ut;
    reinterpret_cast<ut::type&>(a) ^= reinterpret_cast<ut::type&>(b);
    reinterpret_cast<ut::type&>(b) ^= ut.val(swap) & reinterpret_cast<ut::type&>(a);
    reinterpret_cast<ut::type&>(a) ^= reinterpret_cast<ut::type&>(b);
}

} // namespace oneapi::dal::backend::primitives
