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

#include <cstdint>

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/super_accumulator/int128.hpp"

namespace oneapi::dal::backend::primitives {

namespace detail::float64 {

constexpr inline int nbins = 64;
constexpr inline int binsize = 2;
constexpr inline int binalloc = nbins * binsize;
constexpr inline int pbias = 1023;
constexpr inline int maxbins = 2048;
constexpr inline int binratio = maxbins / nbins;
constexpr inline duality64 mpowd{ static_cast<std::uint64_t>(0x3cb0000000000000ul) };

inline bool sign(const duality64& val) {
    constexpr int shift = 63;
    return !bool(val.integer >> shift);
}

inline std::int32_t expn(const duality64& val) {
    constexpr int shift = 52;
    constexpr std::uint64_t mask = 0x7FF0000000000000;
    return (val.integer & mask) >> shift;
}

inline std::int64_t mant(const duality64& val,
                         const std::int32_t& expn) {
    constexpr std::uint64_t mask = 0x000FFFFFFFFFFFFFul;
    constexpr std::uint64_t comp = 0x0010000000000000ul;
    const std::uint64_t valbits = val.integer & mask;
    // Fixes normalized mantis
    return expn ? (valbits | comp) : (valbits << 1);
}

inline std::int64_t mant(const duality64& val) {
    return mant(val, expn(val));
}

struct double_u {
    double_u(const duality64& arg) :
        sign_{ sign(arg) },
        expn_{ expn(arg) },
        mant_{ mant(arg) } {}

    const bool sign_;
    const std::int32_t expn_;
    const std::int64_t mant_;
};


inline std::int32_t bin_idx(const std::int32_t& expn) {
    return expn / binratio;
}

inline std::int32_t exp_dif(const std::int32_t& expn) {
    return expn % binratio;
}

inline int128_raw new_mant(const std::int64_t& mant,
                           const std::int32_t& expn) {
    return int128_raw::make(0l, mant) << exp_dif(expn);
}

} // namespace detail::float32

} // namespace oneapi::dal::backend::primitives
