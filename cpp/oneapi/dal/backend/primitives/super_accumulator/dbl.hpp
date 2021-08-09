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

#include <cstdint>

#include "oneapi/dal/backend/primitives/common.hpp"

namespace oneapi::dal::backend::primitives {

namespace detail::float64 {

union duality32 {
    float floatingpoint;
    std::uint32_t integer;
    constexpr duality32(const std::uint32_t& val) : integer(val) {}
    constexpr duality32(const float& val) : floatingpoint(val) {}
};

constexpr static int nbins = 64;
constexpr static int pbias = 127;
constexpr static int maxbins = 256;
constexpr static int binratio = maxbins / nbins;
constexpr static duality32 mpowf{ 0x34000000u };

inline std::int32_t sign(const duality32& val) {
    constexpr int shift = 31;
    return (val.integer >> shift) ? -1 : 1;
}

inline std::int32_t expn(const duality32& val) {
    constexpr int shift = 23;
    constexpr std::uint32_t mask = 0x7f800000;
    return (val.integer & mask) >> shift;
}

inline std::int32_t mant(const duality32& val, const std::int32_t& expn) {
    constexpr std::uint32_t mask = 0x7fffff;
    constexpr std::uint32_t comp = 0x800000;
    const std::uint32_t valbits = val.integer & mask;
    // Fixes normalized mantis
    return expn ? (valbits | comp) : (valbits << 1);
}

inline std::int32_t mant(const duality32& val) {
    return mant(val, expn(val));
}

struct float_u {
    float_u(const duality32& arg) : sign_{ sign(arg) }, expn_{ expn(arg) }, mant_{ mant(arg) } {}

    const std::int32_t sign_;
    const std::int32_t expn_;
    const std::int32_t mant_;
};

inline std::int32_t bin_idx(const std::int32_t& expn) {
    return expn / binratio;
}

inline std::int32_t exp_dif(const std::int32_t& expn) {
    return expn % binratio;
}

inline std::int64_t new_mant(const std::int32_t& mant, const std::int32_t& expn) {
    return std::int64_t(mant) << exp_dif(expn);
}

#ifdef ONEDAL_DATA_PARALLEL

template <typename T>
inline T atomic_global_add(T* ptr, T operand) {
    using address = cl::sycl::access::address_space;
    return cl::sycl::atomic_fetch_add<T, address::global_space>(
        { cl::sycl::multi_ptr<T, address::global_space>{ ptr } },
        operand);
}

#endif

} // namespace detail::float32

} // namespace oneapi::dal::backend::primitives
