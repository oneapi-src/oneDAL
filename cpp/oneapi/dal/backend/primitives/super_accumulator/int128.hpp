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
#include <type_traits>

#include "oneapi/dal/backend/primitives/common.hpp"

namespace oneapi::dal::backend::primitives {

union duality64 {
    double floatingpoint;
    std::uint64_t integer;
    constexpr duality64(const std::uint64_t& val) : integer(val) {}
    constexpr duality64(const double& val) : floatingpoint(val) {}
};

template<typename Impl>
class int128_base {
public:
    using derived_t = Impl;

    int128_base() = default;

    std::int64_t& get_high_half() {
        return static_cast<Impl*>(this)->get_high_half_impl();
    }

    std::uint64_t& get_low_half() {
        return static_cast<Impl*>(this)->get_low_half_impl();
    }

    const std::int64_t& get_high_half() const {
        return static_cast<const Impl*>(this)->get_high_half_impl();
    }

    const std::uint64_t& get_low_half() const {
        return static_cast<const Impl*>(this)->get_low_half_impl();
    }

    static derived_t make(std::int64_t h,
                          std::uint64_t l,
                          derived_t result = {}) {
        result.get_high_half() = h;
        result.get_low_half() = l;
        return result;
    }

    template<typename AnotherImpl, bool atomic = false>
    Impl& add(const int128_base<AnotherImpl>& rhs) {
        if constexpr (atomic) {
#ifdef __SYCL_DEVICE_ONLY__
            const std::uint64_t low = atomic_global_add(&get_low_half(), rhs.get_low_half());
            // We need to add number again because atomic_fetch_add returns prev value
            const bool carry = bool((low + rhs.get_low_half()) < rhs.get_low_half());
            const auto incrt = rhs.get_high_half() + carry;
            atomic_global_add(&get_high_half(), incrt);
#else
            const std::uint64_t low = (get_low_half() += rhs.get_low_half());
            const bool carry = bool(low < rhs.get_low_half());
            get_high_half() += rhs.get_high_half() + carry;
#endif
        } else {
            get_low_half() += rhs.get_low_half();
            bool carry = bool(get_low_half() < rhs.get_low_half());
            get_high_half() += rhs.get_high_half() + carry;
        }
        return *static_cast<Impl*>(this);
    }

    operator double() const {
        return to_floatingpoint<double>();
    }

protected:
    template<typename Float>
    auto to_floatingpoint() const {
        return (get_high_half() < 0l)
            ? -to_floatingpoint_pos<Float>(-(*this))
            : to_floatingpoint_pos<Float>(*this);
    }

    template<typename Float, typename AnotherImpl>
    static auto to_floatingpoint_pos(const int128_base<AnotherImpl>& x) {
        constexpr duality64 mul{ static_cast<std::uint64_t>(0x43f0000000000000ul) };
        return static_cast<Float>(x.get_low_half()) + static_cast<Float>(x.get_high_half()) * mul.floatingpoint;
    }

private:
#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    static inline T atomic_global_add(T* ptr, T operand) {
        using address = cl::sycl::access::address_space;
        return cl::sycl::atomic_fetch_add<T, address::global_space>(
            { cl::sycl::multi_ptr<T, address::global_space>{ ptr } },
            operand);
    }
#endif
};

class int128_raw : public int128_base<int128_raw> {
    friend int128_base<int128_raw>;

public:
    using base_t = int128_base<int128_raw>;

    int128_raw() = default;

    template<typename Impl>
    int128_raw(const int128_base<Impl>& x)
        : high_(x.get_high_half()), low_(x.get_low_half()) {}

protected:
    std::int64_t& get_high_half_impl() {
        return high_;
    }

    std::uint64_t& get_low_half_impl() {
        return low_;
    }

    const std::int64_t& get_high_half_impl() const {
        return high_;
    }

    const std::uint64_t& get_low_half_impl() const {
        return low_;
    }

private:
    std::int64_t high_;
    std::uint64_t low_;
};

class int128_ptr : public int128_base<int128_ptr> {
    friend int128_base<int128_ptr>;

public:
    using base_t = int128_base<int128_ptr>;

    int128_ptr(std::int64_t* const ptr) : ptr_(ptr) {}

protected:
    std::int64_t& get_high_half_impl() {
        return *reinterpret_cast<std::int64_t*>(ptr_);
    }

    std::uint64_t& get_low_half_impl() {
        return *reinterpret_cast<std::uint64_t*>(ptr_ + 1ul);
    }

    const std::int64_t& get_high_half_impl() const {
        return *reinterpret_cast<std::int64_t*>(ptr_);
    }

    const std::uint64_t& get_low_half_impl() const {
        return *reinterpret_cast<std::uint64_t*>(ptr_ + 1ul);
    }

private:
    std::int64_t* const ptr_;
};

template<typename Impl1, typename Impl2>
inline bool operator==(const int128_base<Impl1>& lhs, const int128_base<Impl2>& rhs) {
    const auto hc = lhs.get_high_half() == rhs.get_high_half();
    const auto lc = lhs.get_low_half() == rhs.get_low_half();
    return (hc && lc);
}

template<typename Impl1, typename Impl2>
inline auto operator+(const int128_base<Impl1>& lhs, const int128_base<Impl2>& rhs) {
    const std::uint64_t low = lhs.get_low_half() + rhs.get_low_half();
    const bool carry = low < lhs.get_low_half();
    const std::int64_t high = lhs.get_high_half() + rhs.get_high_half() + carry;
    return int128_raw::make(high, low);
}

template<typename Impl1, typename Impl2, bool atomic = false>
inline auto& operator+=(int128_base<Impl1>& lhs, const int128_base<Impl2>& rhs) {
    return lhs.template add<Impl2, atomic>(rhs);
}

template<typename Impl>
inline auto operator<<(const int128_base<Impl>& lhs, const int rhs) {
    constexpr int max_shift = 64;
    if(rhs == 0) {
        return int128_raw(lhs);
    } else if(rhs < max_shift) {
        const std::uint64_t low = lhs.get_low_half() << rhs;
        const std::int64_t high1 = lhs.get_high_half() << rhs;
        const std::uint64_t high2 = lhs.get_low_half() >> (max_shift - rhs);
        return int128_raw::make(high1 | static_cast<std::int64_t>(high2), low);
    } else {
        const std::uint64_t high = lhs.get_low_half() << (rhs - max_shift);
        return int128_raw::make(static_cast<std::int64_t>(high), 0ul);
    }
}

template<typename Impl>
inline auto operator-(const int128_base<Impl>& arg) {
    const auto low = ~arg.get_low_half() + 1ul;
    const bool carry = bool(low == 0ul);
    const auto high = ~arg.get_high_half() + carry;
    return int128_raw::make(high, low);
}

} // namespace oneapi::dal::backend::primitives
