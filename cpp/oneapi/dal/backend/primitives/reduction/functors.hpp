/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <cmath>
#include <cstdint>

namespace oneapi::dal::backend::primitives {

template <typename T>
struct identity {
    T operator()(const T& arg) const {
        return arg;
    }
};

template <typename T>
struct abs {
    T operator()(const T& arg) const {
#ifdef __SYCL_DEVICE_ONLY__
        return sycl::fabs(arg);
#else
        return std::abs(arg);
#endif
    }
};

template <typename T>
struct square {
    T operator()(const T& arg) const {
        return (arg * arg);
    }
};

template <typename T>
struct sum {
    constexpr static inline T init_value = 0;
#ifdef __SYCL_DEVICE_ONLY__
    constexpr static inline sycl::ONEAPI::plus<T> native{};
#else
    constexpr static inline std::plus<T> native{};
#endif
    T operator()(const T& a, const T& b) const {
        return native(a, b);
    }
};

template <typename T>
struct max {
    constexpr static inline T init_value = std::numeric_limits<T>::min();
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ONEAPI::maximum<T> native{};
#else
    constexpr static inline auto native = [](const T& a, const T& b) {
        return std::max(a, b);
    };
#endif
    T operator()(const T& a, const T& b) const {
        return native(a, b);
    }
};

template <typename T>
struct min {
    constexpr static inline T init_value = std::numeric_limits<T>::max();
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ONEAPI::minimum<T> native{};
#else
    constexpr static inline auto native = [](const T& a, const T& b) {
        return std::min(a, b);
    };
#endif
    T operator()(const T& a, const T& b) const {
        return native(a, b);
    }
};

} // namespace oneapi::dal::backend::primitives
