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
struct unary_functor {
    inline T operator()(T arg);
};

template <typename T>
struct identity : public unary_functor<T> {
    inline T operator()(T arg) const {
        return arg;
    }
};

template <typename T>
struct abs : public unary_functor<T> {
    inline T operator()(T arg) const {
#ifdef __SYCL_DEVICE_ONLY__
        return sycl::fabs(arg);
#else
        return std::abs(arg);
#endif
    }
};

template <typename T>
struct square : public unary_functor<T> {
    inline T operator()(T arg) const {
        return (arg * arg);
    }
};

template <typename T>
struct binary_functor {
    constexpr static inline T init_value = 0;
    inline T operator()(T a, T b) const;
};

template <typename T>
struct sum : public binary_functor<T> {
    constexpr static inline T init_value = 0;
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ONEAPI::plus<T> native{};
#endif
    inline T operator()(T a, T b) const {
        return (a + b);
    }
};

template <typename T>
struct mul : public binary_functor<T> {
    constexpr static inline T init_value = 1;
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ONEAPI::multiplies<T> native{};
#endif
    inline T operator()(T a, T b) const {
        return (a * b);
    }
};

template <typename T>
struct max : public binary_functor<T> {
    constexpr static inline T init_value = std::numeric_limits<T>::min();
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ONEAPI::maximum<T> native{};
#endif
    inline T operator()(T a, T b) const {
        return (a < b) ? b : a;
    }
};

template <typename T>
struct min : public binary_functor<T> {
    constexpr static inline T init_value = std::numeric_limits<T>::max();
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ONEAPI::minimum<T> native{};
#endif
    inline T operator()(T a, T b) const {
        return (a < b) ? a : b;
    }
};

template <typename T, typename BinOp>
struct initialized_binary_functor : public binary_functor<T> {
    initialized_binary_functor(T init_value_ = BinOp::init_value) : init_value(init_value_) {}
    const T init_value;
};

} // namespace oneapi::dal::backend::primitives
