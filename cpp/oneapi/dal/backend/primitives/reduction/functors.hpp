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

namespace oneapi::dal::backend::primitives {

enum class unary_operation : int { identity = 0, square = 1, abs = 2 };

template <typename T, unary_operation Op>
struct unary_functor {
    inline T operator()(T arg);
};

template <typename T>
struct unary_functor<T, unary_operation::identity> {
    inline T operator()(T arg) const {
        return arg;
    }
};

template <typename T>
using identity = unary_functor<T, unary_operation::identity>;

template <typename T>
struct unary_functor<T, unary_operation::abs> {
    inline T operator()(T arg) const {
        return ((arg < 0) ? -arg : arg);
    }
};

template <typename T>
using abs = unary_functor<T, unary_operation::abs>;

template <typename T>
struct unary_functor<T, unary_operation::square> {
    inline T operator()(T arg) const {
        return (arg * arg);
    }
};

template <typename T>
using square = unary_functor<T, unary_operation::square>;

enum class binary_operation : int { min = 0, max = 1, sum = 2, mul = 3 };

template <typename T, binary_operation Op>
struct binary_functor {
    constexpr static inline T init_value = 0;
    inline T operator()(T a, T b) const;
};

template <typename T>
struct binary_functor<T, binary_operation::sum> {
    constexpr static inline T init_value = 0;
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ONEAPI::plus<T> native{};
#endif
    inline T operator()(T a, T b) const {
        return (a + b);
    }
};

template <typename T>
using sum = binary_functor<T, binary_operation::sum>;

template <typename T>
struct binary_functor<T, binary_operation::mul> {
    constexpr static inline T init_value = 1;
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ONEAPI::multiplies<T> native{};
#endif
    inline T operator()(T a, T b) const {
        return (a * b);
    }
};

template <typename T>
using mul = binary_functor<T, binary_operation::mul>;

template <typename T>
struct binary_functor<T, binary_operation::max> {
    constexpr static inline T init_value = std::numeric_limits<T>::min();
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ONEAPI::maximum<T> native{};
#endif
    inline T operator()(T a, T b) const {
        return (a < b) ? b : a;
    }
};

template <typename T>
using max = binary_functor<T, binary_operation::max>;

template <typename T>
struct binary_functor<T, binary_operation::min> {
    constexpr static inline T init_value = std::numeric_limits<T>::max();
#ifdef ONEDAL_DATA_PARALLEL
    constexpr static inline sycl::ONEAPI::minimum<T> native{};
#endif
    inline T operator()(T a, T b) const {
        return (a < b) ? a : b;
    }
};

template <typename T>
using min = binary_functor<T, binary_operation::min>;

template <typename T, binary_operation BinOp>
struct initialized_binary_functor : public binary_functor<T, BinOp> {
    initialized_binary_functor(T init_value_ = binary_functor<T, BinOp>::init_value)
            : init_value(init_value_) {}
    const T init_value;
};

} // namespace oneapi::dal::backend::primitives
