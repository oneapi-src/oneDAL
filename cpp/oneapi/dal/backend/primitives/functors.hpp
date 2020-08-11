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

#include <algorithm>
#include <cmath>
#include <limits>

namespace oneapi::dal::backend::primitives {

// Unary Functors Section

enum class unary_operation : int { identity = 0, square = 1, abs = 2 };

template <typename Float, unary_operation Op = unary_operation::identity>
struct unary_functor {
    inline Float operator()(Float arg);
};

template <typename Float>
struct unary_functor<Float, unary_operation::identity> {
    inline Float operator()(Float arg) const {
        return arg;
    }
};

template <typename Float>
struct unary_functor<Float, unary_operation::abs> {
    inline Float operator()(Float arg) const {
        return std::abs(arg);
    }
};

template <typename Float>
struct unary_functor<Float, unary_operation::square> {
    inline Float operator()(Float arg) const {
        return (arg * arg);
    }
};

// Binary Functors Section

enum class binary_operation : int { min = 0, max = 1, sum = 2, mul = 3 };

template <typename Float, binary_operation Op>
struct binary_functor {
    constexpr static inline Float init_value = static_cast<Float>(NAN);
    inline Float operator()(Float a, Float b) const;
};

template <typename Float>
struct binary_functor<Float, binary_operation::sum> {
    constexpr static inline Float init_value = 0;
    inline Float operator()(Float a, Float b) const {
        return (a + b);
    }
};

template <typename Float>
struct binary_functor<Float, binary_operation::mul> {
    constexpr static inline Float init_value = 1;
    inline Float operator()(Float a, Float b) const {
        return (a * b);
    }
};

template <typename Float>
struct binary_functor<Float, binary_operation::max> {
    constexpr static inline Float init_value = std::numeric_limits<Float>::min();
    inline Float operator()(Float a, Float b) const {
        return std::max(a, b);
    }
};

template <typename Float>
struct binary_functor<Float, binary_operation::min> {
    constexpr static inline Float init_value = std::numeric_limits<Float>::max();
    inline Float operator()(Float a, Float b) const {
        return std::min(a, b);
    }
};

template <typename Float, binary_operation BinOp>
struct initialized_binary_functor : public binary_functor<Float, BinOp> {
    initialized_binary_functor(Float init_value_ = binary_functor<Float, BinOp>::init_value)
            : init_value(init_value_) {}
    const Float init_value;
};

} // namespace oneapi::dal::backend::primitives
