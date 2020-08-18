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
#include <type_traits>

namespace oneapi::dal::backend::primitives {

enum class unary_operation : int { identity = 0, square = 1, abs = 2 };

template <typename Type, unary_operation Op = unary_operation::identity>
struct unary_functor {
    inline Type operator()(Type arg);
};

template <typename Type>
struct unary_functor<Type, unary_operation::identity> {
    inline Type operator()(Type arg) const {
        return arg;
    }
};

template <typename Type>
struct unary_functor<Type, unary_operation::abs> {
    inline Type operator()(Type arg) const {
        return std::abs(arg);
    }
};

template <typename Type>
struct unary_functor<Type, unary_operation::square> {
    inline Type operator()(Type arg) const {
        return (arg * arg);
    }
};

enum class binary_operation : int { min = 0, max = 1, sum = 2, mul = 3 };

template <typename Type, binary_operation Op>
struct binary_functor {
    constexpr static inline Type init_value = 0;
    inline Type operator()(Type a, Type b) const;
};

template <typename Type>
struct binary_functor<Type, binary_operation::sum> {
    constexpr static inline Type init_value = 0;
    inline Type operator()(Type a, Type b) const {
        return (a + b);
    }
};

template <typename Type>
struct binary_functor<Type, binary_operation::mul> {
    constexpr static inline Type init_value = 1;
    inline Type operator()(Type a, Type b) const {
        return (a * b);
    }
};

template <typename Type>
struct binary_functor<Type, binary_operation::max> {
    constexpr static inline Type init_value = std::numeric_limits<Type>::min();
    inline Type operator()(Type a, Type b) const {
        return std::max(a, b);
    }
};

template <typename Type>
struct binary_functor<Type, binary_operation::min> {
    constexpr static inline Type init_value = std::numeric_limits<Type>::max();
    inline Type operator()(Type a, Type b) const {
        return std::min(a, b);
    }
};

template <typename Type, binary_operation BinOp>
struct initialized_binary_functor : public binary_functor<Type, BinOp> {
    initialized_binary_functor(Type init_value_ = binary_functor<Type, BinOp>::init_value)
            : init_value(init_value_) {}
    const Type init_value;
};

} // namespace oneapi::dal::backend::primitives
