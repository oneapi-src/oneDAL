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

namespace oneapi::dal::backend::primitives {

enum class unary_operation : int {
    identity,
    square,
    abs
};

template<typename Float, unary_operation Op = unary_operation::identity>
struct unary_functor { 
    constexpr Float operator() (Float arg);
};

enum class binary_operation : int {
    min,
    max,
    sum,
    mul
};

template<typename Float, binary_operation Op = binary_operation::sum>
struct binary_functor {
    constexpr Float init_value;
    constexpr Float operator() (Float arg);
    constexpr Float operator() (Float a, Float b);
};

template<typename Float, unary_operation UnOp, binary_operation BinOp>
struct composed_functor : public unary_functor<Float, UnOp>, binary_functor<Float, BinOp> {
    constexpr Float operator() (Float a) override;
    constexpr Float operator() (Float a, Float b) override;
};

template<typename Float>
using l1_functor = typename composed_functor<Float, unary_operation::abs, binary_operation::sum>;
template<typename Float>
using l2_functor = typename composed_functor<Float, unary_operation::square, binary_operation::sum>;
template<typename Float>
using linf_functor = typename composed_functor<Float, unary_operation::abs, binary_operation::max>;

} // namespace oneapi::dal::detail