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

#include <cmath>
#include <limits>
#include <algorithm>

#include "oneapi/dal/backend/primirives/reducer.hpp"

namespace oneapi::dal::backend::primitives {

template<typename Float>
constexpr Float unary_functor<Float, unary_operation::identity>::operator() (Float arg) {
    return arg;
}

template<typename Float>
constexpr Float unary_functor<Float, unary_operation::abs>::operator() (Float arg) {
    return std::abs(arg);
}

template<typename Float>
constexpr Float unary_functor<Float, unary_operation::square>::operator() (Float arg) {
    return (arg * arg);
}

//Direct instatiation
template struct unary_functor<float, unary_operation::identity>;
template struct unary_functor<double, unary_operation::identity>;
template struct unary_functor<float, unary_operation::square>;
template struct unary_functor<double, unary_operation::square>;
template struct unary_functor<float, unary_operation::abs>;
template struct unary_functor<double, unary_operation::abs>;

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::sum>::init_value = 0;

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::mul>::init_value = 1;

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::min>::init_value = std::numeric_limits<Float>::max();

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::max>::init_value = std::numeric_limits<Float>::min();

template<typename Float, binary_operation Op>
constexpr Float binary_functor<Float, Op>::init_value = 0;

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::sum>::operator() (Float a) {
    return operator()(init_value, a);
}

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::mul>::operator() (Float a, Float b) {
    return (a * b);
}

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::min>::operator() (Float a, Float b) {
    return std::min(a, b);
}

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::max>::operator() (Float a, Float b) {
    return std::max(a, b);
}

//Direct instatiation
template struct binary_functor<float, binary::sum>;
template struct binary_functor<double, binary::sum>;
template struct binary_functor<float, binary::mul>;
template struct binary_functor<double, binary::mul>;
template struct binary_functor<float, binary::min>;
template struct binary_functor<double, binary::max>;

template<typename Float, unary_operation UnOp, binary_operation BinOp>
constexpr Float composed_functor<Float, UnOp, BinOp>::operator() (Float a, Float b) {
    Float unary_res = unary_functor<Float, UnOp>::operator()(b);
    return binary_functor<Float, BinOp>::operator()(a, std::move(unary_res)); 
}

template<typename Float, unary_operation UnOp, binary_operation BinOp>
constexpr Float composed_functor<Float, UnOp, BinOp>::operator() (Float a) {
    Float unary_res = unary_functor<Float, UnOp>::operator()(b);
    return binary_functor<Float, BinOp>::operator()(std::move(unary_res)); 
}

//Direct instantiation
template struct l1_functor<float>;
template struct l1_functor<double>;
template struct l2_functor<float>;
template struct l2_functor<double>;
template struct linf_functor<float>;
template struct linf_functor<double>;

} // oneapi::dal::backend::primitives