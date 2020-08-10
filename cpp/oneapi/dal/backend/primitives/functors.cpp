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

#include "oneapi/dal/backend/functors.hpp"

namespace oneapi::dal::backend::primitives {

// Unary Functors Section

template struct unary_functor<float, unary_operation::identity>;
template struct unary_functor<double, unary_operation::identity>;
template struct unary_functor<float, unary_operation::square>;
template struct unary_functor<double, unary_operation::square>;
template struct unary_functor<float, unary_operation::abs>;
template struct unary_functor<double, unary_operation::abs>;

// Binary Functors Section

template struct binary_functor<float, binary_operation::sum>;
template struct binary_functor<double, binary_operation::sum>;
template struct binary_functor<float, binary_operation::mul>;
template struct binary_functor<double, binary_operation::mul>;
template struct binary_functor<float, binary_operation::min>;
template struct binary_functor<double, binary_operation::min>;
template struct binary_functor<float, binary_operation::max>;
template struct binary_functor<double, binary_operation::max>;

} // oneapi::dal::backend::primitives

