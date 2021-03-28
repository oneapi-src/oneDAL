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

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction/functors.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float, ndorder order, typename BinaryOp, typename UnaryOp>
sycl::event reduce_rows(sycl::queue& q,
                        const ndview<Float, 2, order>& input,
                        ndview<Float, 1>& output,
                        const BinaryOp& binary = BinaryOp{},
                        const UnaryOp& unary = UnaryOp{},
                        const event_vector& deps = {});

template <typename Float, ndorder order, typename BinaryOp, typename UnaryOp>
sycl::event reduce_cols(sycl::queue& q,
                        const ndview<Float, 2, order>& input,
                        ndview<Float, 1>& output,
                        const BinaryOp& binary = BinaryOp{},
                        const UnaryOp& unary = UnaryOp{},
                        const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
