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

#include <daal/src/algorithms/service_kernel_math.h>

namespace oneapi::dal::knn::detail {
namespace v1 {

using daal_distance_t = daal::algorithms::internal::PairwiseDistanceType;

class distance_impl : public base {
public:
    virtual ~distance_impl() = default;
    virtual daal_distance_t get_daal_distance_type() = 0;
    virtual double get_degree() = 0;
};

} // namespace v1

using v1::distance_impl;

} // namespace oneapi::dal::knn::detail
