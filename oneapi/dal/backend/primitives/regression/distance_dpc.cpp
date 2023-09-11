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

#include <algorithm>

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/backend/primitives/regression/distance.hpp"

namespace oneapi::dal::backend::primitives {

template <typename DistType, typename ResponseType>
distance_regression<DistType, ResponseType>::distance_regression(sycl::queue& queue)
        : queue_{ queue } {}

template <typename DistType, typename ResponseType>
distance_regression<DistType, ResponseType>::~distance_regression() {
    get_queue().wait_and_throw();
}

template <typename DistType, typename ResponseType>
sycl::queue& distance_regression<DistType, ResponseType>::get_queue() const {
    return this->queue_;
}

template <typename DistType, typename ResponseType>
std::unique_ptr<distance_regression<DistType, ResponseType>>
make_distance_regression(sycl::queue& queue, std::int64_t max_block, std::int64_t neighbor_count) {
    using naive_t = naive_distance_regression<DistType, ResponseType>;
    return std::make_unique<naive_t>(queue);
}

#define INSTANTIATE(DISTANCE, RESPONSE)                               \
    template class distance_regression<DISTANCE, RESPONSE>;           \
    template std::unique_ptr<distance_regression<DISTANCE, RESPONSE>> \
    make_distance_regression<DISTANCE, RESPONSE>(sycl::queue&, std::int64_t, std::int64_t);

INSTANTIATE(float, float);
INSTANTIATE(float, double);
INSTANTIATE(double, float);
INSTANTIATE(double, double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
