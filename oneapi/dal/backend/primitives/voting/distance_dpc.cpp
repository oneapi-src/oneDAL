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

#include "oneapi/dal/backend/primitives/voting/distance.hpp"

namespace oneapi::dal::backend::primitives {

template <typename DistType, typename ClassType>
distance_voting<DistType, ClassType>::distance_voting(sycl::queue& queue, std::int64_t class_count)
        : queue_{ queue },
          class_count_{ class_count } {}

template <typename DistType, typename ClassType>
distance_voting<DistType, ClassType>::~distance_voting() {
    get_queue().wait_and_throw();
}

template <typename DistType, typename ClassType>
sycl::queue& distance_voting<DistType, ClassType>::get_queue() const {
    return this->queue_;
}

template <typename DistType, typename ClassType>
std::int64_t distance_voting<DistType, ClassType>::get_class_count() const {
    return this->class_count_;
}

template <typename DistType, typename ClassType>
std::unique_ptr<distance_voting<DistType, ClassType>>
make_distance_voting(sycl::queue& queue, std::int64_t max_block, std::int64_t class_count) {
    using naive_t = naive_distance_voting<DistType, ClassType>;
    return std::make_unique<naive_t>(queue, max_block, class_count);
}

#define INSTANTIATE(FLOAT, CLASS)                                                               \
    template class distance_voting<FLOAT, CLASS>;                                               \
    template std::unique_ptr<distance_voting<FLOAT, CLASS>> make_distance_voting<FLOAT, CLASS>( \
        sycl::queue&,                                                                           \
        std::int64_t,                                                                           \
        std::int64_t);

INSTANTIATE(float, std::int32_t);
INSTANTIATE(float, std::int64_t);
INSTANTIATE(double, std::int32_t);
INSTANTIATE(double, std::int64_t);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
