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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename DistType, typename ClassType = std::int32_t>
class distance_voting {
public:
    virtual sycl::event operator()(const ndview<ClassType, 2>& responses,
                                   const ndview<DistType, 2>& distances,
                                   ndview<ClassType, 1>& results,
                                   const event_vector& deps = {}) = 0;
    virtual ~distance_voting();

    distance_voting(sycl::queue& queue, std::int64_t class_count);

    sycl::queue& get_queue() const;
    std::int64_t get_class_count() const;

private:
    sycl::queue& queue_;
    std::int64_t class_count_;
};

template <typename DistType, typename ClassType = std::int32_t>
class naive_distance_voting : public distance_voting<DistType, ClassType> {
    using base_t = distance_voting<DistType, ClassType>;

public:
    naive_distance_voting(sycl::queue& queue, std::int64_t max_block, std::int64_t class_count);
    sycl::event operator()(const ndview<ClassType, 2>& responses,
                           const ndview<DistType, 2>& distances,
                           ndview<ClassType, 1>& results,
                           const event_vector& deps = {}) final;

    ndview<DistType, 2>& get_global_probas();

private:
    ndarray<DistType, 2> global_probas_;
};

template <typename DistType, typename ClassType = std::int32_t>
std::unique_ptr<distance_voting<DistType, ClassType>>
make_distance_voting(sycl::queue& queue, std::int64_t max_block, std::int64_t class_count);

#endif

} // namespace oneapi::dal::backend::primitives
