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

template <typename DistType, typename ResponseType = float>
class distance_regression {
public:
    virtual sycl::event operator()(const ndview<ResponseType, 2>& responses,
                                   const ndview<DistType, 2>& distances,
                                   ndview<ResponseType, 1>& results,
                                   const event_vector& deps = {}) = 0;
    virtual ~distance_regression();

    distance_regression(sycl::queue& queue);

    sycl::queue& get_queue() const;

private:
    sycl::queue& queue_;
};

template <typename DistType, typename ResponseType = float>
class naive_distance_regression : public distance_regression<DistType, ResponseType> {
    using base_t = distance_regression<DistType, ResponseType>;

public:
    naive_distance_regression(sycl::queue& queue);
    sycl::event operator()(const ndview<ResponseType, 2>& responses,
                           const ndview<DistType, 2>& distances,
                           ndview<ResponseType, 1>& results,
                           const event_vector& deps = {}) final;
};

template <typename DistType, typename ResponseType = float>
std::unique_ptr<distance_regression<DistType, ResponseType>>
make_distance_regression(sycl::queue& queue, std::int64_t max_block, std::int64_t neighbor_count);

#endif

} // namespace oneapi::dal::backend::primitives
