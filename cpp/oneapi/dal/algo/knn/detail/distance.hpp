/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/minkowski_distance/common.hpp"
#include "oneapi/dal/algo/chebychev_distance/common.hpp"

namespace oneapi::dal::knn::detail {
namespace v1 {

class distance_iface {
public:
    virtual ~distance_iface() {}
};

using distance_ptr = std::shared_ptr<distance_iface>;

template <typename Distance>
class distance : public base, public distance_iface {
    explicit distance(const Distance& distance) : distance_(distance) {}

    const Distance& get_distance() const {
        return distance_;
    }

private:
    Distance distance_;
};

template <typename Float, typename Method>
class distance<minkowski_distance::descriptor<Float, Method>> : public base, public distance_iface {
public:
    using kernel_t = minkowski_distance::descriptor<Float, Method>;
};

template <typename Float, typename Method>
class distance<chebychev_distance::descriptor<Float, Method>> : public base, public distance_iface {
public:
    using kernel_t = chebychev_distance::descriptor<Float, Method>;
};

} // namespace v1

using v1::distance_iface;
using v1::distance_ptr;
using v1::distance;

} // namespace oneapi::dal::knn::detail
