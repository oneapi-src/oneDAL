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

#include "oneapi/dal/algo/minkowski_distance/common.hpp"
#include "oneapi/dal/algo/chebyshev_distance/common.hpp"
#include "oneapi/dal/algo/cosine_distance/common.hpp"

namespace oneapi::dal::knn::detail {
namespace v1 {

class distance_impl;

class distance_iface {
public:
    virtual ~distance_iface() {}
    virtual distance_impl* get_impl() const = 0;
};

using distance_ptr = std::shared_ptr<distance_iface>;

template <typename Distance>
class distance : public base, public distance_iface {
public:
    explicit distance(const Distance& distance) : distance_(distance) {}

    distance_impl* get_impl() const override {
        return nullptr;
    }

    const Distance& get_distance() const {
        return distance_;
    }

private:
    Distance distance_;
    dal::detail::pimpl<distance_impl> impl_;
};

template <typename Float, typename Method>
class distance<minkowski_distance::descriptor<Float, Method>> : public base, public distance_iface {
public:
    using distance_t = minkowski_distance::descriptor<Float, Method>;
    explicit distance(const distance_t& dist);
    distance_impl* get_impl() const override;

private:
    distance_t distance_;
    dal::detail::pimpl<distance_impl> impl_;
};

template <typename Float, typename Method>
class distance<chebyshev_distance::descriptor<Float, Method>> : public base, public distance_iface {
public:
    using distance_t = chebyshev_distance::descriptor<Float, Method>;
    explicit distance(const distance_t& dist);
    distance_impl* get_impl() const override;

private:
    distance_t distance_;
    dal::detail::pimpl<distance_impl> impl_;
};

template <typename Float, typename Method>
class distance<cosine_distance::descriptor<Float, Method>> : public base, public distance_iface {
public:
    using distance_t = cosine_distance::descriptor<Float, Method>;
    explicit distance(const distance_t& dist);
    distance_impl* get_impl() const override;

private:
    distance_t distance_;
    dal::detail::pimpl<distance_impl> impl_;
};

struct distance_accessor {
    template <typename Descriptor>
    const distance_ptr& get_distance_impl(Descriptor&& desc) const {
        return desc.get_distance_impl();
    }
};

template <typename Descriptor>
distance_impl* get_distance_impl(Descriptor&& desc) {
    const auto& distance = distance_accessor{}.get_distance_impl(std::forward<Descriptor>(desc));
    return distance ? distance->get_impl() : nullptr;
}

} // namespace v1

using v1::distance_iface;
using v1::distance_ptr;
using v1::distance;
using v1::distance_accessor;
using v1::get_distance_impl;

} // namespace oneapi::dal::knn::detail
