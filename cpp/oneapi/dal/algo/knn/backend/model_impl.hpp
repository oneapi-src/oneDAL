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

#include "oneapi/dal/algo/knn/common.hpp"
#include "oneapi/dal/algo/knn/backend/model_interop.hpp"
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::knn {

template <typename Task>
class detail::v1::model_impl : public base {
public:
    model_impl() {}
    model_impl(const model_impl&) = delete;
    model_impl& operator=(const model_impl&) = delete;

    virtual backend::model_interop* get_interop() = 0;
};

namespace detail {
namespace v1 {

template <typename Task>
class brute_force_model_impl : public model_impl<Task>, public ONEDAL_SERIALIZABLE(knn_brute_force_classification_model_impl_id) {
public:
    brute_force_model_impl() : interop_(nullptr) {}
    brute_force_model_impl(const brute_force_model_impl&) = delete;
    brute_force_model_impl& operator=(const brute_force_model_impl&) = delete;

    brute_force_model_impl(const table& data, const table& labels)
            : data_(data),
              labels_(labels),
              interop_(nullptr) {}

    ~brute_force_model_impl() {
        delete interop_;
        interop_ = nullptr;
    }

    backend::model_interop* get_interop() override {
        return interop_;
    }

    void serialize(dal::detail::output_archive& ar) const override {
        ar(data_, labels_);
        dal::detail::serialize_polymorphic(interop_, ar);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        ar(data_, labels_);
        interop_ = dal::detail::deserialize_polymorphic<backend::model_interop>(ar);
    }

    table data_;
    table labels_;
private:
    backend::model_interop* interop_;
};

template <typename Task>
class kdtree_model_impl : public model_impl<Task>, public ONEDAL_SERIALIZABLE(knn_kdtree_classification_model_impl_id) {
public:
    kdtree_model_impl() : interop_(nullptr) {}
    kdtree_model_impl(const kdtree_model_impl&) = delete;
    kdtree_model_impl& operator=(const kdtree_model_impl&) = delete;

    kdtree_model_impl(backend::model_interop* interop) : interop_(interop) {}

    ~kdtree_model_impl() {
        delete interop_;
        interop_ = nullptr;
    }

    backend::model_interop* get_interop() override {
        return interop_;
    }

    void serialize(dal::detail::output_archive& ar) const override {
        dal::detail::serialize_polymorphic(interop_, ar);
    }

    void deserialize(dal::detail::input_archive& ar) override {
        interop_ = dal::detail::deserialize_polymorphic<backend::model_interop>(ar);
    }

private:
    backend::model_interop* interop_;
};
} // namespace v1

using v1::brute_force_model_impl;
using v1::kdtree_model_impl;

} // namespace detail

namespace backend {

template <typename Task>
using model_impl = detail::model_impl<Task>;

template <typename Task>
using brute_force_model_impl = detail::brute_force_model_impl<Task>;

template <typename Task>
using kdtree_model_impl = detail::kdtree_model_impl<Task>;

} // namespace backend
} // namespace oneapi::dal::knn
