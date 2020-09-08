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

#pragma once

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::knn {

namespace detail {
struct tag {};
class descriptor_impl;
class model_impl;
} // namespace detail

namespace method {
struct kd_tree {};
struct brute_force {};
using by_default = kd_tree;
} // namespace method

class ONEAPI_DAL_EXPORT descriptor_base : public base {
public:
    using tag_t = detail::tag;
    using float_t = float;
    using method_t = method::by_default;

    descriptor_base();

    auto get_class_count() const -> std::int64_t;
    auto get_neighbor_count() const -> std::int64_t;
    auto get_data_use_in_model() const -> bool;

protected:
    void set_class_count_impl(std::int64_t value);
    void set_neighbor_count_impl(std::int64_t value);
    void set_data_use_in_model_impl(bool value);

    dal::detail::pimpl<detail::descriptor_impl> impl_;
};

template <typename Float = descriptor_base::float_t, typename Method = descriptor_base::method_t>
class descriptor : public descriptor_base {
public:
    using tag_t = detail::tag;
    using float_t = Float;
    using method_t = Method;

    auto& set_class_count(std::int64_t value) {
        set_class_count_impl(value);
        return *this;
    }

    auto& set_neighbor_count(std::int64_t value) {
        set_neighbor_count_impl(value);
        return *this;
    }

    auto& set_data_use_in_model(bool value) {
        set_data_use_in_model_impl(value);
        return *this;
    }
};

class ONEAPI_DAL_EXPORT model : public base {
    friend dal::detail::pimpl_accessor;

public:
    model();

private:
    explicit model(const std::shared_ptr<detail::model_impl>& impl);
    dal::detail::pimpl<detail::model_impl> impl_;
};

} // namespace oneapi::dal::knn
