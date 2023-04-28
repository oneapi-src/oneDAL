/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/logloss_objective/common.hpp"

namespace oneapi::dal::objective_function::detail {
namespace v1 {

class objective_impl;

class objective_iface {
public:
    virtual ~objective_iface() {}
    virtual objective_impl* get_impl() const = 0;
};

using objective_ptr = std::shared_ptr<objective_iface>;

template <typename Objective>
class objective : public base, public objective_iface {
public:
    explicit objective(const Objective& objective) : objective_(objective) {}

    objective_impl* get_impl() const override {
        return nullptr;
    }

    const Objective& get_objective() const {
        return objective_;
    }

private:
    Objective objective_;
    dal::detail::pimpl<objective_impl> impl_;
};

template <typename Float, typename Method>
class objective<logloss_objective::descriptor<Float, Method>> : public base,
                                                                public objective_iface {
public:
    using objective_t = logloss_objective::descriptor<Float, Method>;
    explicit objective(const objective_t& obj);
    objective_impl* get_impl() const override;

private:
    objective_t objective_;
    dal::detail::pimpl<objective_impl> impl_;
};

struct objective_accessor {
    template <typename Descriptor>
    const objective_ptr& get_objective_impl(Descriptor&& desc) const {
        return desc.get_objective_impl();
    }
};

template <typename Descriptor>
objective_impl* get_objective_impl(Descriptor&& desc) {
    const auto& objective = objective_accessor{}.get_objective_impl(std::forward<Descriptor>(desc));
    return objective ? objective->get_impl() : nullptr;
}

} // namespace v1

using v1::objective_iface;
using v1::objective_ptr;
using v1::objective;
using v1::objective_accessor;
using v1::get_objective_impl;

} // namespace oneapi::dal::objective_function::detail
