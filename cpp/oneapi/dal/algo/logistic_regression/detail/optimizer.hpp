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

#include "oneapi/dal/algo/newton_cg/common.hpp"

namespace oneapi::dal::logistic_regression::detail {
namespace v1 {

class optimizer_impl;

class optimizer_iface {
public:
    virtual ~optimizer_iface() {}
    virtual optimizer_impl* get_impl() const = 0;
};

using optimizer_ptr = std::shared_ptr<optimizer_iface>;

template <typename Optimizer>
class optimizer : public base, public optimizer_iface {
public:
    explicit optimizer(const Optimizer& optimizer) : optimizer_(optimizer) {}

    optimizer_impl* get_impl() const override {
        return nullptr;
    }

    const Optimizer& get_optimizer() const {
        return optimizer_;
    }

private:
    Optimizer optimizer_;
    dal::detail::pimpl<optimizer_impl> impl_;
};

template <typename Float, typename Method>
class optimizer<newton_cg::descriptor<Float, Method>> : public base, public optimizer_iface {
public:
    using optimizer_t = newton_cg::descriptor<Float, Method>;
    explicit optimizer(const optimizer_t& opt);
    optimizer_impl* get_impl() const override;

private:
    optimizer_t optimizer_;
    dal::detail::pimpl<optimizer_impl> impl_;
};

struct optimizer_accessor {
    template <typename Optimizer>
    const optimizer_ptr& get_optimizer_impl(Optimizer&& desc) const {
        return desc.get_optimizer_impl();
    }
};

template <typename Descriptor>
optimizer_impl* get_optimizer_impl(Descriptor&& desc) {
    const auto& optimizer = optimizer_accessor{}.get_optimizer_impl(std::forward<Descriptor>(desc));
    return optimizer ? optimizer->get_impl() : nullptr;
}

} // namespace v1

using v1::optimizer_iface;
using v1::optimizer_ptr;
using v1::optimizer;
using v1::optimizer_accessor;
using v1::get_optimizer_impl;

} // namespace oneapi::dal::logistic_regression::detail
