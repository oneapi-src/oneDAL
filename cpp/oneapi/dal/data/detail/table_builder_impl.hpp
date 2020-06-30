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

#include "oneapi/dal/data/table.hpp"

namespace oneapi::dal::detail {

class table_builder_impl_iface : public accessible_iface {
public:
    virtual table build() = 0;
};

template <typename Impl>
class table_builder_impl_wrapper : public table_builder_impl_iface, public base {
private:
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(const host_access_iface&, get_host_access_iface, () const)

public:
    table_builder_impl_wrapper(Impl&& obj)
        : impl_(std::move(obj)) {
        if constexpr (has_method_get_host_access_iface_v<Impl>) {
            access_iface_ = impl_.get_host_access_iface();
        } else {
            access_iface_ = make_host_access_iface(impl_);
        }
    }

    virtual table build() override {
        return impl_.build();
    }

    virtual const host_access_iface& get_host_access_iface() const override {
        return access_iface_;
    }

    Impl& get() {
        return impl_;
    }

private:
    Impl impl_;
    host_access_iface access_iface_;
};

} // namespace oneapi::dal::detail
