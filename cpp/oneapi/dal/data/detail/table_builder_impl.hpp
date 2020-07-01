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

class table_builder_impl_iface : public access_provider_iface {
public:
    virtual table build() = 0;
};

template <typename Impl>
class table_builder_impl_wrapper : public table_builder_impl_iface, public base {
public:
    table_builder_impl_wrapper(Impl&& obj)
        : impl_(std::move(obj)) {}

    virtual table build() override {
        return impl_.build();
    }

    virtual access_iface_host& get_access_iface_host() const override {
        return impl_.get_access_iface_host();
        // TODO: need to re-design builder implementations
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    virtual access_iface_dpcpp& get_access_iface_dpcpp() const override {
        return impl_.get_access_iface_dpc();
    }
#endif

    Impl& get() {
        return impl_;
    }

private:
    Impl impl_;
};

} // namespace oneapi::dal::detail
