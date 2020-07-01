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

#include "oneapi/dal/data/detail/table_impl_iface.hpp"
#include "oneapi/dal/data/detail/access_iface_wrappers_host.hpp"

namespace oneapi::dal::detail {

template <typename Impl>
class table_impl_wrapper : public table_impl_iface, public base {
public:
    table_impl_wrapper(Impl&& obj)
        : impl_(std::move(obj)),
          host_access_ptr_(new host_access_wrapper<Impl>{impl_}) {}

    virtual std::int64_t get_column_count() const override {
        return impl_.get_column_count();
    }

    virtual std::int64_t get_row_count() const override {
        return impl_.get_row_count();
    }

    virtual const table_metadata& get_metadata() const override {
        return impl_.get_metadata();
    }

    virtual std::int64_t get_kind() const override {
        return impl_.get_kind();
    }

    virtual host_access_iface& get_host_access_iface() const override {
        return *host_access_ptr_.get();
    }

    Impl& get() {
        return impl_;
    }

private:
    Impl impl_;
    unique<host_access_iface> host_access_ptr_;
};

template <typename Impl>
class homogen_table_impl_wrapper : public homogen_table_impl_iface, public base {
public:
    homogen_table_impl_wrapper(Impl&& obj, std::int64_t homogen_table_kind)
            : kind_(homogen_table_kind),
              impl_(std::move(obj)),
              host_access_ptr_(new host_access_wrapper<Impl>{impl_}) {}

    virtual std::int64_t get_column_count() const override {
        return impl_.get_column_count();
    }

    virtual std::int64_t get_row_count() const override {
        return impl_.get_row_count();
    }

    virtual const homogen_table_metadata& get_metadata() const override {
        return impl_.get_metadata();
    }

    virtual const void* get_data() const override {
        return impl_.get_data();
    }

    virtual std::int64_t get_kind() const override {
        return kind_;
    }

    virtual host_access_iface& get_host_access_iface() const override {
        return *host_access_ptr_.get();
    }

    Impl& get() {
        return impl_;
    }

private:
    const std::int64_t kind_;
    Impl impl_;
    unique<host_access_iface> host_access_ptr_;
};

} // namespace oneapi::dal::detail
