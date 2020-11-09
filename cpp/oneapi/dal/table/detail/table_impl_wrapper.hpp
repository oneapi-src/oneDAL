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

#include "oneapi/dal/table/detail/access_wrapper_dpc.hpp"
#include "oneapi/dal/table/detail/access_wrapper_host.hpp"
#include "oneapi/dal/table/detail/table_impl_iface.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename Impl>
class table_impl_wrapper : public table_impl_iface, public base {
public:
#ifdef ONEDAL_DATA_PARALLEL
    table_impl_wrapper(Impl&& obj)
            : impl_(std::move(obj)),
              host_access_ptr_(new access_wrapper_host<Impl>{ impl_ }),
              dpc_access_ptr_(new access_wrapper_dpc<Impl>{ impl_ }) {}
#else
    table_impl_wrapper(Impl&& obj)
            : impl_(std::move(obj)),
              host_access_ptr_(new access_wrapper_host<Impl>{ impl_ }) {}
#endif

    std::int64_t get_column_count() const override {
        return impl_.get_column_count();
    }

    std::int64_t get_row_count() const override {
        return impl_.get_row_count();
    }

    const table_metadata& get_metadata() const override {
        return impl_.get_metadata();
    }

    std::int64_t get_kind() const override {
        return impl_.get_kind();
    }

    data_layout get_data_layout() const override {
        return impl_.get_data_layout();
    }

    access_iface_host& get_access_iface_host() const override {
        return *host_access_ptr_.get();
    }

#ifdef ONEDAL_DATA_PARALLEL
    access_iface_dpc& get_access_iface_dpc() const override {
        return *dpc_access_ptr_.get();
    }
#endif

    Impl& get() {
        return impl_;
    }

private:
    Impl impl_;

    unique<access_iface_host> host_access_ptr_;
#ifdef ONEDAL_DATA_PARALLEL
    unique<access_iface_dpc> dpc_access_ptr_;
#endif
};

template <typename Impl>
class homogen_table_impl_wrapper : public homogen_table_impl_iface, public base {
public:
#ifdef ONEDAL_DATA_PARALLEL
    homogen_table_impl_wrapper(Impl&& obj, std::int64_t homogen_table_kind)
            : kind_(homogen_table_kind),
              impl_(std::move(obj)),
              host_access_ptr_(new access_wrapper_host<Impl>{ impl_ }),
              dpc_access_ptr_(new access_wrapper_dpc<Impl>{ impl_ }) {}
#else
    homogen_table_impl_wrapper(Impl&& obj, std::int64_t homogen_table_kind)
            : kind_(homogen_table_kind),
              impl_(std::move(obj)),
              host_access_ptr_(new access_wrapper_host<Impl>{ impl_ }) {}
#endif

    std::int64_t get_column_count() const override {
        return impl_.get_column_count();
    }

    std::int64_t get_row_count() const override {
        return impl_.get_row_count();
    }

    const table_metadata& get_metadata() const override {
        return impl_.get_metadata();
    }

    const void* get_data() const override {
        return impl_.get_data();
    }

    std::int64_t get_kind() const override {
        return kind_;
    }

    data_layout get_data_layout() const override {
        return impl_.get_data_layout();
    }

    access_iface_host& get_access_iface_host() const override {
        return *host_access_ptr_.get();
    }

#ifdef ONEDAL_DATA_PARALLEL
    access_iface_dpc& get_access_iface_dpc() const override {
        return *dpc_access_ptr_.get();
    }
#endif

    Impl& get() {
        return impl_;
    }

private:
    const std::int64_t kind_;
    Impl impl_;

    unique<access_iface_host> host_access_ptr_;
#ifdef ONEDAL_DATA_PARALLEL
    unique<access_iface_dpc> dpc_access_ptr_;
#endif
};

} // namespace v1

using v1::table_impl_wrapper;
using v1::homogen_table_impl_wrapper;

} // namespace oneapi::dal::detail
