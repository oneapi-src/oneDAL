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

namespace dal::detail {

class table_builder_impl_iface {
public:
    using dense_rw_storage = detail::dense_storage_iface<storage_readable_writable>;
public:
    virtual ~table_builder_impl_iface() = default;

    virtual table build_table() = 0;
    virtual dense_rw_storage& get_storage() = 0;
};

template <typename Impl>
class table_builder_impl_wrapper : public table_builder_impl_iface,
                                   public base {
public:
    table_builder_impl_wrapper(Impl&& obj)
        : impl_(std::forward<Impl>(obj)) {}

    virtual table build_table() override {
        return impl_.build_table();
    }

    virtual dense_rw_storage& get_storage() override {
        return impl_.get_storage();
    }

    Impl& get() {
        return impl_;
    }

private:
    Impl impl_;
};

} // namespace dal::detail
