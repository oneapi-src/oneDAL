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

#include "oneapi/dal/table/detail/rows_access_iface.hpp"
#include "oneapi/dal/table/detail/columns_access_iface.hpp"

namespace oneapi::dal {
namespace v1 {
class table_metadata;
enum class data_layout;
enum class feature_type;
} // namespace v1

using v1::table_metadata;
using v1::data_layout;
using v1::feature_type;

} // namespace oneapi::dal

namespace oneapi::dal::detail {
namespace v1 {

class table_iface {
public:
    virtual ~table_iface() = default;
    virtual std::int64_t get_column_count() const = 0;
    virtual std::int64_t get_row_count() const = 0;
    virtual std::int64_t get_kind() const = 0;
    virtual data_layout get_data_layout() const = 0;
    virtual const table_metadata& get_metadata() const = 0;
    virtual pull_rows_iface* get_pull_rows_iface() = 0;
    virtual pull_column_iface* get_pull_column_iface() = 0;
};

class homogen_table_iface : public table_iface {
public:
    virtual array<byte_t> get_data() const = 0;
};

class table_builder_iface {
public:
    virtual ~table_builder_iface() = default;
    virtual table_iface* build() = 0;
    virtual pull_rows_iface* get_pull_rows_iface() = 0;
    virtual pull_column_iface* get_pull_column_iface() = 0;
    virtual push_rows_iface* get_push_rows_iface() = 0;
    virtual push_column_iface* get_push_column_iface() = 0;
};

class homogen_table_builder_iface : public table_builder_iface {
public:
    virtual homogen_table_iface* build_homogen() = 0;

    virtual void set_data_type(data_type dt) = 0;
    virtual void set_layout(data_layout layout) = 0;
    virtual void set_feature_type(feature_type ft) = 0;

    virtual void reset(const array<byte_t>& data,
                       std::int64_t row_count,
                       std::int64_t column_count) = 0;

    virtual void allocate(std::int64_t row_count, //
                          std::int64_t column_count) = 0;

    virtual void copy_data(const void* data, //
                           std::int64_t row_count,
                           std::int64_t column_count) = 0;

#ifdef ONEDAL_DATA_PARALLEL
    virtual void allocate(const data_parallel_policy& policy,
                          std::int64_t row_count,
                          std::int64_t column_count,
                          sycl::usm::alloc alloc) = 0;

    virtual void copy_data(const data_parallel_policy& policy,
                           const void* data,
                           std::int64_t row_count,
                           std::int64_t column_count) = 0;
#endif
};

template <typename Iface, typename Derived>
class table_template : public Iface,
                       public pull_rows_template<Derived>,
                       public pull_column_template<Derived> {
public:
    pull_rows_iface* get_pull_rows_iface() override {
        return this;
    }

    pull_column_iface* get_pull_column_iface() override {
        return this;
    }
};

template <typename Iface, typename Derived>
class table_builder_template : public Iface,
                               public pull_rows_template<Derived>,
                               public pull_column_template<Derived>,
                               public push_rows_template<Derived>,
                               public push_column_template<Derived> {
public:
    pull_rows_iface* get_pull_rows_iface() override {
        return this;
    }

    pull_column_iface* get_pull_column_iface() override {
        return this;
    }

    push_rows_iface* get_push_rows_iface() override {
        return this;
    }

    push_column_iface* get_push_column_iface() override {
        return this;
    }
};

template <typename Object>
inline std::shared_ptr<homogen_table_iface> get_homogen_table_iface(Object&& obj) {
    const auto pimpl = pimpl_accessor{}.get_pimpl(std::forward<Object>(obj));
    auto homogen_iface_ptr = dynamic_cast<homogen_table_iface*>(pimpl.get());
    return std::shared_ptr<homogen_table_iface>{ pimpl, homogen_iface_ptr };
}

} // namespace v1

using v1::table_iface;
using v1::homogen_table_iface;
using v1::table_template;
using v1::table_builder_iface;
using v1::homogen_table_builder_iface;
using v1::table_builder_template;
using v1::get_homogen_table_iface;

} // namespace oneapi::dal::detail
