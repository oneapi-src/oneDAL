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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/heterogen.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class ONEDAL_EXPORT table_builder {
    friend pimpl_accessor;

public:
    table build() const {
        return detail::make_private<table>(impl_->build());
    }

protected:
    explicit table_builder(table_builder_iface* impl) : impl_(impl) {}

private:
    pimpl<table_builder_iface> impl_;
};

class ONEDAL_EXPORT homogen_table_builder : public table_builder {
public:
    homogen_table_builder();

    homogen_table build() {
        return detail::make_private<homogen_table>(get_impl().build_homogen());
    }

    auto& reset(homogen_table&& t) {
        const homogen_table local_table = std::move(t);

        const std::int64_t row_count = local_table.get_row_count();
        const std::int64_t column_count = local_table.get_column_count();
        const data_type dtype = local_table.get_metadata().get_data_type(0);
        const auto byte_data = get_original_data(local_table);

        get_impl().set_data_type(dtype);
        get_impl().reset(byte_data, row_count, column_count);
        return *this;
    }

    template <typename Data>
    auto& reset(const dal::array<Data>& data, std::int64_t row_count, std::int64_t column_count) {
        const auto byte_data = detail::reinterpret_array_cast<byte_t>(data);
        get_impl().set_data_type(detail::make_data_type<Data>());
        get_impl().reset(byte_data, row_count, column_count);
        return *this;
    }

    auto& set_data_type(data_type dt) {
        get_impl().set_data_type(dt);
        return *this;
    }

    auto& set_feature_type(feature_type ft) {
        get_impl().set_feature_type(ft);
        return *this;
    }

    auto& set_layout(data_layout layout) {
        get_impl().set_layout(layout);
        return *this;
    }

    auto& allocate(std::int64_t row_count, std::int64_t column_count) {
        get_impl().allocate(row_count, column_count);
        return *this;
    }

    template <typename Data>
    auto& copy_data(const Data* data, std::int64_t row_count, std::int64_t column_count) {
        get_impl().copy_data(data, row_count, column_count);
        return *this;
    }

    template <typename Data>
    auto& copy_data(const array<Data>& data) {
        get_impl().copy_data(reinterpret_array_cast<dal::byte_t>(data));
        return *this;
    }

#ifdef ONEDAL_DATA_PARALLEL
    auto& allocate(const sycl::queue& queue,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        get_impl().allocate(detail::data_parallel_policy{ queue }, row_count, column_count, alloc);
        return *this;
    }

    template <typename Data>
    auto& copy_data(sycl::queue& queue,
                    const Data* data,
                    std::int64_t row_count,
                    std::int64_t column_count,
                    const std::vector<sycl::event>& dependencies = {}) {
        sycl::event::wait_and_throw(dependencies);
        get_impl().copy_data(detail::data_parallel_policy{ queue }, data, row_count, column_count);
        return *this;
    }
#endif

private:
    homogen_table_builder_iface& get_impl() {
        return cast_impl<homogen_table_builder_iface>(*this);
    }
};

/*class ONEDAL_EXPORT heterogen_table_builder : public table_builder {
public:
    heterogen_table_builder();

    heterogen_table_builder(const table_metadata& meta);

    heterogen_table build() {
        //return detail::make_private<heterogen_table>( //
        //    get_impl().build_heterogen());
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    auto& reset(const table_metadata& meta) {
        //get_impl().reset_data();
        //get_impl().set_metadata(meta);

        return *this;
    }

    auto& reset(const heterogen_table& t) {
        const auto input_meta = t.get_metadata();
        const auto column_count = t.get_column_count();
        ONEDAL_ASSERT(column_count == input_meta.get_feature_count());

        this->reset(input_meta);

        for (std::int64_t c = 0l; c < column_count; ++ c) {
            chunked_array_base feature = t.get_column(c);
            this->set_feature(c, std::move(feature));
        }

        return *this;
    }

    auto& set_data_type(std::int64_t column, data_type dt) {
        //get_impl().set_data_type(column, dt);
        return *this;
    }

    auto& set_feature_type(std::int64_t column, feature_type ft) {
        //get_impl().set_feature_type(column, ft);
        return *this;
    }

    template <typename T>
    auto& set_feature(std::int64_t column, array<T> data) {
        //constexpr data_type dt = detail::make_data_type<T>();
        //get_impl().set_feature(column, dt, std::move(data));
        return *this;
    }

    template <typename T>
    auto& set_feature(std::int64_t column, chunked_array<T> data) {
        //constexpr data_type dt = detail::make_data_type<T>();
        //get_impl().set_feature(column, dt, std::move(data));
        return *this;
    }

private:
    heterogen_table_builder& set_feature(std::int64_t column, detail::chunked_array_base data) {
        //constexpr data_type dt = detail::make_data_type<T>();
        //get_impl().set_feature(column, dt, std::move(data));
        return *this;
    }

    heterogen_table_builder_iface& get_impl() {
        return cast_impl<heterogen_table_builder_iface>(*this);
    }
};*/

} // namespace v1

using v1::table_builder;
using v1::homogen_table_builder;
//using v1::heterogen_table_builder;

} // namespace oneapi::dal::detail
