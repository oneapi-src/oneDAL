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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

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
        get_impl().reset(detail::cast_impl<detail::homogen_table_iface>(std::move(t)));
        return *this;
    }

    template <typename Data>
    auto& reset(const array<Data>& data, std::int64_t row_count, std::int64_t column_count) {
        array<byte_t> byte_data;

        // TODO: Replace to reinterpret_array_cast
        if (data.has_mutable_data()) {
            byte_data.reset(data,
                            reinterpret_cast<byte_t*>(data.get_mutable_data()),
                            data.get_size());
        }
        else {
            byte_data.reset(data,
                            reinterpret_cast<const byte_t*>(data.get_data()),
                            data.get_size());
        }

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
                    const sycl::vector_class<sycl::event>& dependencies = {}) {
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

} // namespace v1

using v1::table_builder;
using v1::homogen_table_builder;

} // namespace oneapi::dal::detail
