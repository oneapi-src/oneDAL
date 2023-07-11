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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/chunked_array.hpp"

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal {
namespace v1 {

/*class ONEDAL_EXPORT heterogen_table : public table {
    friend detail::pimpl_accessor;

public:
    static std::int64_t kind();

    template <typename... Arrays>
    static heterogen_table wrap(Arrays&&... arrays) {
    }

    template <typename... Arrays>
    static heterogen_table wrap(std::int64_t column_count) {
    }

    template <typename T>
    chunked_array<T> get_column(std::int64_t i) const {

    }

    template <typename T>
    heterogen_table& set_column(std::int64_t i, array<T>&& arr) {
        return *this;
    }

    template <typename T>
    heterogen_table& set_column(std::int64_t i, chunked_array<T>&& arr) {
        return *this;
    }

    /// The unique id of the homogen table type.
    std::int64_t get_kind() const {
        return kind();
    }

private:
    template <typename Data>
    homogen_table(const dal::array<Data>& data,
                  std::int64_t row_count,
                  std::int64_t column_count,
                  data_layout layout = data_layout::row_major) {
        init_impl(data, row_count, column_count, layout);
    }

    explicit homogen_table(detail::homogen_table_iface* impl) : table(impl) {}
    explicit homogen_table(const detail::shared<detail::homogen_table_iface>& impl) : table(impl) {}

    template <typename Policy, typename Data, typename ConstDeleter>
    void init_impl(const Policy& policy,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   const Data* data_pointer,
                   ConstDeleter&& data_deleter,
                   data_layout layout) {
        validate_input_dimensions(row_count, column_count);

        const auto data = detail::array_via_policy<Data>::wrap(
            policy,
            data_pointer,
            detail::check_mul_overflow(row_count, column_count),
            std::forward<ConstDeleter>(data_deleter));

        init_impl(policy,
                  row_count,
                  column_count,
                  detail::reinterpret_array_cast<byte_t>(data),
                  detail::make_data_type<Data>(),
                  layout);
    }

    template <typename Data>
    void init_impl(const dal::array<Data>& data,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   data_layout layout) {
        validate_input_dimensions(row_count, column_count);

        if (data.get_count() < detail::check_mul_overflow(row_count, column_count)) {
            using msg = detail::error_messages;
            throw invalid_argument{ msg::rc_and_cc_do_not_match_element_count_in_array() };
        }

        detail::dispath_by_policy(data, [&](auto policy) {
            init_impl(policy,
                      row_count,
                      column_count,
                      detail::reinterpret_array_cast<byte_t>(data),
                      detail::make_data_type<Data>(),
                      layout);
        });
    }

    template <typename Policy>
    void init_impl(const Policy& policy,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   const dal::array<byte_t>& data,
                   const data_type& dtype,
                   data_layout layout);
};*/

} // namespace v1

//using v1::heterogen_table;

} // namespace oneapi::dal*/
