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

#include <utility>

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
        using detail::integral_cast;

        const std::size_t ccount = sizeof...(Arrays);
        const auto count = integral_cast<std::int64_t>(ccount);
        auto result = heterogen_table::empty(count);

        ([&](){ result.set_column( //
            std::forward<Arrays>(arrays) ); }(), ...);

        return result;
    }

    static heterogen_table empty(std::int64_t column_count) {
        throw dal::unimplemented(dal::detail::error_messages::method_not_implemented());
    }

    template <typename T>
    chunked_array<T> get_column(std::int64_t column) const {
        auto [dtype, arr] = this->get_column_impl(column);
        ONEDAL_ASSERT(dtype == detail::make_data_type<T>());
        return chunked_array<T>{ std::move(arr) };
    }

    template <typename T>
    heterogen_table& set_column(std::int64_t column, array<T> arr) {
        auto as_chunked = chunked_array<T>{ std::move(arr) };
        this->set_column(column, std::move(as_chunked) );
        return *this;
    }

    template <typename T>
    heterogen_table& set_column(std::int64_t column, chunked_array<T> arr) {
        this->set_column_impl(column, std::move(arr));
        return *this;
    }

    /// The unique id of the homogen table type.
    std::int64_t get_kind() const {
        return kind();
    }

private:
    void set_column_impl(std::int64_t column, data_type dt, detail::chunked_array_base&& arr);
    std::pair<data_type, detail::chunked_array_base> get_column_impl(std::int64_t column) const;

    explicit heterogen_table(detail::heterogen_table_iface* impl) : table(impl) {}
    explicit heterogen_table(const detail::shared<detail::heterogen_table_iface>& impl) : table(impl) {}
};*/

} // namespace v1

//using v1::heterogen_table;

} // namespace oneapi::dal*/
