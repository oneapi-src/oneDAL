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

#include "oneapi/dal/data/table_builder.hpp"

namespace oneapi::dal {

template <typename T>
class row_accessor {
private:
    using storage_t = detail::get_dense_storage_iface_t<T>;

public:
    using data_t = std::remove_const_t<T>;
    static constexpr bool is_readonly = std::is_const_v<T>;

    template <typename Q = T, typename = std::enable_if_t<sizeof(Q) && is_readonly>>
    row_accessor(const table& t)
        : storage_(detail::get_impl<storage_t>(t)) {}

    row_accessor(const table_builder& b)
        : storage_(detail::get_impl<detail::table_builder_impl_iface>(b).get_storage()) {}

    array<data_t> pull(const range& rows = {0, -1}) const {
        array<data_t> block;
        storage_.pull_rows(block, rows);
        return block;
    }

    T* pull(array<data_t>& block, const range& rows = {0, -1}) const {
        storage_.pull_rows(block, rows);
        if constexpr (is_readonly) {
            return block.get_data();
        } else {
            return block.get_mutable_data();
        }
    }

    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const array<data_t>& block,
                                                     const range& rows = {0, -1}) {
        storage_.push_back_rows(block, rows);
    }

private:
    storage_t& storage_;
};

template <typename T>
class column_accessor {
public:
    using storage_t = detail::get_dense_storage_iface_t<T>;

public:
    using data_t = std::remove_const_t<T>;
    static constexpr bool is_readonly = std::is_const_v<T>;

    template <typename Q = T, typename = std::enable_if_t<sizeof(Q) && is_readonly>>
    column_accessor(const table& t)
        : storage_(detail::get_impl<storage_t>(t)) {}

    column_accessor(const table_builder& b)
        : storage_(detail::get_impl<detail::table_builder_impl_iface>(b).get_storage()) {}

    array<data_t> pull(std::int64_t column_index, const range& rows = {0, -1}) const {
        array<data_t> block;
        storage_.pull_column(block, column_index, rows);
        return block;
    }

    T* pull(array<data_t>& block, std::int64_t column_index, const range& rows = {0, -1}) const {
        storage_.pull_column(block, column_index, rows);
        if constexpr (is_readonly) {
            return block.get_data();
        } else {
            return block.get_mutable_data();
        }
    }

    template <typename Q = T>
    std::enable_if_t<sizeof(Q) && !is_readonly> push(const array<data_t>& block,
                                                     std::int64_t column_index,
                                                     const range& rows = {0, -1}) {
        storage_.push_back_column(block, column_index, rows);
    }

private:
    storage_t& storage_;
};

} // namespace oneapi::dal
