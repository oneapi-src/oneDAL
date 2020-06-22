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
#include "oneapi/dal/data/detail/table_builder_impl.hpp"

namespace oneapi::dal {

template <typename T>
struct is_table_builder_impl {
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(table, build_table, ())
    INSTANTIATE_HAS_METHOD_DEFAULT_CHECKER(detail::dense_storage_iface<detail::storage_readable_writable>&, get_storage, ())

    static constexpr bool value = has_method_build_table_v<T> && has_method_get_storage_v<T>;
};

template <typename T>
inline constexpr bool is_table_builder_impl_v = is_table_builder_impl<T>::value;

class ONEAPI_DAL_EXPORT table_builder {
    friend detail::pimpl_accessor;

public:
    template <typename BuilderImpl,
              typename = std::enable_if_t<is_table_builder_impl_v<std::decay_t<BuilderImpl>>>>
    table_builder(BuilderImpl&& impl) {
        init_impl(new detail::table_builder_impl_wrapper(std::forward<BuilderImpl>(impl)));
    }

    table_builder(table&&);

    table build() const {
        return impl_->build_table();
    }

private:
    void init_impl(detail::table_builder_impl_iface* obj) {
        impl_ = detail::pimpl<detail::table_builder_impl_iface>{ obj };
    }

private:
    detail::pimpl<detail::table_builder_impl_iface> impl_;
};

class ONEAPI_DAL_EXPORT homogen_table_builder : public table_builder {
public:
    // TODO: revise - const DataType* or DataType*
    template <typename DataType>
    homogen_table_builder(std::int64_t row_count, std::int64_t column_count,
                          const DataType* data_pointer,
                          data_layout layout = data_layout::row_major);

    template <typename DataType, typename = std::enable_if_t<!std::is_pointer_v<DataType>>>
    homogen_table_builder(std::int64_t row_count, std::int64_t column_count,
                          DataType value,
                          data_layout layout = data_layout::row_major);

    template <typename DataType>
    homogen_table_builder(std::int64_t column_count, const array<DataType>& data,
                          data_layout layout = data_layout::row_major);

    // TODO:
    // homogen_table_builder(std::int64_t row_count, std::int64_t column_count,
    //                       const void* data_pointer, const feature_info& finfo,
    //                       data_layout layout = data_layout::row_major);

    homogen_table_builder(homogen_table&&);

    homogen_table build() const;
};

} // namespace oneapi::dal
