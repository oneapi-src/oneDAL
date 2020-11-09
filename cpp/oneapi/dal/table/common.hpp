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

#include <type_traits>

#include "oneapi/dal/detail/common_dpc.hpp"
#include "oneapi/dal/table/detail/table_impl_wrapper.hpp"

namespace oneapi::dal {

namespace detail {
namespace v1 {

class table_metadata_impl;

template <typename T>
struct is_table_impl {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(std::int64_t, get_column_count, () const)
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(std::int64_t, get_row_count, () const)
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(const table_metadata&, get_metadata, () const)
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(std::int64_t, get_kind, () const)
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(data_layout, get_data_layout, () const)

    static constexpr bool value = has_method_get_column_count_v<T> &&
                                  has_method_get_row_count_v<T> && has_method_get_metadata_v<T> &&
                                  has_method_get_kind_v<T> && has_method_get_data_layout_v<T>;
};

template <typename T>
inline constexpr bool is_table_impl_v = is_table_impl<T>::value;

} // namespace v1

using v1::table_metadata_impl;
using v1::is_table_impl;
using v1::is_table_impl_v;

} // namespace detail

namespace v1 {

enum class feature_type { nominal, ordinal, interval, ratio };
enum class data_layout { unknown, row_major, column_major };

class ONEDAL_EXPORT table_metadata {
    friend detail::pimpl_accessor;
    using pimpl = detail::pimpl<detail::table_metadata_impl>;

public:
    table_metadata();
    table_metadata(const array<data_type>& dtypes, const array<feature_type>& ftypes);

    std::int64_t get_feature_count() const;
    const feature_type& get_feature_type(std::int64_t feature_index) const;
    const data_type& get_data_type(std::int64_t feature_index) const;

private:
    table_metadata(const pimpl& impl) : impl_(impl) {}

private:
    pimpl impl_;
};

class ONEDAL_EXPORT table {
    friend detail::pimpl_accessor;
    using pimpl = detail::pimpl<detail::table_impl_iface>;

public:
    table();
    table(const table&) = default;
    table(table&&);

    template <typename Impl,
              typename ImplType = std::decay_t<Impl>,
              typename = std::enable_if_t<detail::is_table_impl_v<ImplType> &&
                                          !std::is_base_of_v<table, ImplType>>>
    table(Impl&& impl) {
        init_impl(new detail::table_impl_wrapper(std::forward<Impl>(impl)));
    }

    table& operator=(const table&) = default;
    table& operator=(table&&);

    bool has_data() const noexcept;
    std::int64_t get_column_count() const;
    std::int64_t get_row_count() const;
    const table_metadata& get_metadata() const;
    std::int64_t get_kind() const;
    data_layout get_data_layout() const;

protected:
    table(const pimpl& impl) : impl_(impl) {}

    void init_impl(pimpl::element_type* impl);

private:
    pimpl impl_;
};

} // namespace v1

using v1::feature_type;
using v1::data_layout;
using v1::table_metadata;
using v1::table;

} // namespace oneapi::dal
