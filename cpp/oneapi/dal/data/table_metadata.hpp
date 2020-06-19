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

#include "onedapi/dal/common_helpers.hpp"
#include "oneapi/dal/data/array.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal {

enum class feature_type {
    nominal,
    ordinal,
    interval,
    ratio
};

namespace detail {
    class table_feature_impl;
    class table_metadata_impl;
} // namespace detail

class table_feature {
public:
    table_feature();
    table_feature(data_type);
    table_feature(data_type, feature_type);


    data_type get_data_type() const;
    table_feature& set_data_type(data_type);

    feature_type get_type() const;
    table_feature& set_type(feature_type);

private:
    detail::pimpl<detail::table_feature_impl> impl_;
};

template <typename T>
table_feature make_table_feature() {
    return table_feature { make_data_type<T>() };
}

class table_metadata {
    friend detail::pimpl_accessor;

public:
    table_metadata();

    table_metadata(const table_feature&,
                   std::int64_t feature_count = 1);

    table_metadata(array<table_feature> features);

    std::int64_t get_feature_count() const;
    const table_feature& get_feature(std::int64_t feature_index) const;

protected:
    table_metadata(const detail::pimpl<detail::table_metadata_impl>& impl)
        : impl_(impl) {}

private:
    detail::pimpl<detail::table_metadata_impl> impl_;
};

enum class homogen_data_layout {
    row_major,
    column_major
};

class homogen_table_metadata : public table_metadata {
public:
    homogen_table_metadata();

    homogen_table_metadata(const table_feature&,
                           homogen_data_layout,
                           std::int64_t feature_count = 1);

    homogen_data_layout get_data_layout() const;
    homogen_table_metadata& set_data_layout(homogen_data_layout);
};

} // namespace oneapi::dal
