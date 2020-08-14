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

#include "oneapi/dal/table/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend {

class homogen_table_impl {
public:
    homogen_table_impl() : row_count_(0), col_count_(0) {}

    homogen_table_impl(std::int64_t p,
                       const array<byte_t>& data,
                       data_type dtype,
                       data_layout layout)
            : meta_(
                array<data_type>::full(p, dtype),
                array<feature_type>::full(p, detail::is_floating_point(dtype) ? feature_type::ratio : feature_type::ordinal)
            ),
              data_(data),
              row_count_(data.get_count() / p /
                         detail::get_data_type_size(dtype)),
              col_count_(p),
              layout_(layout) {}

    std::int64_t get_column_count() const {
        return col_count_;
    }

    std::int64_t get_row_count() const {
        return row_count_;
    }

    const table_metadata& get_metadata() const {
        return meta_;
    }

    const void* get_data() const {
        return data_.get_data();
    }

    data_layout get_data_layout() const {
        return layout_;
    }

    template <typename T>
    void pull_rows(array<T>& a, const range& r) const;

    template <typename T>
    void push_rows(const array<T>& a, const range& r);

    template <typename T>
    void pull_column(array<T>& a, std::int64_t idx, const range& r) const;

    template <typename T>
    void push_column(const array<T>& a, std::int64_t idx, const range& r);

#ifdef ONEAPI_DAL_DATA_PARALLEL
    template <typename T>
    void pull_rows(sycl::queue& q, array<T>& a, const range& r, const sycl::usm::alloc& kind) const;

    template <typename T>
    void push_rows(sycl::queue& q, const array<T>& a, const range& r);

    template <typename T>
    void pull_column(sycl::queue& q,
                     array<T>& a,
                     std::int64_t idx,
                     const range& r,
                     const sycl::usm::alloc& kind) const;

    template <typename T>
    void push_column(sycl::queue& q, const array<T>& a, std::int64_t idx, const range& r);
#endif

private:
    table_metadata meta_;
    array<byte_t> data_;
    int64_t row_count_;
    int64_t col_count_;
    data_layout layout_;
};

} // namespace oneapi::dal::backend
