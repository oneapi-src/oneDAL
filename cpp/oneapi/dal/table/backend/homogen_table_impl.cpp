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

#include "oneapi/dal/table/backend/homogen_table_impl.hpp"
#include "oneapi/dal/table/backend/convert.hpp"

#include <cstring>

namespace oneapi::dal::backend {

using std::int32_t;

template <typename T>
void homogen_table_impl::pull_rows(array<T>& block, const range& rows) const {
    // TODO: check range correctness
    // TODO: check array size if non-zero

    const int64_t row_count = get_row_count();
    const int64_t column_count = get_column_count();
    const int64_t range_count = rows.get_element_count(row_count) * column_count;
    const data_type block_dtype = detail::make_data_type<T>();

    if (layout_ != data_layout::row_major) {
        throw std::runtime_error("unsupported data layout");
    }

    const auto feature_type = meta_.get_data_type(0);
    if (block_dtype == feature_type) {
        auto row_data = reinterpret_cast<const T*>(data_.get_data());
        auto row_start_pointer = row_data + rows.start_idx * column_count;
        block.reset(data_, row_start_pointer, range_count);
    }
    else {
        if (block.get_count() < range_count) {
            block.reset(range_count);
        }

        auto type_size = detail::get_data_type_size(feature_type);
        auto row_start_pointer = data_.get_data() + rows.start_idx * column_count * type_size;
        backend::convert_vector(row_start_pointer,
                                block.get_mutable_data(),
                                feature_type,
                                block_dtype,
                                range_count);
    }
}

template <typename T>
void homogen_table_impl::push_rows(const array<T>& block, const range& rows) {
    // TODO: check range correctness
    // TODO: check array size if non-zero

    const int64_t row_count = get_row_count();
    const int64_t column_count = get_column_count();
    const int64_t range_count = rows.get_element_count(row_count) * column_count;
    const data_type block_dtype = detail::make_data_type<T>();

    if (layout_ != data_layout::row_major) {
        throw std::runtime_error("unsupported data layout");
    }

    data_.need_mutable_data();
    const auto feature_type = meta_.get_data_type(0);
    if (block_dtype == feature_type) {
        auto row_data = reinterpret_cast<T*>(data_.get_mutable_data());
        auto row_start_pointer = row_data + rows.start_idx * column_count;

        if (row_start_pointer == block.get_data()) {
            return;
        }
        else {
            std::memcpy(row_start_pointer, block.get_data(), range_count * sizeof(T));
        }
    }
    else {
        const auto type_size = detail::get_data_type_size(feature_type);
        auto row_start_pointer =
            data_.get_mutable_data() + rows.start_idx * column_count * type_size;

        backend::convert_vector(block.get_data(),
                                row_start_pointer,
                                block_dtype,
                                feature_type,
                                range_count);
    }
}

template <typename T>
void homogen_table_impl::pull_column(array<T>& block, int64_t idx, const range& rows) const {
    // TODO: check inputs

    const int64_t row_count = get_row_count();
    const int64_t column_count = get_column_count();
    const int64_t range_count = rows.get_element_count(row_count);
    const data_type block_dtype = detail::make_data_type<T>();

    if (layout_ != data_layout::row_major) {
        throw std::runtime_error("unsupported data layout");
    }

    const auto feature_type = meta_.get_data_type(0);
    if (block_dtype == feature_type && column_count == 1) {
        // TODO: assert idx == 0

        auto col_data = reinterpret_cast<const T*>(data_.get_data());
        block.reset(data_, col_data + rows.start_idx * column_count, range_count);
    }
    else {
        if (block.get_count() < range_count) {
            block.reset(range_count);
        }

        auto src_ptr = data_.get_data() + detail::get_data_type_size(feature_type) *
                                              (idx + rows.start_idx * column_count);
        backend::convert_vector(src_ptr,
                                block.get_mutable_data(),
                                feature_type,
                                block_dtype,
                                detail::get_data_type_size(feature_type) * column_count,
                                sizeof(T),
                                range_count);
    }
}

template <typename T>
void homogen_table_impl::push_column(const array<T>& block, int64_t idx, const range& rows) {
    // TODO: check inputs

    const int64_t row_count = get_row_count();
    const int64_t column_count = get_column_count();
    const int64_t range_count = rows.get_element_count(row_count);
    const data_type block_dtype = detail::make_data_type<T>();

    auto feature_type = meta_.get_data_type(0);
    const int64_t row_offset =
        detail::get_data_type_size(feature_type) * (idx + rows.start_idx * column_count);

    if (block_dtype == feature_type && column_count == 1) {
        if (reinterpret_cast<const void*>(data_.get_data() + row_offset) !=
            reinterpret_cast<const void*>(block.get_data())) {
            data_.need_mutable_data();
            auto dst_ptr = data_.get_mutable_data() + row_offset;
            backend::convert_vector(block.get_data(),
                                    dst_ptr,
                                    block_dtype,
                                    feature_type,
                                    range_count);
        }
    }
    else {
        data_.need_mutable_data();
        auto dst_ptr = data_.get_mutable_data() + row_offset;
        backend::convert_vector(block.get_data(),
                                dst_ptr,
                                block_dtype,
                                feature_type,
                                sizeof(T),
                                detail::get_data_type_size(feature_type) * column_count,
                                range_count);
    }
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <typename T>
void homogen_table_impl::pull_rows(sycl::queue& q,
                                   array<T>& a,
                                   const range& r,
                                   const sycl::usm::alloc& kind) const {
    pull_rows(a, r);
}

template <typename T>
void homogen_table_impl::push_rows(sycl::queue& q, const array<T>& a, const range& r) {
    push_rows(a, r);
}

template <typename T>
void homogen_table_impl::pull_column(sycl::queue& q,
                                     array<T>& a,
                                     std::int64_t idx,
                                     const range& r,
                                     const sycl::usm::alloc& kind) const {
    pull_column(a, idx, r);
}

template <typename T>
void homogen_table_impl::push_column(sycl::queue& q,
                                     const array<T>& a,
                                     std::int64_t idx,
                                     const range& r) {
    push_column(a, idx, r);
}
#endif

template void homogen_table_impl::pull_rows(array<float>&, const range&) const;
template void homogen_table_impl::pull_rows(array<double>&, const range&) const;
template void homogen_table_impl::pull_rows(array<int32_t>&, const range&) const;

template void homogen_table_impl::push_rows(const array<float>&, const range&);
template void homogen_table_impl::push_rows(const array<double>&, const range&);
template void homogen_table_impl::push_rows(const array<int32_t>&, const range&);

template void homogen_table_impl::pull_column(array<float>&, std::int64_t, const range&) const;
template void homogen_table_impl::pull_column(array<double>&, std::int64_t, const range&) const;
template void homogen_table_impl::pull_column(array<int32_t>&, std::int64_t, const range&) const;

template void homogen_table_impl::push_column(const array<float>&, std::int64_t, const range&);
template void homogen_table_impl::push_column(const array<double>&, std::int64_t, const range&);
template void homogen_table_impl::push_column(const array<int32_t>&, std::int64_t, const range&);

#ifdef ONEAPI_DAL_DATA_PARALLEL
template void homogen_table_impl::pull_rows(sycl::queue&,
                                            array<float>&,
                                            const range&,
                                            const sycl::usm::alloc&) const;
template void homogen_table_impl::pull_rows(sycl::queue&,
                                            array<double>&,
                                            const range&,
                                            const sycl::usm::alloc&) const;
template void homogen_table_impl::pull_rows(sycl::queue&,
                                            array<int32_t>&,
                                            const range&,
                                            const sycl::usm::alloc&) const;

template void homogen_table_impl::push_rows(sycl::queue&, const array<float>&, const range&);
template void homogen_table_impl::push_rows(sycl::queue&, const array<double>&, const range&);
template void homogen_table_impl::push_rows(sycl::queue&, const array<int32_t>&, const range&);

template void homogen_table_impl::pull_column(sycl::queue&,
                                              array<float>&,
                                              std::int64_t,
                                              const range&,
                                              const sycl::usm::alloc&) const;
template void homogen_table_impl::pull_column(sycl::queue&,
                                              array<double>&,
                                              std::int64_t,
                                              const range&,
                                              const sycl::usm::alloc&) const;
template void homogen_table_impl::pull_column(sycl::queue&,
                                              array<int32_t>&,
                                              std::int64_t,
                                              const range&,
                                              const sycl::usm::alloc&) const;

template void homogen_table_impl::push_column(sycl::queue&,
                                              const array<float>&,
                                              std::int64_t,
                                              const range&);
template void homogen_table_impl::push_column(sycl::queue&,
                                              const array<double>&,
                                              std::int64_t,
                                              const range&);
template void homogen_table_impl::push_column(sycl::queue&,
                                              const array<int32_t>&,
                                              std::int64_t,
                                              const range&);
#endif

} // namespace oneapi::dal::backend
