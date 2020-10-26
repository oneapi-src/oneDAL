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

table_metadata create_homogen_metadata(int64_t feature_count, data_type dtype) {
    auto default_ftype =
        detail::is_floating_point(dtype) ? feature_type::ratio : feature_type::ordinal;

    auto dtypes = array<data_type>::full(feature_count, dtype);
    auto ftypes = array<feature_type>::full(feature_count, default_ftype);
    return table_metadata{ dtypes, ftypes };
}

template <typename Policy, typename Data>
void make_mutable_data(const Policy& policy, array<Data>& array) {
    if constexpr (std::is_same_v<Policy, detail::default_host_policy>) {
        array.need_mutable_data();
    }
#ifdef ONEDAL_DATA_PARALLEL
    else if constexpr (std::is_same_v<Policy, detail::data_parallel_policy>) {
        auto& queue = policy.get_queue();
        auto kind = sycl::get_pointer_type(array.get_data(), queue.get_context());
        array.need_mutable_data(queue, kind);
    }
#endif
    else {
        static_assert("make_mutable_data(): undefined policy type");
    }
}

template <typename Policy, typename Data, typename Alloc>
void reset_array(const Policy& policy, array<Data>& array, int64_t count, const Alloc& kind) {
    if constexpr (std::is_same_v<Policy, detail::default_host_policy>) {
        array.reset(count);
    }
#ifdef ONEDAL_DATA_PARALLEL
    else if constexpr (std::is_same_v<Policy, detail::data_parallel_policy>) {
        array.reset(policy.get_queue(), count, kind);
    }
#endif
    else {
        static_assert("reset_array(): undefined policy type");
    }
}

template <typename Policy, typename Data, typename Alloc>
bool has_array_data_kind(const Policy& policy, const array<Data>& array, const Alloc& kind) {
    if (array.get_count() <= 0) {
        return false;
    }

    if constexpr (std::is_same_v<Policy, detail::default_host_policy>) {
        // We assume that no sycl::usm::alloc::device pointers used with host policies.
        // It is responsibility of user to pass right pointers because we cannot check
        // the right pointer type with the host policy.
        return true;
    }
#ifdef ONEDAL_DATA_PARALLEL
    else if constexpr (std::is_same_v<Policy, detail::data_parallel_policy>) {
        static_assert(std::is_same_v<Alloc, sycl::usm::alloc>);
        auto array_data_kind =
            sycl::get_pointer_type(array.get_data(), policy.get_queue().get_context());
        return array_data_kind == kind;
    }
#endif
    else {
        static_assert("has_array_data_kind(): undefined policy type");
    }

    return false;
}

template <typename Policy, typename Data, typename Alloc>
void homogen_table_impl::pull_rows_impl(const Policy& policy,
                                        array<Data>& block,
                                        const range& rows,
                                        const Alloc& kind) const {
    // TODO: check range correctness
    // TODO: check array size if non-zero

    const int64_t range_size = rows.get_element_count(row_count_) * col_count_;
    const data_type block_dtype = detail::make_data_type<Data>();

    if (layout_ != data_layout::row_major) {
        throw std::runtime_error("unsupported data layout"); // TODO: oneDAL exception
    }

    const auto table_dtype = meta_.get_data_type(0);
    if (block_dtype == table_dtype && has_array_data_kind(policy, data_, kind)) {
        if (data_.has_mutable_data()) {
            auto row_data = reinterpret_cast<Data*>(data_.get_mutable_data());
            auto row_start_pointer = row_data + rows.start_idx * col_count_;
            block.reset(data_, row_start_pointer, range_size);
        }
        else {
            auto row_data = reinterpret_cast<const Data*>(data_.get_data());
            auto row_start_pointer = row_data + rows.start_idx * col_count_;
            block.reset(data_, row_start_pointer, range_size);
        }
    }
    else {
        if (block.get_count() < range_size || block.has_mutable_data() == false ||
            has_array_data_kind(policy, block, kind) == false) {
            reset_array(policy, block, range_size, kind);
        }

        auto type_size = detail::get_data_type_size(table_dtype);
        auto row_start_pointer = data_.get_data() + rows.start_idx * col_count_ * type_size;
        backend::convert_vector(policy,
                                row_start_pointer,
                                block.get_mutable_data(),
                                table_dtype,
                                block_dtype,
                                range_size);
    }
}

template <typename Policy, typename Data>
void homogen_table_impl::push_rows_impl(const Policy& policy,
                                        const array<Data>& block,
                                        const range& rows) {
    // TODO: check range correctness
    // TODO: check array size if non-zero

    const int64_t row_count = get_row_count();
    const int64_t column_count = get_column_count();
    const int64_t range_count = rows.get_element_count(row_count) * column_count;
    const data_type block_dtype = detail::make_data_type<Data>();

    if (layout_ != data_layout::row_major) {
        throw std::runtime_error("unsupported data layout");
    }

    make_mutable_data(policy, data_);

    const auto table_dtype = meta_.get_data_type(0);
    if (block_dtype == table_dtype) {
        auto row_data = reinterpret_cast<Data*>(data_.get_mutable_data());
        auto row_start_pointer = row_data + rows.start_idx * column_count;

        if (row_start_pointer == block.get_data()) {
            return;
        }
        else {
            detail::memcpy(policy, row_start_pointer, block.get_data(), range_count * sizeof(Data));
        }
    }
    else {
        const auto type_size = detail::get_data_type_size(table_dtype);
        auto row_start_pointer =
            data_.get_mutable_data() + rows.start_idx * column_count * type_size;

        backend::convert_vector(policy,
                                block.get_data(),
                                row_start_pointer,
                                block_dtype,
                                table_dtype,
                                range_count);
    }
}

template <typename Policy, typename Data, typename Alloc>
void homogen_table_impl::pull_column_impl(const Policy& policy,
                                          array<Data>& block,
                                          int64_t column_index,
                                          const range& rows,
                                          const Alloc& kind) const {
    // TODO: check inputs

    const int64_t row_count = get_row_count();
    const int64_t column_count = get_column_count();
    const int64_t range_count = rows.get_element_count(row_count);
    const data_type block_dtype = detail::make_data_type<Data>();

    if (layout_ != data_layout::row_major) {
        throw std::runtime_error("unsupported data layout");
    }

    const auto table_dtype = meta_.get_data_type(0);
    if (block_dtype == table_dtype && column_count == 1 &&
        has_array_data_kind(policy, data_, kind)) {
        // TODO: assert column_index == 0

        if (data_.has_mutable_data()) {
            auto col_data = reinterpret_cast<Data*>(data_.get_mutable_data());
            block.reset(data_, col_data + rows.start_idx * column_count, range_count);
        }
        else {
            auto col_data = reinterpret_cast<const Data*>(data_.get_data());
            block.reset(data_, col_data + rows.start_idx * column_count, range_count);
        }
    }
    else {
        if (block.get_count() < range_count || block.has_mutable_data() == false ||
            has_array_data_kind(policy, block, kind) == false) {
            reset_array(policy, block, range_count, kind);
        }

        auto src_ptr = data_.get_data() + detail::get_data_type_size(table_dtype) *
                                              (column_index + rows.start_idx * column_count);
        backend::convert_vector(policy,
                                src_ptr,
                                block.get_mutable_data(),
                                table_dtype,
                                block_dtype,
                                detail::get_data_type_size(table_dtype) * column_count,
                                sizeof(Data),
                                range_count);
    }
}

template <typename Policy, typename Data>
void homogen_table_impl::push_column_impl(const Policy& policy,
                                          const array<Data>& block,
                                          int64_t column_index,
                                          const range& rows) {
    // TODO: check inputs

    const int64_t row_count = get_row_count();
    const int64_t column_count = get_column_count();
    const int64_t range_count = rows.get_element_count(row_count);
    const data_type block_dtype = detail::make_data_type<Data>();

    auto table_dtype = meta_.get_data_type(0);
    const int64_t row_offset =
        detail::get_data_type_size(table_dtype) * (column_index + rows.start_idx * column_count);

    if (block_dtype == table_dtype && column_count == 1) {
        if (reinterpret_cast<const void*>(data_.get_data() + row_offset) !=
            reinterpret_cast<const void*>(block.get_data())) {
            make_mutable_data(policy, data_);
            auto dst_ptr = data_.get_mutable_data() + row_offset;
            backend::convert_vector(policy,
                                    block.get_data(),
                                    dst_ptr,
                                    block_dtype,
                                    table_dtype,
                                    range_count);
        }
    }
    else {
        make_mutable_data(policy, data_);

        auto dst_ptr = data_.get_mutable_data() + row_offset;
        backend::convert_vector(policy,
                                block.get_data(),
                                dst_ptr,
                                block_dtype,
                                table_dtype,
                                sizeof(Data),
                                detail::get_data_type_size(table_dtype) * column_count,
                                range_count);
    }
}

#define INSTANTIATE_IMPL(Policy, Data, Alloc)                               \
    template void homogen_table_impl::pull_rows_impl(const Policy&,         \
                                                     array<Data>&,          \
                                                     const range&,          \
                                                     const Alloc&) const;   \
    template void homogen_table_impl::push_rows_impl(const Policy&,         \
                                                     const array<Data>&,    \
                                                     const range&);         \
    template void homogen_table_impl::pull_column_impl(const Policy&,       \
                                                       array<Data>&,        \
                                                       int64_t,             \
                                                       const range&,        \
                                                       const Alloc&) const; \
    template void homogen_table_impl::push_column_impl(const Policy&,       \
                                                       const array<Data>&,  \
                                                       int64_t,             \
                                                       const range&);

#ifdef ONEDAL_DATA_PARALLEL
#define INSTANTIATE_IMPL_ALL_POLICIES(Data)                                               \
    INSTANTIATE_IMPL(detail::default_host_policy, Data, homogen_table_impl::host_alloc_t) \
    INSTANTIATE_IMPL(detail::data_parallel_policy, Data, sycl::usm::alloc)
#else
#define INSTANTIATE_IMPL_ALL_POLICIES(Data) \
    INSTANTIATE_IMPL(detail::default_host_policy, Data, homogen_table_impl::host_alloc_t)
#endif

INSTANTIATE_IMPL_ALL_POLICIES(float)
INSTANTIATE_IMPL_ALL_POLICIES(double)
INSTANTIATE_IMPL_ALL_POLICIES(int32_t)

#undef INSTANTIATE_IMPL_ALL_POLICIES
#undef INSTANTIATE_IMPL

} // namespace oneapi::dal::backend
