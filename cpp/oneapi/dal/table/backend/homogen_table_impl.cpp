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
#ifdef ONEAPI_DAL_DATA_PARALLEL
    else if constexpr (std::is_same_v<Policy, detail::data_parallel_policy>) {
        auto queue = policy.get_queue();
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
#ifdef ONEAPI_DAL_DATA_PARALLEL
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
#ifdef ONEAPI_DAL_DATA_PARALLEL
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

template <typename DataSrc, typename DataDest>
void refer_source_data(const array<DataSrc>& src,
                       std::int64_t src_start_index,
                       std::int64_t dest_count,
                       array<DataDest>& dest) {
    if (src.has_mutable_data()) {
        auto start_pointer = reinterpret_cast<DataDest*>(src.get_mutable_data() + src_start_index);
        dest.reset(src, start_pointer, dest_count);
    }
    else {
        auto start_pointer = reinterpret_cast<const DataDest*>(src.get_data() + src_start_index);
        dest.reset(src, start_pointer, dest_count);
    }
}

template <typename Policy, typename Data, typename Alloc>
void homogen_table_impl::pull_rowmajor_impl(const Policy& policy,
                                            array<Data>& block,
                                            const range& rows,
                                            const range& cols,
                                            const Alloc& kind) const {
    // TODO: check range correctness
    // TODO: check array size if non-zero

    const int64_t range_column_count = cols.get_element_count(col_count_);
    const int64_t range_row_count = rows.get_element_count(row_count_);
    const int64_t range_size = range_row_count * range_column_count;
    const data_type block_dtype = detail::make_data_type<Data>();
    const auto table_dtype = meta_.get_data_type(0);
    const bool contiguous_block_requested =
        range_column_count == col_count_ || range_row_count == 1;

    if (block_dtype == table_dtype && contiguous_block_requested == true &&
        has_array_data_kind(policy, data_, kind)) {
        refer_source_data(data_,
                          (rows.start_idx * col_count_ + cols.start_idx) * sizeof(Data),
                          range_size,
                          block);
    }
    else {
        if (block.get_count() < range_size || block.has_mutable_data() == false ||
            has_array_data_kind(policy, block, kind) == false) {
            reset_array(policy, block, range_size, kind);
        }

        const auto type_size = detail::get_data_type_size(table_dtype);

        auto src_pointer =
            data_.get_data() + (rows.start_idx * col_count_ + cols.start_idx) * type_size;
        auto dst_pointer = block.get_mutable_data();

        // TODO: futher optimizations possible: choose optimal column count to switch
        // between strided and non-stirded convert, perform block convertions in parallel
        if (range_column_count > 1) {
            const int64_t blocks_count = contiguous_block_requested ? 1 : range_row_count;
            const int64_t block_size = contiguous_block_requested ? range_size : range_column_count;

            for (int64_t block_idx = 0; block_idx < blocks_count; block_idx++) {
                backend::convert_vector(policy,
                                        src_pointer + block_idx * col_count_ * type_size,
                                        dst_pointer + block_idx * range_column_count,
                                        table_dtype,
                                        block_dtype,
                                        block_size);
            }
        }
        else {
            backend::convert_vector(policy,
                                    src_pointer,
                                    dst_pointer,
                                    table_dtype,
                                    block_dtype,
                                    type_size * col_count_,
                                    sizeof(Data),
                                    range_size);
        }
    }
}

template <typename Policy, typename Data>
void homogen_table_impl::push_rowmajor_impl(const Policy& policy,
                                            const array<Data>& block,
                                            const range& rows,
                                            const range& cols) {
    // TODO: check range correctness
    // TODO: check array size if non-zero

    const int64_t range_row_count = rows.get_element_count(row_count_);
    const int64_t range_column_count = cols.get_element_count(col_count_);
    const int64_t range_size = range_row_count * range_column_count;
    const data_type block_dtype = detail::make_data_type<Data>();
    const auto table_dtype = meta_.get_data_type(0);
    const bool contiguous_block_requested =
        range_column_count == col_count_ || range_row_count == 1;

    make_mutable_data(policy, data_);

    if (block_dtype == table_dtype && contiguous_block_requested == true) {
        auto row_data = reinterpret_cast<Data*>(data_.get_mutable_data());
        auto row_start_pointer = row_data + rows.start_idx * col_count_ + cols.start_idx;

        if (row_start_pointer == block.get_data()) {
            return;
        }
        else {
            detail::memcpy(policy, row_start_pointer, block.get_data(), range_size * sizeof(Data));
        }
    }
    else {
        const auto type_size = detail::get_data_type_size(table_dtype);

        auto src_pointer = block.get_data();
        auto dst_pointer =
            data_.get_mutable_data() + (rows.start_idx * col_count_ + cols.start_idx) * type_size;

        // TODO: futher optimizations possible: choose optimal column count to switch
        // between strided and non-stirded convert, perform block convertions in parallel
        if (range_column_count > 1) {
            const int64_t blocks_count = contiguous_block_requested ? 1 : range_row_count;
            const int64_t block_size = contiguous_block_requested ? range_size : range_column_count;

            for (int64_t block_idx = 0; block_idx < blocks_count; block_idx++) {
                backend::convert_vector(policy,
                                        src_pointer + block_idx * range_column_count,
                                        dst_pointer + block_idx * col_count_ * type_size,
                                        block_dtype,
                                        table_dtype,
                                        block_size);
            }
        }
        else {
            backend::convert_vector(policy,
                                    src_pointer,
                                    dst_pointer,
                                    block_dtype,
                                    table_dtype,
                                    sizeof(Data),
                                    type_size * col_count_,
                                    range_size);
        }
    }
}

#define INSTANTIATE_IMPL(Policy, Data, Alloc)                                 \
    template void homogen_table_impl::pull_rowmajor_impl(const Policy&,       \
                                                         array<Data>&,        \
                                                         const range&,        \
                                                         const range&,        \
                                                         const Alloc&) const; \
    template void homogen_table_impl::push_rowmajor_impl(const Policy&,       \
                                                         const array<Data>&,  \
                                                         const range&,        \
                                                         const range&);

#ifdef ONEAPI_DAL_DATA_PARALLEL
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
