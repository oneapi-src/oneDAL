/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

namespace oneapi::dal::backend {

using error_msg = dal::detail::error_messages;

table_metadata create_homogen_metadata(std::int64_t feature_count, data_type dtype) {
    auto default_ftype =
        detail::is_floating_point(dtype) ? feature_type::ratio : feature_type::ordinal;

    auto dtypes = array<data_type>::full(feature_count, dtype);
    auto ftypes = array<feature_type>::full(feature_count, default_ftype);
    return table_metadata{ dtypes, ftypes };
}

table_metadata create_homogen_metadata(std::int64_t feature_count,
                                       data_type dtype,
                                       const array<feature_type>& ftypes) {
    auto dtypes = array<data_type>::full(feature_count, dtype);
    if (ftypes.get_count() != 0) {
        ONEDAL_ASSERT(dtypes.get_count() == ftypes.get_count());
        return table_metadata{ dtypes, ftypes };
    }

    auto default_ftype =
        detail::is_floating_point(dtype) ? feature_type::ratio : feature_type::ordinal;
    auto new_ftypes = array<feature_type>::full(feature_count, default_ftype);
    return table_metadata{ dtypes, new_ftypes };
}

template <typename Policy, typename Data>
static void make_mutable_data(const Policy& policy, array<Data>& array) {
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
static void reset_array(const Policy& policy,
                        array<Data>& array,
                        std::int64_t count,
                        const Alloc& kind) {
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
static bool has_array_data_kind(const Policy& policy, const array<Data>& array, const Alloc& kind) {
    if (array.get_count() <= 0) {
        return false;
    }

    if constexpr (std::is_same_v<Policy, detail::default_host_policy>) {
        // We assume that no sycl::usm::alloc::device pointers used with host policies.
        // It is responsibility of user to pass right pointers because we cannot check
        // the right pointer type with the host policy.
        static_assert(std::is_same_v<Alloc, homogen_table_impl::host_alloc_t>);
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

template <typename DataSrc, typename DataDest>
static void refer_source_data(const array<DataSrc>& src,
                              std::int64_t src_start_index,
                              std::int64_t dst_count,
                              array<DataDest>& dst) {
    ONEDAL_ASSERT(src_start_index >= 0);
    ONEDAL_ASSERT(src.get_count() > src_start_index);
    ONEDAL_ASSERT((src.get_count() - src_start_index) * sizeof(DataSrc) >=
                  dst_count * sizeof(DataDest));

    if (src.has_mutable_data()) {
        // TODO: in future, when table knows about mutability of its data this branch shall be
        // available only for builders, not for tables.
        auto start_pointer = reinterpret_cast<DataDest*>(src.get_mutable_data() + src_start_index);
        dst.reset(src, start_pointer, dst_count);
    }
    else {
        auto start_pointer = reinterpret_cast<const DataDest*>(src.get_data() + src_start_index);
        dst.reset(src, start_pointer, dst_count);
    }
}

struct block_info {
    block_info(std::int64_t row_count,
               std::int64_t column_count,
               std::int64_t row_offset,
               std::int64_t column_offset)
            : row_count(row_count),
              column_count(column_count),
              row_offset(row_offset),
              column_offset(column_offset),
              element_count(row_count * column_count) {
        detail::check_mul_overflow(row_count, column_count);

        ONEDAL_ASSERT(row_count > 0);
        ONEDAL_ASSERT(column_count > 0);
        ONEDAL_ASSERT(row_offset >= 0);
        ONEDAL_ASSERT(column_offset >= 0);
    }

    std::int64_t row_count;
    std::int64_t column_count;
    std::int64_t row_offset;
    std::int64_t column_offset;
    std::int64_t element_count;
};

struct origin_info {
    origin_info(data_type dtype, std::int64_t row_count, std::int64_t column_count)
            : dtype(dtype),
              row_count(row_count),
              column_count(column_count),
              element_count(row_count * column_count) {
        // row_count * column_count overflow checked in homogen_table_impl

        ONEDAL_ASSERT(row_count > 0);
        ONEDAL_ASSERT(column_count > 0);
    }

    data_type dtype;
    std::int64_t row_count;
    std::int64_t column_count;
    std::int64_t element_count;
};

class block_access_provider {
public:
    block_access_provider(std::int64_t origin_row_count,
                          std::int64_t origin_column_count,
                          data_type origin_data_type,
                          const range& block_row_range,
                          const range& block_column_range)
            : block_(block_row_range.get_element_count(origin_row_count),
                     block_column_range.get_element_count(origin_column_count),
                     block_row_range.start_idx,
                     block_column_range.start_idx),
              origin_(origin_data_type, origin_row_count, origin_column_count) {
        ONEDAL_ASSERT(block_.row_count + block_.row_offset <= origin_.row_count);
        ONEDAL_ASSERT(block_.column_count + block_.column_offset <= origin_.column_count);
        // thowable checks done in homogen_table_impl, including overflows
    }

    template <typename Policy, typename BlockData, typename Alloc>
    void pull_by_row_major(const Policy& policy,
                           const array<byte_t>& origin_data,
                           array<BlockData>& block_data,
                           const Alloc& kind) const {
        constexpr std::int64_t block_dtype_size = sizeof(BlockData);
        const auto origin_dtype_size = detail::get_data_type_size(origin_.dtype);

        // overflows checked here
        check_origin_data(origin_data, origin_dtype_size, block_dtype_size);

        const auto block_dtype = detail::make_data_type<BlockData>();

        const std::int64_t origin_offset =
            (block_.row_offset * origin_.column_count + block_.column_offset);
        // operation is safe because block offsets do not exceed origin element count

        const bool contiguous_block_requested =
            block_.column_count == origin_.column_count || block_.row_count == 1;

        if (block_dtype == origin_.dtype && contiguous_block_requested == true &&
            has_array_data_kind(policy, origin_data, kind)) {
            refer_source_data(origin_data,
                              origin_offset * block_dtype_size,
                              block_.element_count,
                              block_data);
        }
        else {
            if (block_data.get_count() < block_.element_count ||
                block_data.has_mutable_data() == false ||
                has_array_data_kind(policy, block_data, kind) == false) {
                reset_array(policy, block_data, block_.element_count, kind);
            }

            auto src_data = origin_data.get_data() + origin_offset * origin_dtype_size;
            auto dst_data = block_data.get_mutable_data();

            if (block_.column_count > 1) {
                const std::int64_t subblocks_count =
                    contiguous_block_requested ? 1 : block_.row_count;
                const std::int64_t subblock_size =
                    contiguous_block_requested ? block_.element_count : block_.column_count;

                for (std::int64_t subblock_idx = 0; subblock_idx < subblocks_count;
                     subblock_idx++) {
                    backend::convert_vector(
                        policy,
                        src_data + subblock_idx * origin_.column_count * origin_dtype_size,
                        dst_data + subblock_idx * block_.column_count,
                        origin_.dtype,
                        block_dtype,
                        subblock_size);
                }
            }
            else {
                backend::convert_vector(policy,
                                        src_data,
                                        dst_data,
                                        origin_.dtype,
                                        block_dtype,
                                        origin_dtype_size * origin_.column_count,
                                        block_dtype_size,
                                        block_.element_count);
            }
        }
    }

    template <typename Policy, typename BlockData, typename Alloc>
    void pull_by_column_major(const Policy& policy,
                              const array<byte_t>& origin_data,
                              array<BlockData>& block_data,
                              const Alloc& kind) const {
        constexpr std::int64_t block_dtype_size = sizeof(BlockData);
        const auto origin_dtype_size = detail::get_data_type_size(origin_.dtype);

        // overflows checked here
        check_origin_data(origin_data, origin_dtype_size, block_dtype_size);

        const auto block_dtype = detail::make_data_type<BlockData>();
        const std::int64_t origin_offset =
            block_.row_offset + block_.column_offset * origin_.row_count;
        // operation is safe because block offsets do not exceed origin element count

        if (block_data.get_count() < block_.element_count ||
            block_data.has_mutable_data() == false ||
            has_array_data_kind(policy, block_data, kind) == false) {
            reset_array(policy, block_data, block_.element_count, kind);
        }

        auto src_data = origin_data.get_data() + origin_offset * origin_dtype_size;
        auto dst_data = block_data.get_mutable_data();

        for (std::int64_t row_idx = 0; row_idx < block_.row_count; row_idx++) {
            backend::convert_vector(policy,
                                    src_data + row_idx * origin_dtype_size,
                                    dst_data + row_idx * block_.column_count,
                                    origin_.dtype,
                                    block_dtype,
                                    origin_dtype_size * origin_.row_count,
                                    block_dtype_size,
                                    block_.column_count);
        }
    }

    template <typename Policy, typename BlockData>
    void push_by_row_major(const Policy& policy,
                           array<byte_t>& origin_data,
                           const array<BlockData>& block_data) const {
        constexpr std::int64_t block_dtype_size = sizeof(BlockData);
        const auto origin_dtype_size = detail::get_data_type_size(origin_.dtype);

        // overflows checked here
        check_origin_data(origin_data, origin_dtype_size, block_dtype_size);
        if (block_data.get_count() != block_.element_count) {
            throw dal::range_error(error_msg::small_data_block());
        }

        make_mutable_data(policy, origin_data);

        const auto block_dtype = detail::make_data_type<BlockData>();
        const std::int64_t origin_offset =
            block_.row_offset * origin_.column_count + block_.column_offset;
        // operation is safe because block offsets do not exceed origin element count

        const bool contiguous_block_requested =
            block_.column_count == origin_.column_count || block_.row_count == 1;

        if (origin_.dtype == block_dtype && contiguous_block_requested == true) {
            auto row_data = reinterpret_cast<BlockData*>(origin_data.get_mutable_data());
            auto row_start_pointer = row_data + origin_offset;

            if (row_start_pointer == block_data.get_data()) {
                return;
            }
            else {
                detail::memcpy(policy,
                               row_start_pointer,
                               block_data.get_data(),
                               block_.element_count * block_dtype_size);
            }
        }
        else {
            auto src_data = block_data.get_data();
            auto dst_data = origin_data.get_mutable_data() + origin_offset * origin_dtype_size;

            if (block_.column_count > 1) {
                const std::int64_t blocks_count = contiguous_block_requested ? 1 : block_.row_count;
                const std::int64_t block_size =
                    contiguous_block_requested ? block_.element_count : block_.column_count;

                for (std::int64_t block_idx = 0; block_idx < blocks_count; block_idx++) {
                    backend::convert_vector(
                        policy,
                        src_data + block_idx * block_.column_count,
                        dst_data + block_idx * origin_.column_count * origin_dtype_size,
                        block_dtype,
                        origin_.dtype,
                        block_size);
                }
            }
            else {
                backend::convert_vector(policy,
                                        src_data,
                                        dst_data,
                                        block_dtype,
                                        origin_.dtype,
                                        block_dtype_size,
                                        origin_dtype_size * origin_.column_count,
                                        block_.element_count);
            }
        }
    }

    template <typename Policy, typename BlockData>
    void push_by_column_major(const Policy& policy,
                              array<byte_t>& origin_data,
                              const array<BlockData>& block_data) const {
        constexpr std::int64_t block_dtype_size = sizeof(BlockData);
        const auto origin_dtype_size = detail::get_data_type_size(origin_.dtype);

        // overflows checked here
        check_origin_data(origin_data, origin_dtype_size, block_dtype_size);

        if (block_data.get_count() != block_.element_count) {
            throw dal::range_error(error_msg::small_data_block());
        }

        make_mutable_data(policy, origin_data);
        const auto block_dtype = detail::make_data_type<BlockData>();
        const std::int64_t origin_offset =
            block_.row_offset + block_.column_offset * origin_.row_count;
        // operation is safe because block offsets do not exceed origin element count

        auto src_data = block_data.get_data();

        detail::check_mul_overflow(origin_.element_count, origin_dtype_size);
        auto dst_data = origin_data.get_mutable_data() + origin_offset * origin_dtype_size;

        for (std::int64_t row_idx = 0; row_idx < block_.row_count; row_idx++) {
            backend::convert_vector(policy,
                                    src_data + row_idx * block_.column_count,
                                    dst_data + row_idx * origin_dtype_size,
                                    block_dtype,
                                    origin_.dtype,
                                    block_dtype_size,
                                    origin_dtype_size * origin_.row_count,
                                    block_.column_count);
        }
    }

private:
    void check_origin_data(const array<byte_t>& origin_data,
                           std::int64_t origin_dtype_size,
                           std::int64_t block_dtype_size) const {
        detail::check_mul_overflow(origin_.element_count,
                                   std::max(origin_dtype_size, block_dtype_size));
        ONEDAL_ASSERT(origin_data.get_count() >= origin_.element_count * origin_dtype_size);
    }

private:
    block_info block_;
    origin_info origin_;
};

static void check_block_row_range(const range& rows, std::int64_t origin_row_count) {
    const std::int64_t range_row_count = rows.get_element_count(origin_row_count);
    detail::check_sum_overflow(rows.start_idx, range_row_count);
    if (rows.start_idx + range_row_count > origin_row_count) {
        throw dal::range_error(error_msg::invalid_range_of_rows());
    }
}

static void check_block_column_index(std::int64_t column_index, std::int64_t origin_col_count) {
    if (column_index >= origin_col_count) {
        throw dal::range_error(error_msg::column_index_out_of_range());
    }
}

template <typename Policy, typename Data, typename Alloc>
void homogen_table_impl::pull_rows_impl(const Policy& policy,
                                        array<Data>& block,
                                        const range& rows,
                                        const Alloc& kind) const {
    check_block_row_range(rows, row_count_);

    const auto& data_type = meta_.get_data_type(0);
    const range cols{ 0, -1 };
    const block_access_provider p{ row_count_, col_count_, data_type, rows, cols };

    switch (layout_) {
        case data_layout::row_major: p.pull_by_row_major(policy, data_, block, kind); break;
        case data_layout::column_major: p.pull_by_column_major(policy, data_, block, kind); break;
        default: throw dal::domain_error(error_msg::unsupported_data_layout());
    }
}

template <typename Policy, typename Data, typename Alloc>
void homogen_table_impl::pull_column_impl(const Policy& policy,
                                          array<Data>& block,
                                          std::int64_t column_index,
                                          const range& rows,
                                          const Alloc& kind) const {
    check_block_row_range(rows, row_count_);
    check_block_column_index(column_index, col_count_);

    const auto& data_type = meta_.get_data_type(0);
    const range column{ column_index, column_index + 1 };
    const block_access_provider p{ col_count_, row_count_, data_type, column, rows };

    switch (layout_) {
        case data_layout::row_major: p.pull_by_column_major(policy, data_, block, kind); break;
        case data_layout::column_major: p.pull_by_row_major(policy, data_, block, kind); break;
        default: throw dal::domain_error(error_msg::unsupported_data_layout());
    }
}

template <typename Policy, typename Data>
void homogen_table_impl::push_rows_impl(const Policy& policy,
                                        const array<Data>& block,
                                        const range& rows) {
    check_block_row_range(rows, row_count_);

    const auto& data_type = meta_.get_data_type(0);
    const range cols{ 0, -1 };
    const block_access_provider p{ row_count_, col_count_, data_type, rows, cols };

    switch (layout_) {
        case data_layout::row_major: p.push_by_row_major(policy, data_, block); break;
        case data_layout::column_major: p.push_by_column_major(policy, data_, block); break;
        default: throw dal::domain_error(error_msg::unsupported_data_layout());
    }
}

template <typename Policy, typename Data>
void homogen_table_impl::push_column_impl(const Policy& policy,
                                          const array<Data>& block,
                                          std::int64_t column_index,
                                          const range& rows) {
    check_block_row_range(rows, row_count_);
    check_block_column_index(column_index, col_count_);

    const auto& data_type = meta_.get_data_type(0);
    const range column{ column_index, column_index + 1 };
    const block_access_provider p{ col_count_, row_count_, data_type, column, rows };

    switch (layout_) {
        case data_layout::row_major: p.push_by_column_major(policy, data_, block); break;
        case data_layout::column_major: p.push_by_row_major(policy, data_, block); break;
        default: throw dal::domain_error(error_msg::unsupported_data_layout());
    }
}

#define INSTANTIATE_IMPL(Policy, Data, Alloc)                                     \
    template void homogen_table_impl::pull_rows_impl(const Policy& policy,        \
                                                     array<Data>& block,          \
                                                     const range& rows,           \
                                                     const Alloc& kind) const;    \
    template void homogen_table_impl::pull_column_impl(const Policy& policy,      \
                                                       array<Data>& block,        \
                                                       std::int64_t column_index, \
                                                       const range& rows,         \
                                                       const Alloc& kind) const;  \
    template void homogen_table_impl::push_rows_impl(const Policy& policy,        \
                                                     const array<Data>& block,    \
                                                     const range& rows);          \
    template void homogen_table_impl::push_column_impl(const Policy& policy,      \
                                                       const array<Data>& block,  \
                                                       std::int64_t column_index, \
                                                       const range& rows);

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
