/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <utility>
#include <numeric>
#include <algorithm>

#include "oneapi/dal/array.hpp"

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/table/backend/convert/common.hpp"

namespace oneapi::dal::backend {

template <typename Iter1, typename Iter2>
struct pair_compare {
    using val1_t = typename std::iterator_traits<Iter1>::value_type;
    using val2_t = typename std::iterator_traits<Iter2>::value_type;

    pair_compare(const pair_compare&) = default;
    pair_compare& operator=(const pair_compare&) = default;
    pair_compare(Iter1 it1, Iter2 it2) : iter1{ it1 }, iter2{ it2 } {}

    template <typename Index>
    bool operator()(Index l_idx, Index r_idx) const {
        const val1_t l1 = *std::next(iter1, l_idx);
        const val2_t l2 = *std::next(iter2, l_idx);
        const val1_t r1 = *std::next(iter1, r_idx);
        const val2_t r2 = *std::next(iter2, r_idx);
        return (l1 == r1) ? (l2 < r2) : (l1 < r1);
    }

private:
    Iter1 iter1;
    Iter2 iter2;
};

template <typename Iter1, typename Iter2>
inline auto make_pair_compare(Iter1 iter1, Iter2 iter2) {
    return pair_compare<Iter1, Iter2>{ iter1, iter2 };
}

std::int64_t count_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                                        const data_type* inp,
                                        const data_type* out) {
    const auto compare = make_pair_compare(inp, out);

    std::int64_t result = 1l;
    const auto count = indices.get_count();
    const auto* const ids_ptr = indices.get_data();

    for (std::int64_t i = 1l; i < count; ++i) {
        const auto prev = ids_ptr[i - 1l];
        const auto curr = ids_ptr[i];
        result += compare(prev, curr);
    }

    return result;
}

dal::array<std::int64_t> find_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                                                   const data_type* inp,
                                                   const data_type* out) {
    const auto result_count = count_unique_chunk_offsets(indices, inp, out);
    auto result = dal::array<std::int64_t>::empty(result_count);
    const auto compare = make_pair_compare(inp, out);

    const std::int64_t* const ids_ptr = indices.get_data();
    std::int64_t* const res_ptr = result.get_mutable_data();

    std::int64_t offset = 0l;
    const auto count = indices.get_count();
    for (std::int64_t i = 1l; i < count; ++i) {
        const auto prev = ids_ptr[i - 1l];
        const auto curr = ids_ptr[i];

        if (compare(prev, curr)) {
            res_ptr[offset++] = i;
        }
    }

    res_ptr[offset++] = count;

    ONEDAL_ASSERT(offset == result_count);

    return result;
}

dal::array<std::int64_t> find_sets_of_unique_pairs(const data_type* inp,
                                                   const data_type* out,
                                                   std::int64_t count) {
    const auto compare = make_pair_compare(inp, out);
    auto indices = dal::array<std::int64_t>::empty(count);
    std::iota(backend::begin(indices), backend::end(indices), 0l);
    std::sort(backend::begin(indices), backend::end(indices), compare);
    return indices;
}

dal::array<std::int64_t> find_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                                                   const dal::array<data_type>& inp,
                                                   const dal::array<data_type>& out) {
    const auto* const inp_ptr = inp.get_data();
    const auto* const out_ptr = out.get_data();
    ONEDAL_ASSERT(indices.get_count() == inp.get_count());
    ONEDAL_ASSERT(indices.get_count() == out.get_count());
    return find_unique_chunk_offsets(indices, inp_ptr, out_ptr);
}

dal::array<std::int64_t> find_sets_of_unique_pairs(const dal::array<data_type>& inp,
                                                   const dal::array<data_type>& out) {
    const auto count = inp.get_count();
    ONEDAL_ASSERT(count == out.get_count());
    const auto* const inp_ptr = inp.get_data();
    const auto* const out_ptr = out.get_data();
    return find_sets_of_unique_pairs(inp_ptr, out_ptr, count);
}

bool is_known_data_type(data_type dtype) noexcept {
    const auto op = [](auto type) {
        return true;
    };
    const auto unknown = [](data_type dt) {
        return false;
    };
    return backend::dispatch_by_data_type(dtype, op, unknown);
}

dal::array<std::int64_t> compute_lower_bounds(const shape_t& input_shape,
                                              const dal::array<data_type>& input_types) {
    return compute_lower_bounds(input_shape, input_types.get_data());
}

dal::array<std::int64_t> compute_lower_bounds(const shape_t& input_shape,
                                              const data_type* input_types) {
    ONEDAL_ASSERT(input_types != nullptr);
    const auto [row_count, col_count] = input_shape;
    ONEDAL_ASSERT((0l < row_count) && (0l < col_count));
    auto result = dal::array<std::int64_t>::empty(row_count);
    std::int64_t* const result_ptr = result.get_mutable_data();

    std::int64_t offset = 0l;
    for (std::int64_t row = 0l; row < row_count; ++row) {
        const data_type dtype = input_types[row];
        ONEDAL_ASSERT(is_known_data_type(dtype));

        auto raw_size = detail::get_data_type_size(dtype);
        auto row_size = detail::check_mul_overflow(raw_size, col_count);

        offset = detail::check_sum_overflow(offset, row_size);
        result_ptr[row] = offset;
    }

    return result;
}

} // namespace oneapi::dal::backend
