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

#include <numeric>
#include <algorithm>

#include "oneapi/dal/array.hpp"

#include "oneapi/dal/backend/common.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/backend/primitives/convert/copy_convert.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Type>
struct preferred_vector_desc {
    using type = sycl::info::device::preferred_vector_width_float;
};

template <>
struct preferred_vector_desc<char> {
    using type = sycl::info::device::preferred_vector_width_char;
};

template <>
struct preferred_vector_desc<short> {
    using type = sycl::info::device::preferred_vector_width_short;
};

template <>
struct preferred_vector_desc<int> {
    using type = sycl::info::device::preferred_vector_width_int;
};

template <>
struct preferred_vector_desc<long> {
    using type = sycl::info::device::preferred_vector_width_long;
};

template <>
struct preferred_vector_desc<float> {
    using type = sycl::info::device::preferred_vector_width_float;
};

template <>
struct preferred_vector_desc<double> {
    using type = sycl::info::device::preferred_vector_width_double;
};

template <typename Type, bool is_integral>
struct make_signed_map {
    using type = Type;
};

template <typename Type>
struct make_signed_map<Type, true> {
    using type = std::make_signed_t<Type>;
    static_assert(std::is_integral_v<Type>);
};

template <typename T, bool is_integral = std::is_integral_v<T>>
using make_signed_t = typename make_signed_map<T, is_integral>::type;

template <typename Type, typename T = make_signed_t<Type>>
using preferred_vector_desc_t = typename preferred_vector_desc<T>::type;

template <typename Left, typename Right, bool use_left = sizeof(Right) < sizeof(Left)>
using longer_preferred_vector_desc_t = std::conditional_t<use_left,
                        preferred_vector_desc_t<Left>, preferred_vector_desc_t<Right>>;

template <typename InpType, typename OutType>
auto propose_range(const sycl::queue& queue, const shape_t& shape) {
    using prop_t = longer_preferred_vector_desc_t<InpType, OutType>;

    const auto [row_count, col_count] = shape;
    const sycl::device device = queue.get_device();
    const std::size_t vec = device.template get_info<prop_t>();
    const auto pref_vec = detail::integral_cast<std::int64_t>(vec);
    const std::int64_t count = std::max(pref_vec, col_count);
    return std::make_pair(row_count, count);
}

template <typename InpType, typename OutType>
sycl::event copy_convert_impl(sycl::queue& queue,
                              const InpType* const* inp_pointers,
                              const std::int64_t* inp_strides,
                              OutType* const* out_pointers,
                              const std::int64_t* out_strides,
                              const shape_t& shape,
                              const std::vector<sycl::event>& deps) {
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        const auto range = propose_range<InpType, OutType>(queue, shape);
        const auto range_2d = make_range_2d(range.first, range.second);

        const std::int64_t col_count = shape.second;
        const std::int64_t wi_per_row = range.second;
        h.parallel_for(range_2d,
        [=](sycl::id<2> idx) -> void {
            const std::int64_t row = idx[0];

            const auto out_str = out_strides[row];
            const auto inp_str = inp_strides[row];

            OutType* const out_ptr = out_pointers[row];
            const InpType* const inp_ptr = inp_pointers[row];

            for (std::int64_t col = idx[1]; col < col_count; col += wi_per_row) {
                const std::int64_t out_offset = col * out_str;
                const std::int64_t inp_offset = col * inp_str;
                out_ptr[out_offset] = static_cast<OutType>(inp_ptr[inp_offset]);
            }
        });
    });
}

template <typename Iter1, typename Iter2>
struct pair_compare {
    using val1_t = typename std::iterator_traits<Iter1>::value_type;
    using val2_t = typename std::iterator_traits<Iter2>::value_type;

    pair_compare(const pair_compare&) = default;
    pair_compare& operator=(const pair_compare&) = default;
    pair_compare(Iter1 it1, Iter2 it2) : iter1{ it1 }, iter2{ it2 } {}

    template <typename Index>
    bool operator() (Index l_idx, Index r_idx) const {
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

auto count_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                                const data_type* inp, const data_type* out) {
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

auto find_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
                               const data_type* inp, const data_type* out) {
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

auto find_sets_of_unique_pairs(const data_type* inp, const data_type* out, std::int64_t count) {
    const auto compare = make_pair_compare(inp, out);
    auto indices = dal::array<std::int64_t>::empty(count);
    std::iota(backend::begin(indices), backend::end(indices), 0l);
    std::sort(backend::begin(indices), backend::end(indices), compare);
    return indices;
}

auto find_unique_chunk_offsets(const dal::array<std::int64_t>& indices, //
        const dal::array<data_type>& inp, const dal::array<data_type>& out) {
    const auto* const inp_ptr = inp.get_data();
    const auto* const out_ptr = out.get_data();
    ONEDAL_ASSERT(indices.get_count() == inp.get_count());
    ONEDAL_ASSERT(indices.get_count() == out.get_count());
    return find_unique_chunk_offsets(indices, inp_ptr, out_ptr);
}

auto find_sets_of_unique_pairs(const dal::array<data_type>& inp, const dal::array<data_type>& out) {
    const auto count = inp.get_count();
    ONEDAL_ASSERT(count == out.get_count());
    const auto* const inp_ptr = inp.get_data();
    const auto* const out_ptr = out.get_data();
    return find_sets_of_unique_pairs(inp_ptr, out_ptr, count);
}

sycl::event copy_convert(sycl::queue& queue,
                         const dal::byte_t* const* inp_pointers,
                         data_type inp_type,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         data_type out_type,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps) {
    backend::multi_dispatch_by_data_type(
    [&](auto inp, auto out) -> void {
        using out_t = std::decay_t<decltype(out)>;
        using inp_t = std::decay_t<decltype(inp)>;

        auto* const adj_out_ptrs = reinterpret_cast<out_t* const *>(out_pointers);
        auto* const adj_inp_ptrs = reinterpret_cast<const inp_t* const *>(inp_pointers);

        auto res_event = copy_convert_impl<inp_t, out_t>(queue, adj_inp_ptrs, inp_strides,
                                          adj_out_ptrs, out_strides, shape, deps);

        res_event.wait_and_throw();
    }, inp_type, out_type);

    return sycl::event{};
}

} // namespace oneapi::dal::backend::primitives