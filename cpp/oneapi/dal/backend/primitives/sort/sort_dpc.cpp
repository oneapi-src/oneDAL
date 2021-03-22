/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/primitives/sort/sort.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include <CL/sycl/ONEAPI/experimental/builtins.hpp>

namespace oneapi::dal::backend::primitives {

namespace de = dal::detail;

template <typename Float>
struct float2uint_map;

template <>
struct float2uint_map<float> {
    using type_t = std::uint32_t;
};

template <>
struct float2uint_map<double> {
    using type_t = std::uint64_t;
};

inline std::uint32_t inv_bits(std::uint32_t x) {
    return x ^ (-(x >> 31) | 0x80000000u);
}

inline std::uint64_t inv_bits(std::uint64_t x) {
    return x ^ (-(x >> 63) | 0x8000000000000000ul);
}

constexpr std::uint32_t radix_bits = 4;
constexpr std::uint32_t radix_range = (std::uint32_t)1 << radix_bits;
constexpr std::uint32_t radix_range_1 = radix_range - 1;

template <typename Float, typename RadixInteger, typename IndexType>
static sycl::event radix_scan(sycl::queue& queue,
                              const ndview<Float, 1>& val,
                              ndarray<IndexType, 1>& part_hist,
                              IndexType elem_count,
                              std::uint32_t bit_offset,
                              std::int64_t local_size,
                              std::int64_t local_hist_count) {
    ONEDAL_ASSERT(part_hist.get_count() == ((local_hist_count + 1) << radix_bits));

    sycl::range<1> global(local_size * local_hist_count);
    sycl::range<1> local(local_size);
    sycl::nd_range<1> nd_range(global, local);

    const RadixInteger* val_ptr =
        static_cast<const RadixInteger*>(static_cast<const void*>(val.get_data()));
    IndexType* part_hist_ptr = part_hist.get_mutable_data();

    auto event = queue.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id()[0] > 0) {
                return;
            }
            const std::uint32_t n_groups = item.get_group_range(0);
            const std::uint32_t n_sub_groups = sbg.get_group_range()[0];
            const std::uint32_t n_total_sub_groups = n_sub_groups * n_groups;
            const IndexType elems_for_sbg =
                elem_count / n_total_sub_groups + bool(elem_count % n_total_sub_groups);
            const std::uint32_t local_size = item.get_sub_group().get_local_range()[0];

            const std::uint32_t local_id = item.get_sub_group().get_local_id()[0];
            const std::uint32_t sub_group_id = sbg.get_group_id();
            const std::uint32_t group_id = item.get_group().get_id(0) * n_sub_groups + sub_group_id;

            IndexType ind_start = group_id * elems_for_sbg;
            IndexType ind_end = (group_id + 1) * elems_for_sbg;

            if (ind_end > elem_count) {
                ind_end = elem_count;
            }

            IndexType offset[radix_range];
            for (std::uint32_t i = 0; i < radix_range; i++) {
                offset[i] = 0;
            }

            for (IndexType i = ind_start + local_id; i < ind_end; i += local_size) {
                RadixInteger data_bits = ((inv_bits(val_ptr[i]) >> bit_offset) & radix_range_1);
                for (std::uint32_t j = 0; j < radix_range; j++) {
                    IndexType value = static_cast<IndexType>(data_bits == j);
                    IndexType partial_offset =
                        sycl::ONEAPI::reduce(sbg, value, cl::sycl::ONEAPI::plus<IndexType>());
                    offset[j] += partial_offset;
                }
            }

            if (local_id == 0) {
                for (std::uint32_t j = 0; j < radix_range; j++) {
                    part_hist_ptr[group_id * radix_range + j] = offset[j];
                }
            }
        });
    });

    return event;
}

template <typename RadixInteger, typename IndexType>
static sycl::event radix_hist_scan(sycl::queue& queue,
                                   const ndarray<IndexType, 1>& part_hist,
                                   ndarray<IndexType, 1>& part_prefix_hist,
                                   std::int64_t local_size,
                                   std::int64_t local_hist_count) {
    ONEDAL_ASSERT(part_hist.get_count() == ((local_hist_count + 1) << radix_bits));
    ONEDAL_ASSERT(part_prefix_hist.get_count() == ((local_hist_count + 1) << radix_bits));

    const IndexType* part_hist_ptr = part_hist.get_data();
    IndexType* part_prefix_hist_ptr = part_prefix_hist.get_mutable_data();

    sycl::range<1> global(local_size);
    sycl::range<1> local(local_size);
    sycl::nd_range<1> nd_range(global, local);

    auto event = queue.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id()[0] > 0) {
                return;
            }

            const std::uint32_t local_size = sbg.get_local_range()[0];
            const std::uint32_t local_id = sbg.get_local_id()[0];

            IndexType offset[radix_range];
            for (std::uint32_t i = 0; i < radix_range; i++) {
                offset[i] = 0;
            }

            for (std::uint32_t i = local_id; i < local_hist_count; i += local_size) {
                for (std::uint32_t j = 0; j < radix_range; j++) {
                    IndexType value = part_hist_ptr[i * radix_range + j];
                    IndexType boundary =
                        sycl::ONEAPI::exclusive_scan(sbg,
                                                     value,
                                                     cl::sycl::ONEAPI::plus<IndexType>());
                    part_prefix_hist_ptr[i * radix_range + j] = offset[j] + boundary;
                    IndexType partial_offset =
                        sycl::ONEAPI::reduce(sbg, value, cl::sycl::ONEAPI::plus<IndexType>());
                    offset[j] += partial_offset;
                }
            }

            if (local_id == 0) {
                IndexType totalSum = 0;
                for (std::uint32_t j = 0; j < radix_range; j++) {
                    part_prefix_hist_ptr[local_hist_count * radix_range + j] = totalSum;
                    totalSum += offset[j];
                }
            }
        });
    });

    return event;
}

template <typename Float, typename RadixInteger, typename IndexType>
static sycl::event radix_reorder(sycl::queue& queue,
                                 const ndview<Float, 1>& val_in,
                                 const ndview<IndexType, 1>& ind_in,
                                 const ndview<IndexType, 1>& part_prefix_hist,
                                 ndview<Float, 1>& val_out,
                                 ndview<IndexType, 1>& ind_out,
                                 IndexType elem_count,
                                 std::uint32_t bit_offset,
                                 std::int64_t local_size,
                                 std::int64_t local_hist_count) {
    ONEDAL_ASSERT(part_hist.get_count() == ((local_hist_count + 1) << radix_bits));
    ONEDAL_ASSERT(val_in.get_count() == ind_in.get_count() == val_out.get_count() ==
                  ind_out.get_count());

    const RadixInteger* val_in_ptr =
        static_cast<const RadixInteger*>(static_cast<const void*>(val_in.get_data()));
    const IndexType* ind_in_ptr = ind_in.get_data();
    const IndexType* part_prefix_hist_ptr = part_prefix_hist.get_data();
    RadixInteger* val_out_ptr =
        static_cast<RadixInteger*>(static_cast<void*>(val_out.get_mutable_data()));
    IndexType* ind_out_ptr = ind_out.get_mutable_data();

    sycl::range<1> global(local_size * local_hist_count);
    sycl::range<1> local(local_size);
    sycl::nd_range<1> nd_range(global, local);

    auto event = queue.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id()[0] > 0) {
                return;
            }

            const std::uint32_t n_groups = item.get_group_range(0);
            const std::uint32_t n_sub_groups = sbg.get_group_range()[0];
            const std::uint32_t n_total_sub_groups = n_sub_groups * n_groups;
            const IndexType elems_for_sbg =
                elem_count / n_total_sub_groups + bool(elem_count % n_total_sub_groups);
            const std::uint32_t local_size = item.get_sub_group().get_local_range()[0];

            const std::uint32_t local_id = item.get_sub_group().get_local_id()[0];
            const std::uint32_t sub_group_id = sbg.get_group_id();
            const std::uint32_t group_id = item.get_group().get_id(0) * n_sub_groups + sub_group_id;

            IndexType ind_start = group_id * elems_for_sbg;
            IndexType ind_end = (group_id + 1) * elems_for_sbg;

            if (ind_end > elem_count) {
                ind_end = elem_count;
            }

            IndexType offset[radix_range];

            for (std::uint32_t i = 0; i < radix_range; i++) {
                offset[i] = part_prefix_hist_ptr[group_id * radix_range + i] +
                            part_prefix_hist_ptr[n_total_sub_groups * radix_range + i];
            }

            for (IndexType i = ind_start + local_id; i < ind_end; i += local_size) {
                RadixInteger data_value = val_in_ptr[i];
                RadixInteger data_bits = ((inv_bits(data_value) >> bit_offset) & radix_range_1);
                IndexType pos_new = 0;
                for (std::uint32_t j = 0; j < radix_range; j++) {
                    IndexType value = static_cast<IndexType>(data_bits == j);
                    IndexType boundary =
                        sycl::ONEAPI::exclusive_scan(sbg,
                                                     value,
                                                     cl::sycl::ONEAPI::plus<IndexType>());
                    pos_new |= value * (offset[j] + boundary);
                    IndexType partial_offset =
                        sycl::ONEAPI::reduce(sbg, value, cl::sycl::ONEAPI::plus<IndexType>());
                    offset[j] = offset[j] + partial_offset;
                }
                val_out_ptr[pos_new] = data_value;
                ind_out_ptr[pos_new] = ind_in_ptr[i];
            }
        });
    });

    return event;
}

template <typename Float, typename IndexType>
sycl::event radix_sort_indices_inplace(sycl::queue& queue,
                                       ndview<Float, 1>& val_in,
                                       ndview<IndexType, 1>& ind_in,
                                       ndview<Float, 1>& val_buff,
                                       ndview<IndexType, 1>& ind_buff,
                                       const event_vector& deps) {
    using radix_uint_t = typename float2uint_map<Float>::type_t;

    ONEDAL_ASSERT(val_in.has_mutable_data());
    ONEDAL_ASSERT(ind_in.has_mutable_data());
    ONEDAL_ASSERT(val_buff.has_mutable_data());
    ONEDAL_ASSERT(ind_buff.has_mutable_data());
    ONEDAL_ASSERT(val_in.get_count() == ind_in.get_count() == val_buff.get_count() ==
                  ind_buff.get_count());

    sycl::event::wait_and_throw(deps);

    const std::uint32_t elem_count = de::integral_cast<std::uint32_t>(val_in.get_count());

    const std::uint32_t byte_range = 8;
    const std::uint32_t max_local_hist_count = 1024;
    const std::uint32_t preferable_sbg_size = 16;

    const std::uint32_t local_size = preferable_sbg_size;
    const std::uint32_t local_hist_count =
        max_local_hist_count * local_size < elem_count
            ? max_local_hist_count
            : (elem_count / local_size) + bool(elem_count % local_size);

    auto part_hist = ndarray<IndexType, 1>::empty(queue,
                                                  { (local_hist_count + 1) << radix_bits },
                                                  sycl::usm::alloc::device);
    auto part_prefix_hist = ndarray<IndexType, 1>::empty(queue,
                                                         { (local_hist_count + 1) << radix_bits },
                                                         sycl::usm::alloc::device);

    std::uint32_t rev = 0;

    for (std::uint32_t bit_offset = 0; bit_offset < byte_range * sizeof(Float);
         bit_offset += radix_bits, rev ^= 1) {
        if (!rev) {
            radix_scan<Float, radix_uint_t, IndexType>(queue,
                                                       val_in,
                                                       part_hist,
                                                       elem_count,
                                                       bit_offset,
                                                       local_size,
                                                       local_hist_count)
                .wait_and_throw();
            radix_hist_scan<IndexType>(queue,
                                       part_hist,
                                       part_prefix_hist,
                                       local_size,
                                       local_hist_count)
                .wait_and_throw();
            radix_reorder<Float, radix_uint_t, IndexType>(queue,
                                                          val_in,
                                                          ind_in,
                                                          part_prefix_hist,
                                                          val_buff,
                                                          ind_buff,
                                                          elem_count,
                                                          bit_offset,
                                                          local_size,
                                                          local_hist_count)
                .wait_and_throw();
        }
        else {
            radix_scan<Float, radix_uint_t, IndexType>(queue,
                                                       val_buff,
                                                       part_hist,
                                                       elem_count,
                                                       bit_offset,
                                                       local_size,
                                                       local_hist_count)
                .wait_and_throw();
            radix_hist_scan<IndexType>(queue,
                                       part_hist,
                                       part_prefix_hist,
                                       local_size,
                                       local_hist_count)
                .wait_and_throw();
            radix_reorder<Float, radix_uint_t, IndexType>(queue,
                                                          val_buff,
                                                          ind_buff,
                                                          part_prefix_hist,
                                                          val_in,
                                                          ind_in,
                                                          elem_count,
                                                          bit_offset,
                                                          local_size,
                                                          local_hist_count)
                .wait_and_throw();
        }
    }

    ONEDAL_ASSERT(rev == 0); // if not, we need to swap values/indices and
        // valuesOut/indices_bufus);
    return sycl::event();
}

template <typename Integer>
sycl::event radix_sort(sycl::queue& queue,
                       ndview<Integer, 2>& val_in,
                       ndview<Integer, 2>& val_out,
                       ndview<Integer, 2>& buffer,
                       std::uint32_t sorted_elem_count,
                       const event_vector& deps) {
    constexpr std::uint32_t preferable_wg_size = 32;
    constexpr std::uint32_t expected_buffer_size_for_one_vector = 256;
    // radixBuf should be big enough to accumulate radix_range elements
    constexpr std::uint32_t radix_range = expected_buffer_size_for_one_vector;
    constexpr std::uint32_t radix_count = sizeof(Integer);

    ONEDAL_ASSERT(val_in.get_dimension(0) == val_out.get_dimension(0));
    ONEDAL_ASSERT(val_out.get_dimension(0) == buffer.get_dimension(0));
    ONEDAL_ASSERT(val_in.get_dimension(1) == val_out.get_dimension(1));
    ONEDAL_ASSERT(buffer.get_dimension(1) == expected_buffer_size_for_one_vector);
    ONEDAL_ASSERT(sorted_elem_count > 0);

    Integer* labels = val_in.get_mutable_data();
    Integer* sorted = val_out.get_mutable_data();
    Integer* radixbuf = buffer.get_mutable_data();

    const std::uint32_t vector_count = de::integral_cast<std::uint32_t>(val_in.get_dimension(0));
    const std::uint32_t vector_offset = de::integral_cast<std::uint32_t>(val_in.get_dimension(1));

    sycl::range<2> global(vector_count, preferable_wg_size);
    sycl::range<2> local(1, preferable_wg_size);

    sycl::nd_range<2> nd_range2d(global, local);

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range2d, [=](sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            // Code is written for a single subgroup. It's necessary to adjust the local range if idle subgoups are presented
            if (sbg.get_group_id()[0] > 0) {
                return;
            }

            const std::uint32_t global_id = item.get_global_id()[0];
            const std::uint32_t local_id = item.get_local_id()[1];

            const std::uint32_t local_size = sbg.get_local_range()[0];
            const std::uint32_t group_aligned_size =
                sorted_elem_count - sorted_elem_count % local_size;
            const std::uint32_t rem = sorted_elem_count - group_aligned_size;

            Integer* input = &labels[global_id * vector_offset];
            Integer* output = &sorted[global_id * vector_offset];
            Integer* counters = &radixbuf[global_id * radix_range];
            //  Radix sort
            for (std::uint32_t i = 0; i < radix_count; i++) {
                std::uint8_t* cinput = static_cast<std::uint8_t*>(static_cast<void*>(input));
                for (std::uint32_t j = local_id; j < radix_range; j += local_size)
                    counters[j] = 0;
                //  Count elements in sub group to write once per value
                for (std::uint32_t j = local_id; j < group_aligned_size + local_size;
                     j += local_size) {
                    bool exists = j < group_aligned_size || local_id < rem;
                    std::uint8_t c = exists ? cinput[j * radix_count + i] : 0;
                    std::uint32_t entry = 0;
                    bool entry_found = false;
                    for (std::uint32_t k = 0; k < local_size; k++) {
                        bool correct = j < group_aligned_size || k < rem;
                        std::uint32_t done = sycl::ONEAPI::broadcast(sbg, correct ? 0 : 1, k);
                        if (done)
                            break;
                        std::uint8_t value = sycl::ONEAPI::broadcast(sbg, c, k);
                        if (!entry_found && value == c) {
                            entry = k;
                            entry_found = true;
                        }
                        Integer count =
                            sycl::ONEAPI::reduce(sbg,
                                                 static_cast<Integer>(exists && value == c ? 1 : 0),
                                                 cl::sycl::ONEAPI::plus<Integer>());
                        if (entry_found && entry == local_id && entry == k) {
                            counters[value] += count;
                        }
                    }
                    sbg.barrier();
                }
                //  Parallel scan on counters to generate offsets in place
                Integer offset = 0;
                for (std::uint32_t j = local_id; j < radix_range; j += local_size) {
                    Integer value = counters[j];
                    Integer boundary =
                        sycl::ONEAPI::exclusive_scan(sbg, value, cl::sycl::ONEAPI::plus<Integer>());
                    counters[j] = offset + boundary;
                    Integer partial_offset =
                        sycl::ONEAPI::reduce(sbg, value, cl::sycl::ONEAPI::plus<Integer>());
                    offset += partial_offset;
                }

                sbg.barrier();
                for (std::uint32_t j = local_id; j < group_aligned_size + local_size;
                     j += local_size) {
                    bool exists = j < group_aligned_size || local_id < rem;
                    std::uint8_t c = exists ? cinput[j * radix_count + i] : 0;
                    Integer local_offset = 0;
                    std::uint32_t entry = 0;
                    bool entry_found = false;

                    for (std::uint32_t k = 0; k < local_size; k++) {
                        bool correct = j < group_aligned_size || k < rem;
                        std::uint32_t done = sycl::ONEAPI::broadcast(sbg, correct ? 0 : 1, k);
                        if (done)
                            break;
                        std::uint32_t skip = sycl::ONEAPI::broadcast(sbg, entry_found ? 1 : 0, k);
                        if (skip)
                            continue;
                        std::uint8_t value = sycl::ONEAPI::broadcast(sbg, c, k);
                        if (!entry_found && value == c) {
                            entry = k;
                            entry_found = true;
                        }
                        Integer offset = sycl::ONEAPI::exclusive_scan(
                            sbg,
                            static_cast<Integer>(exists && value == c ? 1 : 0),
                            cl::sycl::ONEAPI::plus<Integer>());
                        if (value == c) {
                            local_offset = offset + counters[value];
                        }
                        Integer count =
                            sycl::ONEAPI::reduce(sbg,
                                                 static_cast<Integer>(exists && value == c ? 1 : 0),
                                                 cl::sycl::ONEAPI::plus<Integer>());
                        if (entry_found && entry == local_id && entry == k) {
                            counters[value] += count;
                        }
                    }
                    sbg.barrier();
                    if (exists)
                        output[local_offset] = input[j];
                }
                std::swap(input, output);
            }
            for (std::uint32_t i = local_id; i < sorted_elem_count; i += local_size)
                output[i] = input[i];
        });
    });
    return event;
}

#define INSTANTIATE_SORT_INDICES(F, I)                                                 \
    template ONEDAL_EXPORT sycl::event radix_sort_indices_inplace<F, I>(sycl::queue&,  \
                                                                        ndview<F, 1>&, \
                                                                        ndview<I, 1>&, \
                                                                        ndview<F, 1>&, \
                                                                        ndview<I, 1>&, \
                                                                        const event_vector&);

#define INSTANTIATE_SORT(I)                                               \
    template ONEDAL_EXPORT sycl::event radix_sort<I>(sycl::queue & queue, \
                                                     ndview<I, 2>&,       \
                                                     ndview<I, 2>&,       \
                                                     ndview<I, 2>&,       \
                                                     std::uint32_t,       \
                                                     const event_vector&);

INSTANTIATE_SORT_INDICES(float, std::uint32_t)
INSTANTIATE_SORT_INDICES(double, std::uint32_t)
INSTANTIATE_SORT_INDICES(float, std::int32_t)
INSTANTIATE_SORT_INDICES(double, std::int32_t)

INSTANTIATE_SORT(std::int32_t)
INSTANTIATE_SORT(std::uint32_t)
} // namespace oneapi::dal::backend::primitives
