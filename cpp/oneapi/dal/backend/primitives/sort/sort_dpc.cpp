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

inline std::uint32_t inv_bits(std::uint32_t x) {
    return x ^ (-(x >> 31) | 0x80000000u);
}

inline std::uint64_t inv_bits(std::uint64_t x) {
    return x ^ (-(x >> 63) | 0x8000000000000000ul);
}

using sycl::ONEAPI::broadcast;
using sycl::ONEAPI::reduce;
using sycl::ONEAPI::plus;
using sycl::ONEAPI::exclusive_scan;

template <typename Float, typename IndexType>
sycl::event radix_sort_indices_inplace<Float, IndexType>::radix_scan(
    sycl::queue& queue,
    const ndview<Float, 1>& val,
    ndarray<IndexType, 1>& part_hist,
    IndexType elem_count,
    std::uint32_t bit_offset,
    std::int64_t local_size,
    std::int64_t local_hist_count,
    sycl::event& deps) {
    ONEDAL_ASSERT(part_hist.get_count() == hist_buff_size_);

    const sycl::range<1> global(local_size * local_hist_count);
    const sycl::range<1> local(local_size);
    const sycl::nd_range<1> nd_range(global, local);

    const radix_integer_t* val_ptr = reinterpret_cast<const radix_integer_t*>(val.get_data());
    IndexType* part_hist_ptr = part_hist.get_mutable_data();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
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
            IndexType ind_end =
                sycl::min(static_cast<IndexType>((group_id + 1) * elems_for_sbg), elem_count);

            IndexType offset[radix_range_];
            for (std::uint32_t i = 0; i < radix_range_; i++) {
                offset[i] = 0;
            }

            for (IndexType i = ind_start + local_id; i < ind_end; i += local_size) {
                radix_integer_t data_bits = ((inv_bits(val_ptr[i]) >> bit_offset) & radix_range_1_);
                for (std::uint32_t j = 0; j < radix_range_; j++) {
                    IndexType value = static_cast<IndexType>(data_bits == j);
                    IndexType partial_offset = reduce(sbg, value, plus<IndexType>());
                    offset[j] += partial_offset;
                }
            }

            if (local_id == 0) {
                for (std::uint32_t j = 0; j < radix_range_; j++) {
                    part_hist_ptr[group_id * radix_range_ + j] = offset[j];
                }
            }
        });
    });

    return event;
}

template <typename Float, typename IndexType>
sycl::event radix_sort_indices_inplace<Float, IndexType>::radix_hist_scan(
    sycl::queue& queue,
    const ndarray<IndexType, 1>& part_hist,
    ndarray<IndexType, 1>& part_prefix_hist,
    std::int64_t local_size,
    std::int64_t local_hist_count,
    sycl::event& deps) {
    ONEDAL_ASSERT(part_hist.get_count() == hist_buff_size_);
    ONEDAL_ASSERT(part_prefix_hist.get_count() == hist_buff_size_);

    const IndexType* part_hist_ptr = part_hist.get_data();
    IndexType* part_prefix_hist_ptr = part_prefix_hist.get_mutable_data();

    const sycl::range<1> global(local_size);
    const sycl::range<1> local(local_size);
    const sycl::nd_range<1> nd_range(global, local);

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            if (sbg.get_group_id()[0] > 0) {
                return;
            }

            const std::uint32_t local_size = sbg.get_local_range()[0];
            const std::uint32_t local_id = sbg.get_local_id()[0];

            IndexType offset[radix_range_];
            for (std::uint32_t i = 0; i < radix_range_; i++) {
                offset[i] = 0;
            }

            for (std::uint32_t i = local_id; i < local_hist_count; i += local_size) {
                for (std::uint32_t j = 0; j < radix_range_; j++) {
                    IndexType value = part_hist_ptr[i * radix_range_ + j];
                    IndexType boundary = exclusive_scan(sbg, value, plus<IndexType>());
                    part_prefix_hist_ptr[i * radix_range_ + j] = offset[j] + boundary;
                    IndexType partial_offset = reduce(sbg, value, plus<IndexType>());
                    offset[j] += partial_offset;
                }
            }

            if (local_id == 0) {
                IndexType total_sum = 0;
                for (std::uint32_t j = 0; j < radix_range_; j++) {
                    part_prefix_hist_ptr[local_hist_count * radix_range_ + j] = total_sum;
                    total_sum += offset[j];
                }
            }
        });
    });

    return event;
}

template <typename Float, typename IndexType>
sycl::event radix_sort_indices_inplace<Float, IndexType>::radix_reorder(
    sycl::queue& queue,
    const ndview<Float, 1>& val_in,
    const ndview<IndexType, 1>& ind_in,
    const ndview<IndexType, 1>& part_prefix_hist,
    ndview<Float, 1>& val_out,
    ndview<IndexType, 1>& ind_out,
    IndexType elem_count,
    std::uint32_t bit_offset,
    std::int64_t local_size,
    std::int64_t local_hist_count,
    sycl::event& deps) {
    ONEDAL_ASSERT(part_prefix_hist.get_count() == ((local_hist_count + 1) << radix_bits));
    ONEDAL_ASSERT(val_in.get_count() == ind_in.get_count());
    ONEDAL_ASSERT(val_in.get_count() == val_out.get_count());
    ONEDAL_ASSERT(val_in.get_count() == ind_out.get_count());

    const radix_integer_t* val_in_ptr = reinterpret_cast<const radix_integer_t*>(val_in.get_data());
    const IndexType* ind_in_ptr = ind_in.get_data();
    const IndexType* part_prefix_hist_ptr = part_prefix_hist.get_data();
    radix_integer_t* val_out_ptr = reinterpret_cast<radix_integer_t*>(val_out.get_mutable_data());
    IndexType* ind_out_ptr = ind_out.get_mutable_data();

    const sycl::range<1> global(local_size * local_hist_count);
    const sycl::range<1> local(local_size);
    const sycl::nd_range<1> nd_range(global, local);

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
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
            IndexType ind_end =
                sycl::min(static_cast<IndexType>((group_id + 1) * elems_for_sbg), elem_count);

            IndexType offset[radix_range_];

            for (std::uint32_t i = 0; i < radix_range_; i++) {
                offset[i] = part_prefix_hist_ptr[group_id * radix_range_ + i] +
                            part_prefix_hist_ptr[n_total_sub_groups * radix_range_ + i];
            }

            for (IndexType i = ind_start + local_id; i < ind_end; i += local_size) {
                radix_integer_t data_value = val_in_ptr[i];
                radix_integer_t data_bits = ((inv_bits(data_value) >> bit_offset) & radix_range_1_);
                IndexType pos_new = 0;
                for (std::uint32_t j = 0; j < radix_range_; j++) {
                    IndexType value = static_cast<IndexType>(data_bits == j);
                    IndexType boundary = exclusive_scan(sbg, value, plus<IndexType>());
                    pos_new |= value * (offset[j] + boundary);
                    IndexType partial_offset = reduce(sbg, value, plus<IndexType>());
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
radix_sort_indices_inplace<Float, IndexType>::radix_sort_indices_inplace(sycl::queue& queue,
                                                                         std::int64_t elem_count)
        : queue_(queue),
          elem_count_(0) {
    init(queue, elem_count);
}

template <typename Float, typename IndexType>
radix_sort_indices_inplace<Float, IndexType>::~radix_sort_indices_inplace() {
    sort_event_.wait_and_throw();
}

template <typename Float, typename IndexType>
void radix_sort_indices_inplace<Float, IndexType>::init(sycl::queue& queue,
                                                        std::int64_t elem_count) {
    std::uint32_t uint_elem_count = de::integral_cast<std::uint32_t>(elem_count);
    if (uint_elem_count && elem_count_ != uint_elem_count) {
        elem_count_ = uint_elem_count;
        local_size_ = preferable_sbg_size_;
        local_hist_count_ = max_local_hist_count_ * local_size_ < elem_count_
                                ? max_local_hist_count_
                                : (elem_count_ / local_size_) + bool(elem_count_ % local_size_);

        hist_buff_size_ = (local_hist_count_ + 1) << radix_bits_;

        part_hist_ =
            ndarray<IndexType, 1>::empty(queue, { hist_buff_size_ }, sycl::usm::alloc::device);
        part_prefix_hist_ =
            ndarray<IndexType, 1>::empty(queue, { hist_buff_size_ }, sycl::usm::alloc::device);
        val_buff_ = ndarray<Float, 1>::empty(queue_, { elem_count_ }, sycl::usm::alloc::device);

        ind_buff_ = ndarray<IndexType, 1>::empty(queue_, { elem_count_ }, sycl::usm::alloc::device);
    }
}

template <typename Float, typename IndexType>
sycl::event radix_sort_indices_inplace<Float, IndexType>::operator()(ndview<Float, 1>& val_in,
                                                                     ndview<IndexType, 1>& ind_in,
                                                                     const event_vector& deps) {
    ONEDAL_ASSERT(val_in.has_mutable_data());
    ONEDAL_ASSERT(ind_in.has_mutable_data());
    ONEDAL_ASSERT(val_in.get_count() == ind_in.get_count());

    sycl::event::wait_and_throw(deps);
    sort_event_.wait_and_throw();

    init(queue_, val_in.get_count());

    std::uint32_t rev = 1;

    sycl::event res_deps = {};
    for (std::uint32_t bit_offset = 0; bit_offset < byte_range_ * sizeof(Float);
         bit_offset += radix_bits_, rev ^= 1) {
        if (rev) {
            auto scan_deps = radix_scan(queue_,
                                        val_in,
                                        part_hist_,
                                        elem_count_,
                                        bit_offset,
                                        local_size_,
                                        local_hist_count_,
                                        res_deps);
            auto hist_scan_deps = radix_hist_scan(queue_,
                                                  part_hist_,
                                                  part_prefix_hist_,
                                                  local_size_,
                                                  local_hist_count_,
                                                  scan_deps);
            res_deps = radix_reorder(queue_,
                                     val_in,
                                     ind_in,
                                     part_prefix_hist_,
                                     val_buff_,
                                     ind_buff_,
                                     elem_count_,
                                     bit_offset,
                                     local_size_,
                                     local_hist_count_,
                                     hist_scan_deps);
        }
        else {
            auto scan_deps = radix_scan(queue_,
                                        val_buff_,
                                        part_hist_,
                                        elem_count_,
                                        bit_offset,
                                        local_size_,
                                        local_hist_count_,
                                        res_deps);
            auto hist_scan_deps = radix_hist_scan(queue_,
                                                  part_hist_,
                                                  part_prefix_hist_,
                                                  local_size_,
                                                  local_hist_count_,
                                                  scan_deps);
            res_deps = radix_reorder(queue_,
                                     val_buff_,
                                     ind_buff_,
                                     part_prefix_hist_,
                                     val_in,
                                     ind_in,
                                     elem_count_,
                                     bit_offset,
                                     local_size_,
                                     local_hist_count_,
                                     hist_scan_deps);
        }
    }

    sort_event_ = res_deps;
    return res_deps;
}

template <typename Integer>
radix_sort<Integer>::radix_sort(sycl::queue& queue, std::int64_t vector_count)
        : queue_(queue),
          vector_count_(0) {
    init(queue, vector_count);
}

template <typename Integer>
radix_sort<Integer>::~radix_sort() {
    sort_event_.wait_and_throw();
}

template <typename Integer>
void radix_sort<Integer>::init(sycl::queue& queue, std::int64_t vector_count) {
    std::uint32_t uint_vector_count = de::integral_cast<std::uint32_t>(vector_count);
    if (uint_vector_count && vector_count_ != uint_vector_count) {
        vector_count_ = uint_vector_count;

        buffer_ =
            ndarray<Integer, 2>::empty(queue_,
                                       { vector_count_, expected_buffer_size_for_one_vector_ },
                                       sycl::usm::alloc::device);
    }
}

template <typename Integer>
sycl::event radix_sort<Integer>::operator()(ndview<Integer, 2>& val_in,
                                            ndview<Integer, 2>& val_out,
                                            std::int64_t sorted_elem_count,
                                            const event_vector& deps) {
    // radixBuf should be big enough to accumulate radix_range elements
    ONEDAL_ASSERT(val_in.get_dimension(0) == val_out.get_dimension(0));
    ONEDAL_ASSERT(val_in.get_dimension(1) == val_out.get_dimension(1));
    ONEDAL_ASSERT(val_out.has_mutable_data());
    ONEDAL_ASSERT(sorted_elem_count > 0);

    sort_event_.wait_and_throw();

    const std::uint32_t vector_count = de::integral_cast<std::uint32_t>(val_in.get_dimension(0));
    const std::uint32_t vector_offset = de::integral_cast<std::uint32_t>(val_in.get_dimension(1));

    const std::uint32_t _sorted_elem_count = de::integral_cast<std::uint32_t>(sorted_elem_count);

    init(queue_, val_in.get_dimension(0));

    Integer* labels = val_in.get_mutable_data();
    Integer* sorted = val_out.get_mutable_data();
    Integer* radixbuf = buffer_.get_mutable_data();

    const sycl::range<2> global(vector_count, preferable_wg_size_);
    const sycl::range<2> local(1, preferable_wg_size_);

    const sycl::nd_range<2> nd_range(global, local);

    sort_event_ = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            auto sbg = item.get_sub_group();
            // Code is written for a single subgroup. It's necessary to adjust the local range if idle subgoups are presented
            if (sbg.get_group_id()[0] > 0) {
                return;
            }

            const std::uint32_t global_id = item.get_global_id()[0];
            const std::uint32_t local_id = item.get_local_id()[1];

            const std::uint32_t local_size = sbg.get_local_range()[0];
            const std::uint32_t group_aligned_size =
                _sorted_elem_count - _sorted_elem_count % local_size;
            const std::uint32_t rem = _sorted_elem_count - group_aligned_size;

            Integer* input = &labels[global_id * vector_offset];
            Integer* output = &sorted[global_id * vector_offset];
            Integer* counters = &radixbuf[global_id * radix_range];
            //  Radix sort
            for (std::uint32_t i = 0; i < radix_count; i++) {
                std::uint8_t* cinput = reinterpret_cast<std::uint8_t*>(input);
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
                        std::uint32_t done = broadcast(sbg, correct ? 0 : 1, k);
                        if (done)
                            break;
                        std::uint8_t value = broadcast(sbg, c, k);
                        if (!entry_found && value == c) {
                            entry = k;
                            entry_found = true;
                        }
                        Integer count = reduce(sbg,
                                               static_cast<Integer>(exists && value == c ? 1 : 0),
                                               plus<Integer>());
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
                    Integer boundary = exclusive_scan(sbg, value, plus<Integer>());
                    counters[j] = offset + boundary;
                    Integer partial_offset = reduce(sbg, value, plus<Integer>());
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
                        std::uint32_t done = broadcast(sbg, correct ? 0 : 1, k);
                        if (done)
                            break;
                        std::uint32_t skip = broadcast(sbg, entry_found ? 1 : 0, k);
                        if (skip)
                            continue;
                        std::uint8_t value = broadcast(sbg, c, k);
                        if (!entry_found && value == c) {
                            entry = k;
                            entry_found = true;
                        }
                        Integer offset =
                            exclusive_scan(sbg,
                                           static_cast<Integer>(exists && value == c ? 1 : 0),
                                           plus<Integer>());
                        if (value == c) {
                            local_offset = offset + counters[value];
                        }
                        Integer count = reduce(sbg,
                                               static_cast<Integer>(exists && value == c ? 1 : 0),
                                               plus<Integer>());
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
            for (std::uint32_t i = local_id; i < _sorted_elem_count; i += local_size)
                output[i] = input[i];
        });
    });

    return sort_event_;
}

#define INSTANTIATE_SORT_INDICES(F, I) \
    template class ONEDAL_EXPORT radix_sort_indices_inplace<F, I>;

#define INSTANTIATE_SORT(I) template class ONEDAL_EXPORT radix_sort<I>;

INSTANTIATE_SORT_INDICES(float, std::uint32_t)
INSTANTIATE_SORT_INDICES(double, std::uint32_t)
INSTANTIATE_SORT_INDICES(float, std::int32_t)
INSTANTIATE_SORT_INDICES(double, std::int32_t)

INSTANTIATE_SORT(std::int32_t)
INSTANTIATE_SORT(std::uint32_t)
} // namespace oneapi::dal::backend::primitives
