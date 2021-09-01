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

#include "oneapi/dal/detail/common.hpp"

#include "oneapi/dal/backend/primitives/heap.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_heap.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, typename Index>
struct selection_pair {
    Float dst;
    Index idx;
};

template <typename Float, typename Index>
inline bool operator<(const selection_pair<Float, Index>& lhs,
                      const selection_pair<Float, Index>& rhs) {
    return lhs.dst < rhs.dst;
}

std::int64_t get_max_local_alloc(const sycl::queue& q) {
    return device_local_mem_size(q) / 2;
}

std::int64_t get_preferred_sub_group(const sycl::queue& queue) {
    const auto max_sg = device_max_sg_size(queue);
    constexpr std::int64_t preferred_sg = 16;
    return std::min(max_sg, preferred_sg);
}

template <typename Float>
std::int64_t get_heap_min_k(const sycl::queue& q) {
    return 0;
}

template <typename Float>
std::int64_t get_heap_max_k(const sycl::queue& q) {
    using sel_t = selection_pair<Float, std::int32_t>;
    constexpr auto sel_size_v = sizeof(sel_t);
    const auto mem = get_max_local_alloc(q);
    const auto max_elems = mem / sizeof(Float);
    return max_elems / sel_size_v - 1;
}

template <typename Float, bool dst_out, bool ids_out>
class kernel_select_heap_base {
    static_assert(dst_out || ids_out);
};

template <typename Float>
class kernel_select_heap_base<Float, true, false> {
    using dst_t = Float;

public:
    kernel_select_heap_base() = default;
    kernel_select_heap_base(std::int32_t ids_str, dst_t* const ids_ptr)
            : dst_str_(ids_str),
              dst_ptr_(ids_ptr) {}

    const auto& get_selection_stride() const {
        return dst_str_;
    }

    auto* get_selection_pointer() const {
        return dst_ptr_;
    }

private:
    std::int32_t dst_str_;
    dst_t* dst_ptr_;
};

template <typename Float>
class kernel_select_heap_base<Float, false, true> {
    using idx_t = std::int32_t;

public:
    kernel_select_heap_base() = default;
    kernel_select_heap_base(std::int32_t ids_str, idx_t* const ids_ptr)
            : ids_str_(ids_str),
              ids_ptr_(ids_ptr) {}

    const auto& get_indices_stride() const {
        return ids_str_;
    }

    auto* get_indices_pointer() const {
        return ids_ptr_;
    }

private:
    std::int32_t ids_str_;
    idx_t* ids_ptr_;
};

template <typename Float>
class kernel_select_heap_base<Float, true, true>
        : public kernel_select_heap_base<Float, true, false>,
          public kernel_select_heap_base<Float, false, true> {
    using base_dst_t = kernel_select_heap_base<Float, true, false>;
    using base_ids_t = kernel_select_heap_base<Float, false, true>;

    using idx_t = std::int32_t;
    using dst_t = Float;

public:
    kernel_select_heap_base() = default;
    kernel_select_heap_base(std::int32_t ids_str,
                            std::int32_t dst_str,
                            idx_t* const ids_ptr,
                            dst_t* const dst_ptr)
            : base_dst_t(dst_str, dst_ptr),
              base_ids_t(ids_str, ids_ptr) {}
};

template <typename Float, bool dst_out, bool ids_out, int sg_size>
class kernel_select_heap : public kernel_select_heap_base<Float, dst_out, ids_out> {
    using dst_t = Float;
    using idx_t = std::int32_t;
    using sel_t = selection_pair<dst_t, idx_t>;

    constexpr static inline std::int32_t pbuff_size = 4;
    constexpr static inline idx_t idx_default = dal::detail::limits<idx_t>::min();
    constexpr static inline dst_t dst_default = dal::detail::limits<dst_t>::max();

    constexpr static inline sycl::ONEAPI::maximum<std::int32_t> max_func{};

    using acc_t =
        sycl::accessor<sel_t, 1, sycl::access::mode::read_write, sycl::access::target::local>;

    using base_t = kernel_select_heap_base<Float, dst_out, ids_out>;

public:
    kernel_select_heap() = default;

    template <bool ids_only = (ids_out && !dst_out), typename = std::enable_if_t<ids_only>>
    kernel_select_heap(const dst_t* const src_ptr,
                       idx_t* const ids_ptr,
                       std::int32_t k,
                       std::int32_t width,
                       std::int32_t height,
                       std::int32_t src_str,
                       std::int32_t ids_str,
                       acc_t heaps)
            : base_t(ids_str, ids_ptr),
              src_str_(src_str),
              src_ptr_(src_ptr),
              k_(k),
              width_(width),
              height_(height),
              heaps_(heaps) {}

    template <bool dst_only = (dst_out && !ids_out), typename = std::enable_if_t<dst_only>>
    kernel_select_heap(const dst_t* const src_ptr,
                       dst_t* const dst_ptr,
                       std::int32_t k,
                       std::int32_t width,
                       std::int32_t height,
                       std::int32_t src_str,
                       std::int32_t dst_str,
                       acc_t heaps)
            : base_t(dst_str, dst_ptr),
              src_str_(src_str),
              src_ptr_(src_ptr),
              k_(k),
              width_(width),
              height_(height),
              heaps_(heaps) {}

    template <bool both = (ids_out && dst_out), typename = std::enable_if_t<both>>
    kernel_select_heap(const dst_t* const src_ptr,
                       idx_t* const ids_ptr,
                       dst_t* const dst_ptr,
                       std::int32_t k,
                       std::int32_t width,
                       std::int32_t height,
                       std::int32_t src_str,
                       std::int32_t ids_str,
                       std::int32_t dst_str,
                       acc_t heaps)
            : base_t(ids_str, dst_str, ids_ptr, dst_ptr),
              src_str_(src_str),
              src_ptr_(src_ptr),
              k_(k),
              width_(width),
              height_(height),
              heaps_(heaps) {}

    void operator()(sycl::nd_item<1> item) const {
        using sycl::ONEAPI::reduce;

        auto sg = item.get_sub_group();

        const std::int32_t sid = sg.get_group_linear_id();
        const std::int32_t cid = sg.get_local_linear_id();
        const std::int32_t rid = sid + item.get_group_linear_id();
        const std::int32_t sg_width = sg.get_group_range().size();

        if ((rid >= height_) || (cid >= sg_width))
            return;

        dst_t pbuff_dst[pbuff_size] = { dst_default };
        idx_t pbuff_ids[pbuff_size] = { idx_default };
        std::int32_t pbuff_count, prev_count;

        sel_t* const heaps = heaps_.get_pointer();
        sel_t* const curr_heap = heaps + (k_ + 1) * sid;
        sel_t* const curr_heap_end = curr_heap + (k_ + 1);

        // Heap initialization
        for (std::int32_t i = cid; i <= k_; i += sg_width) {
            auto& values = curr_heap[i];
            values.idx = idx_default;
            values.dst = dst_default;
        }
        sg.barrier();

        std::int32_t k_written = 0;
        dst_t worst_val = dst_default;
        const auto step = sg_width * pbuff_size;
        const auto* const row = src_ptr_ + rid * src_str_;
        for (std::int32_t i = 0; i < width_; i += step) {
            const std::int32_t block_start_col = cid + i;

            pbuff_count = 0, prev_count = 0;
            worst_val = curr_heap->dst;

            // Collecting temporary best values in private memory
            for (std::int32_t j = 0; (j < pbuff_size); ++j) {
                const idx_t idx = block_start_col + sg_width * j;
                const bool handle = idx < width_;
                const dst_t val = handle ? *(row + idx) : dst_default;
                pbuff_count += bool(val < worst_val);
                pbuff_ids[prev_count] = idx;
                pbuff_dst[prev_count] = val;
                prev_count = pbuff_count;
            }
            sg.barrier();

            // writing temporary best values into heap
            std::int32_t cid_to_handle, handle_this;
            do {
                handle_this = pbuff_count ? cid : -1;
                k_written = reduce(sg, k_written, max_func);
                cid_to_handle = reduce(sg, handle_this, max_func);
                if (cid_to_handle == cid) {
                    for (std::int32_t i = 0; i < pbuff_count; ++i, ++k_written) {
                        sel_t result{ pbuff_dst[i], pbuff_ids[i] };
                        if (k_written > k_) {
                            replace_first(std::move(result), curr_heap, curr_heap + k_);
                        }
                        else {
                            *(curr_heap + k_written) = std::move(result);
                            push_heap(curr_heap, curr_heap + k_written);
                        }
                    }
                    pbuff_count = 0;
                }
                sg.barrier();
            } while (cid_to_handle > -1);
        }

        // Sorting heap before writing out
        // TODO: Think out if it can be performed in parallel
        if (cid == 0) {
            make_heap(curr_heap, curr_heap_end);
            sort_heap(curr_heap, curr_heap_end);
        }
        sg.barrier();

        // Writing output from heap
        for (std::int32_t i = cid; i < k_; i += sg_width) {
            const auto& values = curr_heap[i];
            if constexpr (ids_out) {
                *(this->get_indices_pointer() + this->get_indices_stride() * rid + i) = values.idx;
            }
            if constexpr (dst_out) {
                *(this->get_selection_pointer() + this->get_selection_stride() * rid + i) =
                    values.dst;
            }
        }
    }

private:
    std::int32_t src_str_;
    const dst_t* src_ptr_;

    std::int32_t k_;
    std::int32_t width_;
    std::int32_t height_;

    acc_t heaps_;
};

template <typename Float, bool dst_out, bool ids_out, int sg_size>
sycl::event select(sycl::queue& queue,
                   const ndview<Float, 2>& data,
                   std::int64_t k,
                   ndview<Float, 2>& selection,
                   ndview<std::int32_t, 2>& indices,
                   const event_vector& deps) {
    using kernel_t = kernel_select_heap<Float, dst_out, ids_out, sg_size>;
    using sel_t = selection_pair<Float, std::int32_t>;
    using acc_t =
        sycl::accessor<sel_t, 1, sycl::access::mode::read_write, sycl::access::target::local>;

    const auto pref_sbg = get_preferred_sub_group(queue);
    if (pref_sbg == sg_size) {
        ONEDAL_ASSERT(get_heap_min_k<Float>(queue) < k);
        ONEDAL_ASSERT(k < get_heap_max_k<Float>(queue));
        const auto width = data.get_dimension(1);
        const auto height = data.get_dimension(0);
        ONEDAL_ASSERT(!ids_out || indices.get_dimension(1) == k);
        ONEDAL_ASSERT(!dst_out || selection.get_dimension(1) == k);
        ONEDAL_ASSERT(!ids_out || indices.get_dimension(0) == height);
        ONEDAL_ASSERT(!dst_out || selection.get_dimension(0) == height);
        const auto max_wkg = propose_wg_size(queue);
        const auto available_mem = get_max_local_alloc(queue);
        const auto mem_bound = available_mem / ((k + 1) * sizeof(sel_t));
        ONEDAL_ASSERT(mem_bound > 0);
        const auto wkg_bound = max_wkg / pref_sbg;
        const auto wg_size = std::min<std::int64_t>(mem_bound, wkg_bound);
        const auto block_count = height / wg_size + bool(height % wg_size);
        const auto ndrange =
            make_multiple_nd_range_1d({ block_count * wg_size * pref_sbg }, { wg_size * pref_sbg });
        return queue.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            acc_t heaps(make_range_1d(wg_size * (k + 1)), h);
            if constexpr (dst_out && !ids_out) {
                h.parallel_for(
                    ndrange,
                    kernel_t(
                        data.get_data(),
                        selection.get_mutable_data(),
                        dal::detail::integral_cast<std::int32_t>(k),
                        dal::detail::integral_cast<std::int32_t>(width),
                        dal::detail::integral_cast<std::int32_t>(height),
                        dal::detail::integral_cast<std::int32_t>(data.get_leading_stride()),
                        dal::detail::integral_cast<std::int32_t>(selection.get_leading_stride()),
                        heaps));
            }
            if constexpr (ids_out && !dst_out) {
                h.parallel_for(
                    ndrange,
                    kernel_t(data.get_data(),
                             indices.get_mutable_data(),
                             dal::detail::integral_cast<std::int32_t>(k),
                             dal::detail::integral_cast<std::int32_t>(width),
                             dal::detail::integral_cast<std::int32_t>(height),
                             dal::detail::integral_cast<std::int32_t>(data.get_leading_stride()),
                             dal::detail::integral_cast<std::int32_t>(indices.get_leading_stride()),
                             heaps));
            }
            if constexpr (ids_out && dst_out) {
                h.parallel_for(
                    ndrange,
                    kernel_t(
                        data.get_data(),
                        indices.get_mutable_data(),
                        selection.get_mutable_data(),
                        dal::detail::integral_cast<std::int32_t>(k),
                        dal::detail::integral_cast<std::int32_t>(width),
                        dal::detail::integral_cast<std::int32_t>(height),
                        dal::detail::integral_cast<std::int32_t>(data.get_leading_stride()),
                        dal::detail::integral_cast<std::int32_t>(indices.get_leading_stride()),
                        dal::detail::integral_cast<std::int32_t>(selection.get_leading_stride()),
                        heaps));
            }
        });
    }

    if constexpr (sg_size > 0) {
        return select<Float, dst_out, ids_out, sg_size / 2>(queue,
                                                            data,
                                                            k,
                                                            selection,
                                                            indices,
                                                            deps);
    }

    return sycl::event();
}

template <typename Float>
kselect_by_rows_heap<Float>::kselect_by_rows_heap() {}

template <typename Float>
sycl::event kselect_by_rows_heap<Float>::operator()(sycl::queue& queue,
                                                    const ndview<Float, 2>& data,
                                                    std::int64_t k,
                                                    ndview<Float, 2>& selection,
                                                    ndview<std::int32_t, 2>& indices,
                                                    const event_vector& deps) {
    return select<Float, true, true>(queue, data, k, selection, indices, deps);
}

template <typename Float>
sycl::event kselect_by_rows_heap<Float>::operator()(sycl::queue& queue,
                                                    const ndview<Float, 2>& data,
                                                    std::int64_t k,
                                                    ndview<std::int32_t, 2>& indices,
                                                    const event_vector& deps) {
    ndarray<Float, 2> dummy;
    return select<Float, false, true>(queue, data, k, dummy, indices, deps);
}

template <typename Float>
sycl::event kselect_by_rows_heap<Float>::operator()(sycl::queue& queue,
                                                    const ndview<Float, 2>& data,
                                                    std::int64_t k,
                                                    ndview<Float, 2>& selection,
                                                    const event_vector& deps) {
    ndarray<std::int32_t, 2> dummy;
    return select<Float, true, false>(queue, data, k, selection, dummy, deps);
}

#define INSTANTIATE_FLOAT(F)                                     \
    template class kselect_by_rows_heap<F>;                      \
    template std::int64_t get_heap_max_k<F>(const sycl::queue&); \
    template std::int64_t get_heap_min_k<F>(const sycl::queue&);

INSTANTIATE_FLOAT(float);
INSTANTIATE_FLOAT(double);

#undef INSTANTIATE_FLOAT

} // namespace oneapi::dal::backend::primitives
