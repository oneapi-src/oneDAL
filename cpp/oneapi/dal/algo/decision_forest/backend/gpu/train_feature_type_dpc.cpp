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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_feature_type.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/detail/profiler.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::decision_forest::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float, typename Index>
inline sycl::event sort_inplace(sycl::queue& queue_,
                                pr::ndarray<Float, 1>& src,
                                const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(src.get_count() > 0);
    auto src_ind = pr::ndarray<Index, 1>::empty(queue_, { src.get_count() });
    return pr::radix_sort_indices_inplace<Float, Index>{ queue_ }(src, src_ind, deps);
}

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::extract_column(
    const pr::ndarray<Float, 2>& data_nd,
    pr::ndarray<Float, 1>& values_nd,
    pr::ndarray<Index, 1>& indices_nd,
    Index feature_id,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(indexed_features.extract_column, queue_);

    ONEDAL_ASSERT(data_nd.get_count() == row_count_ * column_count_);
    ONEDAL_ASSERT(values_nd.get_count() == row_count_);
    ONEDAL_ASSERT(indices_nd.get_count() == row_count_);

    const Float* data = data_nd.get_data();

    Float* values = values_nd.get_mutable_data();
    Index* indices = indices_nd.get_mutable_data();
    auto column_count = column_count_;

    const sycl::range<1> range = de::integral_cast<std::size_t>(row_count_);

    auto event = queue_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> idx) {
            values[idx] = data[idx * column_count + feature_id];
            indices[idx] = idx;
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::collect_bin_borders(
    const pr::ndarray<Float, 1>& values_nd,
    Index row_count,
    const pr::ndarray<Index, 1>& bin_offsets_nd,
    pr::ndarray<Float, 1>& bin_borders_nd,
    Index max_bins,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(values_nd.get_count() == row_count);
    ONEDAL_ASSERT(bin_offsets_nd.get_count() == max_bins);
    ONEDAL_ASSERT(bin_borders_nd.get_count() == max_bins);

    const sycl::range<1> range = de::integral_cast<std::size_t>(max_bins);

    const Float* values = values_nd.get_data();
    const Index* bin_offsets = bin_offsets_nd.get_data();
    Float* bin_borders = bin_borders_nd.get_mutable_data();

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            bin_borders[idx] = values[bin_offsets[idx]];
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::fill_bin_map(
    const pr::ndarray<Float, 1>& values_nd,
    const pr::ndarray<Index, 1>& indices_nd,
    const pr::ndarray<Float, 1>& bin_borders_nd,
    const pr::ndarray<Bin, 1>& bins_nd,
    Index bin_count,
    std::size_t local_size,
    std::size_t local_blocks_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(indexed_features.fill_bin_map, queue_);

    ONEDAL_ASSERT(values_nd.get_count() == row_count_);
    ONEDAL_ASSERT(indices_nd.get_count() == row_count_);
    ONEDAL_ASSERT(bin_borders_nd.get_count() >= bin_count);
    ONEDAL_ASSERT(bins_nd.get_count() == row_count_);

    const sycl::nd_range<1> nd_range =
        bk::make_multiple_nd_range_1d(de::check_mul_overflow(local_size, local_blocks_count),
                                      local_size);

    const Index row_count = row_count_;
    const Float* values = values_nd.get_data();
    const Index* indices = indices_nd.get_data();
    const Float* bin_borders = bin_borders_nd.get_data();
    Bin* bins = bins_nd.get_mutable_data();

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            const std::uint32_t n_groups = item.get_group_range(0);
            const std::uint32_t n_sub_groups = sbg.get_group_range()[0];
            const std::uint32_t n_total_sub_groups = n_sub_groups * n_groups;
            const Index elems_for_sbg =
                row_count / n_total_sub_groups + bool(row_count % n_total_sub_groups);
            const std::uint32_t local_size = sbg.get_local_range()[0];

            const std::uint32_t local_id = sbg.get_local_id();
            const std::uint32_t sub_group_id = sbg.get_group_id();
            const std::uint32_t group_id =
                item.get_group().get_group_id(0) * n_sub_groups + sub_group_id;

            Index ind_start = group_id * elems_for_sbg;
            Index ind_end =
                sycl::min(static_cast<Index>((group_id + 1) * elems_for_sbg), row_count);

            Index cur_bin = 0;

            for (Index i = ind_start + local_id; i < ind_end; i += local_size) {
                for (Float value = values[i]; bin_borders[cur_bin] < value; ++cur_bin)
                    ;
                bins[indices[i]] = cur_bin;
            }
        });
    });

    event.wait_and_throw();

    return event;
}

template <typename Float, typename Bin, typename Index>
std::tuple<pr::ndarray<Float, 1>, Index, sycl::event>
indexed_features<Float, Bin, Index>::gather_bin_borders(const pr::ndarray<Float, 1>& values_nd,
                                                        Index row_count,
                                                        const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(indexed_features.gather_bin_borders, queue_);

    ONEDAL_ASSERT(values_nd.get_count() == row_count);

    sycl::event::wait_and_throw(deps);

    const Index max_bins = std::min(max_bins_, row_count);

    auto bin_offsets_nd_host = pr::ndarray<Index, 1>::empty({ max_bins });
    auto bin_borders_nd_device =
        pr::ndarray<Float, 1>::empty(queue_, { max_bins }, sycl::usm::alloc::device);

    auto bin_offsets = bin_offsets_nd_host.get_mutable_data();
    Index offset = 0;
    for (Index i = 0; i < max_bins; i++) {
        offset += (row_count + i) / max_bins;
        ONEDAL_ASSERT(offset > 0); // max_bins = min(max_bins_, row_count_) => offset > 0
        bin_offsets[i] = offset - 1;
    }

    bin_offsets_nd_ = bin_offsets_nd_host.to_device(queue_);

    auto last_event = collect_bin_borders(values_nd,
                                          row_count,
                                          bin_offsets_nd_,
                                          bin_borders_nd_device,
                                          max_bins,
                                          { deps });

    Index bin_count = 0;
    auto bin_borders_nd_host = bin_borders_nd_device.to_host(queue_, { last_event });
    auto bin_borders_ptr = bin_borders_nd_host.get_mutable_data();

    for (Index i = 0; i < max_bins; ++i) {
        if (0 == bin_count ||
            (bin_count > 0 && bin_borders_ptr[i] != bin_borders_ptr[bin_count - 1])) {
            bin_borders_ptr[bin_count] = bin_borders_ptr[i];
            bin_count++;
        }
    }

    bin_borders_nd_device = bin_borders_nd_host.slice(0, bin_count).to_device(queue_);

    return std::make_tuple(bin_borders_nd_device, bin_count, last_event);
}

template <typename Float, typename Bin, typename Index>
std::tuple<pr::ndarray<Float, 1>, Index, sycl::event>
indexed_features<Float, Bin, Index>::gather_bin_borders_distr(
    const pr::ndarray<Float, 1>& values_nd,
    Index row_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(indexed_features.gather_bin_borders, queue_);

    ONEDAL_ASSERT(values_nd.get_count() == row_count);

    sycl::event::wait_and_throw(deps);

    sycl::event last_event;
    pr::ndarray<Float, 1> bin_borders_nd_device;
    Index bin_count = 0;

    // gather local bins
    auto [local_bin_borders_nd_device, local_bin_count, event] =
        gather_bin_borders(values_nd, row_count, deps);

    last_event = event;

    Index com_bin_count = 0;
    // using std::int64_t instead of Index because of it is used as displ in gatherv
    auto com_bin_count_arr = pr::ndarray<std::int64_t, 1>::empty({ comm_.get_rank_count() });
    auto com_bin_offset_arr = pr::ndarray<std::int64_t, 1>::empty({ comm_.get_rank_count() });

    std::int64_t lbc_64 = static_cast<std::int64_t>(local_bin_count);
    {
        ONEDAL_PROFILER_TASK(allgather_lbc_64);
        comm_.allgather(lbc_64, com_bin_count_arr.flatten()).wait();
    }

    com_bin_count = local_bin_count;
    {
        ONEDAL_PROFILER_TASK(allreduce_com_bin_count);
        comm_.allreduce(com_bin_count).wait();
    }

    pr::ndarray<Float, 1> com_bin_brd;
    com_bin_brd = pr::ndarray<Float, 1>::empty(queue_, { com_bin_count }, sycl::usm::alloc::device);

    const std::int64_t* com_bin_count_ptr = com_bin_count_arr.get_data();
    std::int64_t* com_bin_offset_ptr = com_bin_offset_arr.get_mutable_data();

    std::int64_t offset = 0;
    for (Index i = 0; i < comm_.get_rank_count(); ++i) {
        com_bin_offset_ptr[i] = offset;
        offset += com_bin_count_ptr[i];
    }

    {
        ONEDAL_PROFILER_TASK(allgather_com_bin_borders, queue_);
        comm_
            .allgatherv(local_bin_borders_nd_device.flatten(queue_),
                        com_bin_brd.flatten(queue_),
                        com_bin_count_arr.get_data(),
                        com_bin_offset_arr.get_data())
            .wait();
    }

    if (comm_.is_root_rank()) {
        last_event = sort_inplace<Float, Index>(queue_, com_bin_brd, { last_event });

        // filter out fin bin set
        auto [fin_borders_nd_device_temp, fin_bin_count_temp, event] =
            gather_bin_borders(com_bin_brd, com_bin_count);
        event.wait_and_throw();

        bin_borders_nd_device = fin_borders_nd_device_temp;
        bin_count = fin_bin_count_temp;
    }

    {
        ONEDAL_PROFILER_TASK(bcast_bin_count);
        comm_.bcast(bin_count).wait();
    }

    if (!comm_.is_root_rank()) {
        bin_borders_nd_device =
            pr::ndarray<Float, 1>::empty(queue_, { bin_count }, sycl::usm::alloc::device);
    }

    {
        ONEDAL_PROFILER_TASK(bcast_bin_borders, queue_);
        comm_.bcast(queue_, bin_borders_nd_device.get_mutable_data(), bin_count).wait();
    }

    return std::make_tuple(bin_borders_nd_device, bin_count, last_event);
}

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::compute_bins(
    const pr::ndarray<Float, 1>& values_nd,
    const pr::ndarray<Index, 1>& indices_nd,
    pr::ndarray<Bin, 1>& bins_nd,
    feature_entry& entry,
    Index entry_idx,
    const bk::event_vector& deps) {
    ONEDAL_ASSERT(values_nd.get_count() == row_count_);
    ONEDAL_ASSERT(indices_nd.get_count() == row_count_);
    ONEDAL_ASSERT(bins_nd.get_count() == row_count_);

    sycl::event::wait_and_throw(deps);

    sycl::event last_event;

    auto [bin_borders_nd_device, bin_count, event] =
        comm_.get_rank_count() > 1 ? gather_bin_borders_distr(values_nd, row_count_, deps)
                                   : gather_bin_borders(values_nd, row_count_, deps);
    last_event = event;

    const Index local_size = bk::device_max_sg_size(queue_);
    const Index local_block_count = max_local_block_count_ * local_size < row_count_
                                        ? max_local_block_count_
                                        : (row_count_ / local_size) + bool(row_count_ % local_size);

    last_event = fill_bin_map(values_nd,
                              indices_nd,
                              bin_borders_nd_device,
                              bins_nd,
                              bin_count,
                              local_size,
                              local_block_count,
                              { last_event });

    entry.bin_count_ = bin_count;
    entry.bin_borders_nd_ = bin_borders_nd_device;

    return last_event;
}

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::store_column(
    const pr::ndarray<Bin, 1>& column_data_nd,
    pr::ndarray<Bin, 2>& full_data_nd,
    Index column_idx,
    Index column_count,
    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(indexed_features.store_column, queue_);

    ONEDAL_ASSERT(column_data_nd.get_count() == row_count_);
    ONEDAL_ASSERT(full_data_nd.get_count() == row_count_ * column_count_);

    const Bin* column_data = column_data_nd.get_data();
    Bin* full_data = full_data_nd.get_mutable_data();

    const sycl::range<1> range = de::integral_cast<std::size_t>(column_data_nd.get_dimension(0));

    auto event = queue_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> idx) {
            full_data[idx * column_count + column_idx] = column_data[idx];
        });
    });

    return event;
}

template <typename Float, typename Bin, typename Index>
indexed_features<Float, Bin, Index>::indexed_features(sycl::queue& q,
                                                      comm_t& comm,
                                                      std::int64_t min_bin_size,
                                                      std::int64_t max_bins)
        : queue_(q),
          comm_(comm) {
    min_bin_size_ = de::integral_cast<Index>(min_bin_size);
    max_bins_ = de::integral_cast<Index>(max_bins);
}

template <typename Float, typename Bin, typename Index>
std::int64_t indexed_features<Float, Bin, Index>::get_required_mem_size(std::int64_t row_count,
                                                                        std::int64_t column_count,
                                                                        std::int64_t max_bins) {
    std::int64_t required_mem = 0;
    required_mem += sizeof(Bin) * (column_count + 1); // bin_offsets
    required_mem +=
        sizeof(Bin) * row_count * column_count; // data vs ftrs bin map table (full_data_nd)
    required_mem +=
        sizeof(Float) * column_count * std::min(max_bins, row_count); // bin_borders for each column

    return required_mem;
}

template <typename Float, typename Bin, typename Index>
sycl::event indexed_features<Float, Bin, Index>::operator()(const table& tbl,
                                                            const bk::event_vector& deps) {
    sycl::event::wait_and_throw(deps);

    if (tbl.get_row_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_rows());
    }
    if (tbl.get_column_count() > de::limits<Index>::max()) {
        throw domain_error(dal::detail::error_messages::invalid_range_of_columns());
    }

    row_count_ = de::integral_cast<Index>(tbl.get_row_count());
    column_count_ = de::integral_cast<Index>(tbl.get_column_count());
    total_bins_ = 0;

    const auto data_nd_ = pr::table2ndarray<Float>(queue_, tbl, sycl::usm::alloc::device);
    //allocating buffers
    full_data_nd_ =
        pr::ndarray<Bin, 2>::empty(queue_, { row_count_, column_count_ }, sycl::usm::alloc::device);

    entries_.resize(column_count_);

    auto values_nd = pr::ndarray<Float, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);
    auto indices_nd =
        pr::ndarray<Index, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);
    std::vector<pr::ndarray<Bin, 1>> column_bin_vec_;
    column_bin_vec_.resize(column_count_);

    for (Index i = 0; i < column_count_; i++) {
        column_bin_vec_[i] =
            pr::ndarray<Bin, 1>::empty(queue_, { row_count_ }, sycl::usm::alloc::device);
    }

    pr::radix_sort_indices_inplace<Float, Index> sort{ queue_ };

    sycl::event last_event;

    for (Index i = 0; i < column_count_; i++) {
        last_event = extract_column(data_nd_, values_nd, indices_nd, i, { last_event });
        last_event = sort(values_nd, indices_nd, { last_event });
        last_event =
            compute_bins(values_nd, indices_nd, column_bin_vec_[i], entries_[i], i, { last_event });
    }

    last_event.wait_and_throw();

    auto bin_offsets_nd_host = pr::ndarray<Index, 1>::empty({ column_count_ + 1 });
    auto bin_offsets = bin_offsets_nd_host.get_mutable_data();

    Index total = 0;
    for (Index i = 0; i < column_count_; i++) {
        last_event =
            store_column(column_bin_vec_[i], full_data_nd_, i, column_count_, { last_event });
        bin_offsets[i] = total;
        entries_[i].offset_ = total;
        total += entries_[i].bin_count_;
    }

    bin_offsets[column_count_] = total;
    total_bins_ = total;

    bin_offsets_nd_ = bin_offsets_nd_host.to_device(queue_);

    return last_event;
}

#define INSTANTIATE(F, B) template class indexed_features<F, B>;

INSTANTIATE(float, std::uint32_t);
INSTANTIATE(float, std::uint8_t);
INSTANTIATE(double, std::uint32_t);
INSTANTIATE(double, std::uint8_t);

} // namespace oneapi::dal::decision_forest::backend

#endif
