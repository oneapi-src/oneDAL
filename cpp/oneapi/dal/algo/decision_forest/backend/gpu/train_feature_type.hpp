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

#pragma once

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/communicator.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace pr = dal::backend::primitives;
namespace bk = dal::backend;

template <typename Float, typename Bin, typename Index = std::int32_t>
class indexed_features {
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;

public:
    struct feature_entry {
        Index bin_count_ = 0;
        Index offset_ = 0;
        pr::ndarray<Float, 1> bin_borders_nd_; //right bin borders
    };

    indexed_features(sycl::queue& q,
                     comm_t& comm,
                     std::int64_t min_bin_size,
                     std::int64_t max_bins);
    ~indexed_features() = default;

    static std::int64_t get_required_mem_size(std::int64_t row_count,
                                              std::int64_t column_count,
                                              std::int64_t max_bins);

    sycl::event operator()(const table& tbl, const bk::event_vector& deps = {});

    Index get_bin_count(std::int64_t column_idx) const {
        return entries_[column_idx].bin_count_;
    }
    Index get_total_bin_count() const {
        return total_bins_;
    }

    pr::ndarray<Float, 1> get_bin_borders(Index column_idx) {
        return entries_[column_idx].bin_borders_nd_;
    }

    const pr::ndarray<Index, 1> get_bin_offsets() const {
        return bin_offsets_nd_;
    }

    pr::ndarray<Bin, 2> get_full_data() const {
        return full_data_nd_;
    }

    Index get_row_count() const {
        return row_count_;
    }
    Index get_column_count() const {
        return column_count_;
    }

private:
    sycl::event extract_column(const pr::ndarray<Float, 2>& data_nd,
                               pr::ndarray<Float, 1>& values_nd,
                               pr::ndarray<Index, 1>& indices_nd,
                               Index feature_id,
                               const bk::event_vector& deps = {});
    sycl::event collect_bin_borders(const pr::ndarray<Float, 1>& values_nd,
                                    Index row_count,
                                    const pr::ndarray<Index, 1>& bin_offsets_nd,
                                    pr::ndarray<Float, 1>& bin_borders_nd,
                                    Index max_bins,
                                    pr::ndarray<Index, 1>& unique_offsets_nd,
                                    const bk::event_vector& deps = {});

    std::tuple<pr::ndarray<Float, 1>, Index, sycl::event> gather_bin_borders(
        const pr::ndarray<Float, 1>& values_nd,
        Index row_count,
        const bk::event_vector& deps = {});

    std::tuple<pr::ndarray<Float, 1>, Index, sycl::event> gather_bin_borders_distr(
        const pr::ndarray<Float, 1>& values_nd,
        Index row_count,
        const bk::event_vector& deps = {});

    sycl::event fill_bin_map(const pr::ndarray<Float, 1>& values_nd,
                             const pr::ndarray<Index, 1>& indices_nd,
                             const pr::ndarray<Float, 1>& bin_borders_nd,
                             const pr::ndarray<Bin, 1>& bins_nd,
                             Index bin_count,
                             std::size_t local_size,
                             std::size_t local_blocks_count,
                             const bk::event_vector& deps = {});
    sycl::event compute_bins(const pr::ndarray<Float, 1>& values_nd,
                             const pr::ndarray<Index, 1>& indices_nd,
                             pr::ndarray<Bin, 1>& bins_nd,
                             feature_entry& entry,
                             Index entry_idx,
                             const bk::event_vector& deps);

    sycl::event store_column(const pr::ndarray<Bin, 1>& column_data_nd,
                             pr::ndarray<Bin, 2>& full_data_nd,
                             Index column_idx,
                             Index column_count,
                             const bk::event_vector& deps);

    sycl::queue queue_;
    comm_t comm_;

    pr::ndarray<Bin, 2> full_data_nd_;
    pr::ndarray<Index, 1> bin_offsets_nd_;

    std::vector<feature_entry> entries_;

    Index row_count_ = 0;
    Index column_count_ = 0;
    Index total_bins_ = 0;
    Index min_bin_size_ = 0;
    Index max_bins_ = 0;

    static constexpr inline Index max_local_block_count_ = 1024;
};

#endif

} // namespace oneapi::dal::decision_forest::backend
