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

#include <sycl/sycl.hpp>

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/communicator.hpp"

namespace oneapi::dal::dbscan::backend {

namespace bk = oneapi::dal::backend;
namespace spmd = oneapi::dal::preview::spmd;
namespace pr = oneapi::dal::backend::primitives;

template <typename Float>
class data_keeper {
public:
    explicit data_keeper(const bk::context_gpu& ctx)
            : comm_(ctx.get_communicator()),
              queue_(ctx.get_queue()) {}
    auto get_data() const {
        return data_;
    }
    bool has_weights() const {
        return weights_.count() > 0;
    }
    auto get_weights() const {
        return weights_;
    }
    auto get_block_start() const {
        return block_start_;
    }
    auto get_block_size() const {
        return block_size_;
    }
    auto get_row_count() const {
        return row_count_;
    }
    auto get_column_count() const {
        return column_count_;
    }
    void init(const table& local_data, const table& local_weights) {
        std::int64_t rank = comm_.get_rank();
        std::int64_t rank_count = comm_.get_rank_count();

        init_data_dimensions(local_data);
        collect_local_row_counts(rank, rank_count);
        compute_rank_offset(rank);
        compute_input_layouts(rank_count);
        collect_data(local_data);
        collect_weights(local_weights);
    }

protected:
    void collect_local_row_counts(std::int64_t rank, std::int64_t rank_count) {
        ONEDAL_ASSERT(rank >= 0);
        ONEDAL_ASSERT(rank_count > 0);
        local_row_counts_ = array<std::int64_t>::zeros(rank_count);
        auto local_row_counts_ptr = local_row_counts_.get_mutable_data();
        local_row_counts_ptr[rank] = block_size_;
        comm_.allreduce(local_row_counts_).wait();
    }
    void init_data_dimensions(const table& local_data) {
        block_size_ = local_data.get_row_count();
        column_count_ = local_data.get_column_count();
        ONEDAL_ASSERT(block_size_ > 0);
        ONEDAL_ASSERT(column_count_ > 0);
        row_count_ = block_size_;
        comm_.allreduce(row_count_).wait();
    }
    void compute_rank_offset(std::int64_t rank) {
        ONEDAL_ASSERT(rank >= 0);
        block_start_ = 0;
        for (std::int64_t i = 0; i < rank; i++) {
            ONEDAL_ASSERT(local_row_counts_.get_data()[i] >= 0);
            block_start_ += local_row_counts_.get_data()[i];
        }
    }
    void compute_input_layouts(std::int64_t rank_count) {
        ONEDAL_ASSERT(rank_count > 0);
        ONEDAL_ASSERT(column_count_ > 0);
        local_data_counts_ = array<std::int64_t>::zeros(rank_count);
        displs_weights_ = array<std::int64_t>::zeros(rank_count);
        displs_data_ = array<std::int64_t>::zeros(rank_count);
        total_count_data_ = 0;
        total_count_weights_ = 0;
        for (std::int64_t i = 0; i < rank_count; i++) {
            displs_weights_.get_mutable_data()[i] += total_count_weights_;
            displs_data_.get_mutable_data()[i] += total_count_data_;
            ONEDAL_ASSERT(local_row_counts_.get_data()[i] >= 0);
            local_data_counts_.get_mutable_data()[i] =
                local_row_counts_.get_data()[i] * column_count_;
            total_count_data_ += local_row_counts_.get_data()[i] * column_count_;
            total_count_weights_ += local_row_counts_.get_data()[i];
        }
        ONEDAL_ASSERT(total_count_data_ == row_count_ * column_count_);
        ONEDAL_ASSERT(total_count_weights_ == row_count_);
    }
    void collect_data(const table& local_data) {
        ONEDAL_ASSERT(local_data.get_row_count() > 0);
        ONEDAL_ASSERT(row_count_ > 0);
        ONEDAL_ASSERT(column_count_ > 0);
        ONEDAL_ASSERT(block_size_ > 0);
        auto arr_local_data =
            pr::table2ndarray<Float>(queue_, local_data, sycl::usm::alloc::device);
        data_ = pr::ndarray<Float, 2>::empty(queue_,
                                             { row_count_, column_count_ },
                                             sycl::usm::alloc::device);
        comm_
            .allgatherv(arr_local_data.flatten(queue_),
                        data_.flatten(queue_),
                        local_data_counts_.get_data(),
                        displs_data_.get_data())
            .wait();
    }
    void collect_weights(const table& local_weights) {
        if (local_weights.get_row_count() == block_size_) {
            ONEDAL_ASSERT(local_weights.get_column_count() == 1);
            ONEDAL_ASSERT(block_size_ > 0);
            ONEDAL_ASSERT(row_count_ > 0);
            auto arr_local_weights =
                pr::table2ndarray<Float>(queue_, local_weights, sycl::usm::alloc::device);
            weights_ =
                pr::ndarray<Float, 2>::empty(queue_, { row_count_, 1 }, sycl::usm::alloc::device);
            comm_
                .allgatherv(arr_local_weights.flatten(queue_),
                            weights_.flatten(queue_),
                            local_row_counts_.get_data(),
                            displs_weights_.get_data())
                .wait();
        }
    }

private:
    std::int64_t block_start_ = 0;
    std::int64_t block_size_ = 0;
    std::int64_t column_count_ = 0;
    std::int64_t row_count_ = 0;
    std::int64_t total_count_data_;
    std::int64_t total_count_weights_;
    array<std::int64_t> local_row_counts_;
    array<std::int64_t> local_data_counts_;
    array<std::int64_t> displs_data_;
    array<std::int64_t> displs_weights_;
    const bk::communicator<spmd::device_memory_access::usm>& comm_;
    sycl::queue queue_;
    pr::ndarray<Float, 2> data_;
    pr::ndarray<Float, 2> weights_;
};

} // namespace oneapi::dal::dbscan::backend
