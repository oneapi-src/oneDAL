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

#include <limits>

#include <daal/src/externals/service_rng.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>

#include "oneapi/dal/backend/primitives/selection/row_partitioning.hpp"
#include "oneapi/dal/backend/primitives/selection/select_by_rows_quick.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

constexpr std::uint32_t preffered_sg_size = 16;

// Performs k-selection using Quick Select algorithm which is based on row partitioning
template <typename Float, bool selection_out, bool indices_out>
class quick_selection {
public:
    quick_selection() = delete;
    quick_selection(sycl::queue& queue, const ndshape<2>& shape) {
        rnd_seq_ = array<Float>::empty(queue, max_rnd_seq_size_);
        data_ = ndarray<Float, 2>::empty(queue, shape, sycl::usm::alloc::device);
        indices_ = ndarray<std::int32_t, 2>::empty(queue, shape, sycl::usm::alloc::device);
    }
    ~quick_selection() {
        last_call_.wait_and_throw();
    }
    void init(sycl::queue& queue, std::int64_t rnd_size) {
        ONEDAL_ASSERT(rnd_size > 1);
        this->rnd_seq_size_ = rnd_size < quick_selection::max_rnd_seq_size_
                                  ? rnd_size
                                  : quick_selection::max_rnd_seq_size_;
        auto engine = daal::algorithms::engines::mcg59::Batch<>::create();
        auto engine_impl =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&(*engine));
        ONEDAL_ASSERT(engine_impl != nullptr);

        auto number_array = ndarray<size_t, 1>::empty(queue, this->rnd_seq_size_);
        daal::internal::RNGs<size_t, daal::sse2> rng;
        auto number_ptr = number_array.get_mutable_data();
        rng.uniform((size_t)this->rnd_seq_size_,
                    number_ptr,
                    engine_impl->getState(),
                    0,
                    (size_t)(this->rnd_seq_size_));
        auto values = this->rnd_seq_.get_mutable_data();

        for (std::int64_t i = 0; i < this->rnd_seq_size_; i++) {
            values[i] = static_cast<float>(number_ptr[i]) / (this->rnd_seq_size_ - 1);
        }
    }
    sycl::event select(sycl::queue& queue,
                       const ndview<Float, 2>& data,
                       std::int64_t k,
                       ndview<Float, 2>& selection,
                       ndview<std::int32_t, 2>& indices,
                       const event_vector& deps) {
        last_call_.wait_and_throw();
        ONEDAL_ASSERT(data.get_shape() == data_.get_shape());
        const std::int64_t col_count = data.get_dimension(1);
        const std::int64_t stride = data.get_shape()[1];
        const std::int64_t row_count = data.get_dimension(0);

        auto data_ptr = data.get_data();
        auto data_tmp_ptr = data_.get_mutable_data();
        auto cpy_event = queue.submit([&](sycl::handler& cgh) {
            cgh.memcpy(data_tmp_ptr, data_ptr, sizeof(Float) * row_count * col_count);
        });

        auto indices_tmp_ptr = indices_.get_mutable_data();

        auto rnd_period = this->rnd_seq_size_;
        auto rnd_seq_ptr = rnd_seq_.get_data();

        sycl::range<2> global(preffered_sg_size, row_count);
        sycl::range<2> local(preffered_sg_size, 1);
        sycl::nd_range<2> nd_range2d(global, local);

        Float* selection_ptr = selection_out ? selection.get_mutable_data() : nullptr;
        std::int32_t* indices_ptr = indices_out ? indices.get_mutable_data() : nullptr;

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.depends_on(cpy_event);
            cgh.parallel_for(nd_range2d, [=](sycl::nd_item<2> item) {
                quick_selection::kernel_select(item,
                                               row_count,
                                               data_tmp_ptr,
                                               indices_tmp_ptr,
                                               selection_ptr,
                                               indices_ptr,
                                               rnd_seq_ptr,
                                               rnd_period,
                                               col_count,
                                               k,
                                               stride);
            });
        });
        last_call_ = event;
        return event;
    }

private:
    static Float kernel_get_rnd(const Float* rnd_seq,
                                std::int32_t rnd_period,
                                std::int32_t* count) {
        Float ret = rnd_seq[(*count)++];
        if (*count >= rnd_period)
            *count = 0;
        return ret;
    }
    static void kernel_select(sycl::nd_item<2> item,
                              std::int32_t num_rows,
                              Float* in_values,
                              std::int32_t* in_indices,
                              Float* out_values,
                              std::int32_t* out_indices,
                              const Float* rnd_seq,
                              std::int32_t rnd_period,
                              std::int32_t row_count,
                              std::int32_t k,
                              std::int32_t BlockOffset) {
        auto sg = item.get_sub_group();
        const std::int32_t row_id =
            item.get_global_id(1) * sg.get_group_range()[0] + sg.get_group_id()[0];
        const std::int32_t local_id = sg.get_local_id()[0];
        const std::int32_t local_size = sg.get_local_range()[0];
        if (row_id >= num_rows)
            return;

        const std::int32_t offset_in = row_id * BlockOffset;
        const std::int32_t offset_out = row_id * k;
        std::int32_t partition_start = 0;
        std::int32_t partition_end = row_count;
        std::int32_t rnd_count = 0;

        Float* values = &in_values[offset_in];
        std::int32_t* indices = &in_indices[offset_in];
        if constexpr (indices_out) {
            for (std::int32_t i = partition_start + local_id; i < partition_end; i += local_size) {
                indices[i] = i;
            }
        }
        std::int32_t iteration_count;
        for (iteration_count = 0; iteration_count < row_count; iteration_count++) {
            const Float rnd = quick_selection::kernel_get_rnd(rnd_seq, rnd_period, &rnd_count);
            std::int32_t pos = (std::int32_t)(rnd * (partition_end - partition_start - 1));
            pos = pos < 0 ? 0 : pos;
            const Float pivot = values[partition_start + pos];
            std::int32_t split_index = kernel_row_partitioning(item,
                                                               values,
                                                               indices,
                                                               partition_start,
                                                               partition_end,
                                                               pivot);

            if ((split_index) == k)
                break;
            if (split_index > k)
                partition_end = split_index;
            if (split_index < k)
                partition_start = split_index;
        }
        //assert(iteration_count < row_count);
        for (std::int32_t i = local_id; i < k; i += local_size) {
            if constexpr (selection_out) {
                out_values[offset_out + i] = values[i];
            }
            if constexpr (indices_out) {
                out_indices[offset_out + i] = indices[i];
            }
        }
    }
    static constexpr std::int64_t max_rnd_seq_size_ = 1024;
    std::int64_t rnd_seq_size_ = max_rnd_seq_size_;
    array<Float> rnd_seq_;
    ndarray<Float, 2> data_;
    ndarray<std::int32_t, 2> indices_;
    sycl::event last_call_;
};

template <typename Float, bool selection_out, bool indices_out>
sycl::event select_by_rows_quick(sycl::queue& queue,
                                 const ndview<Float, 2>& data,
                                 std::int64_t k,
                                 ndview<Float, 2>& selection,
                                 ndview<std::int32_t, 2>& indices,
                                 const event_vector& deps = {}) {
    if constexpr (selection_out) {
        ONEDAL_ASSERT(data.get_dimension(1) == selection.get_dimension(1));
        ONEDAL_ASSERT(data.get_dimension(0) >= k);
        ONEDAL_ASSERT(selection.get_dimension(0) == k);
        ONEDAL_ASSERT(selection.has_mutable_data());
    }
    if constexpr (indices_out) {
        ONEDAL_ASSERT(data.get_dimension(1) == indices.get_dimension(1));
        ONEDAL_ASSERT(indices.get_dimension(0) == k);
        ONEDAL_ASSERT(indices.has_mutable_data());
    }
    const std::int64_t nx = data.get_dimension(1);
    quick_selection<Float, selection_out, indices_out> qs(queue, data.get_shape());
    qs.init(queue, nx);
    return qs.select(queue, data, k, selection, indices, deps);
}

#endif

} // namespace oneapi::dal::backend::primitives
