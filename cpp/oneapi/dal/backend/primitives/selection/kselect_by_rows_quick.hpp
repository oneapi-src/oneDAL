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
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_base.hpp"
#include "oneapi/dal/backend/primitives/rng/rnd_seq.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

// Performs k-selection using Quick Select algorithm which is based on row partitioning
template <typename Float>
class kselect_by_rows_quick : public kselect_by_rows_base<Float> {
public:
    kselect_by_rows_quick() = delete;
    kselect_by_rows_quick(sycl::queue& queue, const ndshape<2>& shape)
            : rnd_seq_(queue, std::min(shape[1], max_rnd_seq_size_)) {
        data_ = ndarray<Float, 2>::empty(queue, shape, sycl::usm::alloc::device);
        indices_ = ndarray<std::int32_t, 2>::empty(queue, shape, sycl::usm::alloc::device);
    }
    ~kselect_by_rows_quick() {
        last_call_.wait_and_throw();
    }
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<Float, 2>& selection,
                           ndview<std::int32_t, 2>& indices,
                           const event_vector& deps) override {
        return select<true, true>(queue, data, k, selection, indices, deps);
    }
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<Float, 2>& selection,
                           const event_vector& deps) override {
        ndarray<std::int32_t, 2> dummy;
        return select<true, false>(queue, data, k, selection, dummy, deps);
    }
    sycl::event operator()(sycl::queue& queue,
                           const ndview<Float, 2>& data,
                           std::int64_t k,
                           ndview<std::int32_t, 2>& indices,
                           const event_vector& deps) override {
        ndarray<Float, 2> dummy;
        return select<false, true>(queue, data, k, dummy, indices, deps);
    }

private:
    template <bool selection_out, bool indices_out>
    sycl::event select(sycl::queue& queue,
                       const ndview<Float, 2>& data,
                       std::int64_t k,
                       ndview<Float, 2>& selection,
                       ndview<std::int32_t, 2>& indices,
                       const event_vector& deps) {
        if (indices_out) {
            ONEDAL_ASSERT(indices.get_shape()[0] == data.get_shape()[0]);
            ONEDAL_ASSERT(indices.get_shape()[1] == k);
        }
        if (selection_out) {
            ONEDAL_ASSERT(selection.get_shape()[0] == data.get_shape()[0]);
            ONEDAL_ASSERT(selection.get_shape()[1] == k);
        }
        ONEDAL_ASSERT(data.get_shape() == data_.get_shape());
        last_call_.wait_and_throw();
        const std::int64_t col_count = data.get_dimension(1);
        const std::int64_t stride = data.get_shape()[1];
        const std::int64_t row_count = data.get_dimension(0);

        auto data_ptr = data.get_data();
        auto data_tmp_ptr = data_.get_mutable_data();
        auto cpy_event = queue.submit([&](sycl::handler& cgh) {
            cgh.memcpy(data_tmp_ptr, data_ptr, sizeof(Float) * row_count * col_count);
        });

        auto indices_tmp_ptr = indices_.get_mutable_data();

        auto rnd_period = this->rnd_seq_.get_count();
        auto rnd_seq_ptr = this->rnd_seq_.get_data();

        [[maybe_unused]] Float* selection_ptr =
            selection_out ? selection.get_mutable_data() : nullptr;
        [[maybe_unused]] std::int32_t* indices_ptr =
            indices_out ? indices.get_mutable_data() : nullptr;

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.depends_on(cpy_event);
            cgh.parallel_for(make_multiple_nd_range_2d({ preffered_sg_size, row_count },
                                                       { preffered_sg_size, 1 }),
                             [=](sycl::nd_item<2> item) {
                                 kselect_by_rows_quick::kernel_select<selection_out, indices_out>(
                                     item,
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

    static Float kernel_get_rnd(const Float* rnd_seq,
                                std::int32_t rnd_period,
                                std::int32_t* count) {
        Float ret = rnd_seq[(*count)++];
        if (*count >= rnd_period)
            *count = 0;
        return ret;
    }
    template <bool selection_out, bool indices_out>
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
            const Float rnd =
                kselect_by_rows_quick::kernel_get_rnd(rnd_seq, rnd_period, &rnd_count);
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
    static constexpr std::uint32_t preffered_sg_size = 16;
    static constexpr std::int64_t max_rnd_seq_size_ = 1024;
    std::int64_t rnd_seq_size_ = max_rnd_seq_size_;
    rnd_seq<Float> rnd_seq_;
    ndarray<Float, 2> data_;
    ndarray<std::int32_t, 2> indices_;
    sycl::event last_call_;
};

#endif

} // namespace oneapi::dal::backend::primitives
