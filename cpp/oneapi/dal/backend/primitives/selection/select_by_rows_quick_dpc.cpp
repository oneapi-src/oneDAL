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

#include <limits>

#include <daal/src/externals/service_rng.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>

#include "oneapi/dal/backend/primitives/selection/row_partitioning.hpp"
#include "oneapi/dal/backend/primitives/selection/select_by_rows_quick.hpp"

namespace oneapi::dal::backend::primitives {

constexpr uint32_t preffered_sg_size = 16;

template <typename Float, bool selection_out, bool indices_out>
class quick_selection {
public:
    quick_selection() = delete;
    quick_selection(sycl::queue& queue) {
        rng_seq_ = ndarray<Float, 1>::empty(queue, max_rng_seq_size_);
    }
    void init(sycl::queue& queue, std::int64_t rng_size) {
        ONEDAL_ASSERT(rng_size > 1);
        this->rng_seq_size_ = rng_size < quick_selection::max_rng_seq_size_
                                  ? rng_size
                                  : quick_selection::max_rng_seq_size_;
        auto engine = daal::algorithms::engines::mcg59::Batch<>::create();
        auto engine_impl =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&(*engine));
        ONEDAL_ASSERT(engine_impl != nullptr);

        auto number_array = ndarray<size_t, 1>::empty(queue, this->rng_seq_size_);
        daal::internal::RNGs<size_t, daal::sse2> rng;
        auto number_ptr = number_array.get_mutable_data();
        rng.uniform((size_t)this->rng_seq_size_,
                    number_ptr,
                    engine_impl->getState(),
                    0,
                    (size_t)(this->rng_seq_size_));
        auto values = this->rng_seq_.get_mutable_data();

        for (std::int64_t i = 0; i < this->rng_seq_size_; i++) {
            values[i] = static_cast<float>(number_ptr[i]) / (this->rng_seq_size_ - 1);
        }
    }
    sycl::event select(sycl::queue& queue,
                       const ndview<Float, 2>& data,
                       std::int64_t k,
                       std::int64_t col_begin,
                       std::int64_t col_end,
                       ndview<Float, 2>& selection,
                       ndview<int, 2>& indices,
                       const event_vector& deps) {
        const std::int64_t col_count = data.get_dimension(1);
        const std::int64_t row_count = data.get_dimension(0);
        auto indices_tmp = ndarray<int, 2>::empty(queue, { row_count, col_count });
        auto data_ptr = data.get_data();

        auto data_tmp = ndarray<Float, 2>::empty(queue, { row_count, col_count });
        auto data_tmp_ptr = data_tmp.get_mutable_data();
        auto cpy_event = queue.submit([&](sycl::handler& cgh) {
            cgh.memcpy(data_tmp_ptr, data_ptr, sizeof(Float) * row_count * col_count);
        });
        cpy_event.wait();

        sycl::range<2> global(preffered_sg_size, row_count);
        sycl::range<2> local(preffered_sg_size, 1);
        sycl::nd_range<2> nd_range2d(global, local);

        Float* selection_ptr = selection_out ? selection.get_mutable_data() : nullptr;
        int* indices_ptr = indices_out ? indices.get_mutable_data() : nullptr;
        int* indices_tmp_ptr = indices_tmp.get_mutable_data();
        auto rng_seq_ptr = this->rng_seq_.get_data();
        auto rng_period = this->rng_seq_size_;

        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            sycl::stream out(1024 * 8, 1024 * 8, cgh);
            cgh.parallel_for(nd_range2d, [=](sycl::nd_item<2> item) {
                quick_selection::kernel_select(out, item,
                                               row_count,
                                               data_tmp_ptr,
                                               indices_tmp_ptr,
                                               selection_ptr,
                                               indices_ptr,
                                               rng_seq_ptr,
                                               rng_period,
                                               col_count,
                                               col_count,
                                               k,
                                               col_count);
            });
        });
        return event;
    }

private:
    static Float kernel_get_rnd(const Float* rnd_seq, int rnd_period, int* count) {
        float ret = rnd_seq[(*count)++];
        if (*count >= rnd_period)
            *count = 0;
        return ret;
    }
    static void kernel_select(const sycl::stream& out, sycl::nd_item<2> item,
                              int num_rows,
                              Float* in_values,
                              int* in_indices,
                              Float* out_values,
                              int* out_indices,
                              const Float* rnd_seq,
                              int RndPeriod,
                              int N,
                              int NLast,
                              int K,
                              int BlockOffset) {
        
        auto sg = item.get_sub_group();
        const int row_id = item.get_global_id(1) * sg.get_group_range()[0] + sg.get_group_id()[0];
        const int local_id = sg.get_local_id()[0];
        const int local_size = sg.get_local_range()[0];
        if (row_id >= num_rows)
            return;

        N = (row_id == num_rows - 1) ? NLast : N;
//        if(local_id == 0)
//            out << "N: " << N << " NLast: " << NLast << sycl::endl;

        const int offset_in = row_id * BlockOffset;
        const int offset_out = row_id * K;
        int partition_start = 0;
        int partition_end = N;
        int rnd_count = 0;

        Float* values = &in_values[offset_in];
        int* indices = &in_indices[offset_in];
        if constexpr (indices_out) {
            for (int i = partition_start + local_id; i < partition_end; i += local_size) {
                indices[i] = i;
            }
        }
        int iteration_count;
        for (iteration_count = 0; iteration_count < N; iteration_count++) {
            const Float rnd = quick_selection::kernel_get_rnd(rnd_seq, RndPeriod, &rnd_count);
            int pos = (int)(rnd * (partition_end - partition_start - 1));
            pos = pos < 0 ? 0 : pos;
            const Float pivot = values[partition_start + pos];
/*            if(local_id == 0)
                out << "Pivot: " << (partition_start + pos) << " " <<pivot << sycl::endl;
            if(local_id == 0)
                out << "Partition: " << partition_start << " " << partition_end << sycl::endl;
*/
            int split_index = kernel_row_partitioning(item,
                                                        values,
                                                        indices,
                                                        partition_start,
                                                        partition_end,
                                                        pivot);

            if ((split_index) == K /*|| (!split_index && !num_of_great)*/)
                break;
            if (split_index > K)
                partition_end = split_index;
            if (split_index < K)
                partition_start = split_index;
/*            if(local_id == 0) {
                out << "Iteration: " << iteration_count << " split index: " << split_index 
                    << " partition_start: " << partition_start << " partition_end: " << partition_end << sycl::endl;
                for (int i = 0; i < N; i++)        
                    out << "C: " << i << values[i] << sycl::endl;
            }*/
        }
        //assert(iteration_count < N);
        for (int i = local_id; i < K; i += local_size) {
            if constexpr (selection_out) {
                out_values[offset_out + i] = values[i];
            }
            if constexpr (indices_out) {
                out_indices[offset_out + i] = indices[i];
            }
        }
    }
    static constexpr std::int64_t max_rng_seq_size_ = 1024;
    std::int64_t rng_seq_size_ = max_rng_seq_size_;
    ndarray<Float, 1> rng_seq_;
};

template <typename Float, bool selection_out, bool indices_out>
sycl::event select_by_rows_quick(sycl::queue& queue,
                                 const ndview<Float, 2>& data,
                                 std::int64_t k,
                                 std::int64_t col_begin,
                                 std::int64_t col_end,
                                 ndview<Float, 2>& selection,
                                 ndview<int, 2>& indices,
                                 const event_vector& deps) {
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
    std::cout << "Quick selection begin" << std::endl;
    quick_selection<Float, selection_out, indices_out> qs(queue);
    std::cout << "Quick selection init" << std::endl;
    qs.init(queue, nx);
    std::cout << "Quick selection run" << std::endl;
    return qs.select(queue, data, k, col_begin, col_end, selection, indices, deps);
    std::cout << "Quick selection done" << std::endl;
}

#define INSTANTIATE(F, selection_out, indices_out)                                          \
    template ONEDAL_EXPORT sycl::event select_by_rows_quick<F, selection_out, indices_out>( \
        sycl::queue & queue,                                                                \
        const ndview<F, 2>& block,                                                          \
        std::int64_t k,                                                                     \
        std::int64_t col_begin,                                                             \
        std::int64_t col_end,                                                               \
        ndview<F, 2>& selection,                                                            \
        ndview<int, 2>& indices,                                                            \
        const event_vector& deps);

#define INSTANTIATE_FLOAT(selection_out, indices_out) \
    INSTANTIATE(float, selection_out, indices_out)    \
    INSTANTIATE(double, selection_out, indices_out)

INSTANTIATE_FLOAT(true, false)
INSTANTIATE_FLOAT(false, true)
INSTANTIATE_FLOAT(true, true)

} // namespace oneapi::dal::backend::primitives
