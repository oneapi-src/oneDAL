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

#pragma once

#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_csr_impl.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::basic_statistics::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Float>
result_t compute_kernel_csr_impl::operator()(const bk::context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto queue = ctx.get_queue();
    auto csr_table = input.get_data();
    const auto column_count = csr_table.get_column_count();
    const auto row_count = csr_table.get_row_count();
    auto [csr_data, column_indices, row_offsets] = csr_accessor<Float>(csr_table).pull(queue, { 0, -1 }, sparse_indexing::zero_based);

    using limits_t = std::numeric_limits<Float>;
    constexpr Float maximum = limits_t::max();

    auto min_arr = ndarray<Float, 2>::empty(queue, { 1, column_count }, sycl::usm::alloc::device);
    auto max_arr = ndarray<Float, 2>::empty(queue, { 1, column_count }, sycl::usm::alloc::device);
    auto sum_arr = ndarray<Float, 2>::empty(queue, { 1, column_count }, sycl::usm::alloc::device);
    auto mean_arr = ndarray<Float, 2>::empty(queue, { 1, column_count }, sycl::usm::alloc::device);
    auto s2cent_arr = ndarray<Float, 2>::empty(queue, { 1, column_count }, sycl::usm::alloc::device);
    
    auto min_arr_ptr = min_arr.get_mutable_data();
    auto max_arr_ptr = max_arr.get_mutable_data();
    auto sum_arr_ptr = sum_arr.get_mutable_data();
    auto mean_arr_ptr = mean_arr.get_mutable_data();
    auto s2cent_arr_ptr = s2cent_arr.get_mutable_data();

    // const auto local_size = bk::device_max_wg_size();
    const auto nd_range = bk::make_multiple_nd_range_2d({column_count, 1}, {1, 1});
    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](auto item) {
            auto col_idx = item.get_global_id(0);
            // Init result arrays
            min_arr_ptr[col_idx] = maximum;
            max_arr_ptr[col_idx] = -maximum;
            sum_arr_ptr[col_idx] = Float(0);
            mean_arr_ptr[col_idx] = Float(0);
            s2cent_arr_ptr[col_idx] = Float(0);

            for (std::int32_t row_idx = 0; row_idx < row_count; ++row_idx) {
                for (std::int32_t data_idx = row_offsets[row_idx]; data_idx < row_offsets[row_idx + 1]; ++data_idx) {
                    if (column_indices[data_idx] == col_idx) {
                        min_arr_ptr[col_idx] = sycl::min<Float>(min_arr_ptr[col_idx], csr_data[data_idx]);
                        max_arr_ptr[col_idx] = sycl::max<Float>(max_arr_ptr[col_idx], csr_data[data_idx]);
                        sum_arr_ptr[col_idx] += csr_data[data_idx];
                    }
                }
            }
            // In case column is empty need to compare min and max with zero
            min_arr_ptr[col_idx] = sycl::min<Float>(min_arr_ptr[col_idx], Float(0));
            max_arr_ptr[col_idx] = sycl::max<Float>(max_arr_ptr[col_idx], Float(0));
            // Compute mean
            const auto mean = sum_arr_ptr[col_idx] / row_count;
            mean_arr_ptr[col_idx] = mean;
            // Compute squares
            for (std::int32_t row_idx = 0; row_idx < row_count; ++row_idx) {
                for (std::int32_t data_idx = row_offsets[row_idx]; data_idx < row_offsets[row_idx + 1]; ++data_idx) {
                    if (column_indices[data_idx] == col_idx) {
                        const auto val = csr_data[data_idx];
                        s2cent_arr_ptr[col_idx] += (val - mean) * (val - mean);
                    }
                }
            }
        });
    });
}

} // namespace oneapi::dal::basic_statistics::backend
#endif // ONEDAL_DATA_PARALLEL