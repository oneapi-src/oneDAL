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

#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/basic_statistics/backend/gpu/compute_kernel_csr_impl.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::basic_statistics::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

using method_t = method::sparse;
using task_t = task::compute;
using comm_t = bk::communicator<spmd::device_memory_access::usm>;
using input_t = compute_input<task_t, dal::csr_table>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;


template <typename Float>
result_t compute_kernel_csr_impl<Float>::operator()(const bk::context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto queue = ctx.get_queue();
    const csr_table csr_tdata = input.get_data();
    const auto column_count = csr_tdata.get_column_count();
    auto result_options = desc.get_result_options();
    const auto row_count = csr_tdata.get_row_count();
    auto [csr_data, column_indices, row_offsets] = csr_accessor<const Float>(csr_tdata).pull(queue, { 0, -1 }, sparse_indexing::zero_based);
    auto csr_data_ptr = csr_data.get_data();
    auto column_indices_ptr = column_indices.get_data();
    auto row_offsets_ptr = row_offsets.get_data();

    using limits_t = std::numeric_limits<Float>;
    constexpr Float maximum = limits_t::max();

    auto result_data = pr::ndarray<Float, 2>::empty(queue, {res_opt_count_, column_count }, sycl::usm::alloc::device);
    auto result_data_ptr = result_data.get_mutable_data();

    // const auto local_size = bk::device_max_wg_size();
    const auto nd_range = bk::make_multiple_nd_range_2d({column_count, 1}, {1, 1});
    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(nd_range, [=](auto item) {
            std::int32_t col_idx = item.get_global_id(0);
            // Init result arrays
            Float* min = result_data_ptr + 0 * column_count + col_idx;
            Float* max = result_data_ptr + 1 * column_count + col_idx;
            Float* sum = result_data_ptr + 2 * column_count + col_idx;
            Float* sum_squares = result_data_ptr + 3 * column_count + col_idx;
            Float* sum_squares_centered = result_data_ptr + 4 * column_count + col_idx;
            Float* mean = result_data_ptr + 5 * column_count + col_idx;

            min[0] = maximum;
            max[0] = -maximum;
            sum[0] = Float(0);
            sum_squares[0] = Float(0);
            sum_squares_centered[0] = Float(0);
            mean[0] = Float(0);

            for (std::int32_t row_idx = 0; row_idx < row_count; ++row_idx) {
                for (std::int32_t data_idx = row_offsets_ptr[row_idx]; data_idx < row_offsets_ptr[row_idx + 1]; ++data_idx) {
                    if (column_indices_ptr[data_idx] == col_idx) {
                        auto val = csr_data_ptr[data_idx];
                        min[0] = sycl::min<Float>(min[0], val);
                        max[0] = sycl::max<Float>(max[0], val);
                        sum[0] += val;
                        sum_squares[0] += val * val;
                    }
                }
            }
            // In case column is empty need to compare min and max with zero
            min[0] = sycl::min<Float>(min[0], Float(0));
            max[0] = sycl::max<Float>(max[0], Float(0));
            // Compute mean
            const auto mean_val = sum[0] / row_count;
            mean[0] = mean_val;
            // Compute squares
            for (std::int32_t row_idx = 0; row_idx < row_count; ++row_idx) {
                for (std::int32_t data_idx = row_offsets_ptr[row_idx]; data_idx < row_offsets_ptr[row_idx + 1]; ++data_idx) {
                    if (column_indices_ptr[data_idx] == col_idx) {
                        const auto val = csr_data_ptr[data_idx];
                        sum_squares_centered[0] += (val - mean_val) * (val - mean_val);
                    }
                }
            }
        });
    });
    event.wait_and_throw();
    return get_result(queue, result_data, result_options);
}

template class compute_kernel_csr_impl<float>;
template class compute_kernel_csr_impl<double>;

} // namespace oneapi::dal::basic_statistics::backend
#endif // ONEDAL_DATA_PARALLEL