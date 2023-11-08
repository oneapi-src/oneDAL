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
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = detail::descriptor_base<task_t>;

template <typename Float>
void compute_kernel_csr_impl<Float>::finalize_for_distr(sycl::queue& q,
                                                        comm_t& communicator,
                                                        pr::ndarray<Float, 2>& results,
                                                        const input_t& input,
                                                        const std::vector<sycl::event>& deps) {
    auto result_ptr = results.get_mutable_data();
    const csr_table csr_tdata = static_cast<const csr_table&>(input.get_data());
    auto [csr_data, column_indices, row_offsets] =
        csr_accessor<const Float>(csr_tdata).pull(q, { 0, -1 }, sparse_indexing::zero_based);
    const auto column_count = csr_tdata.get_column_count();
    const auto row_count = csr_tdata.get_row_count();
    const auto nonzero_count = csr_tdata.get_non_zero_count();

    auto host_results = results.flatten(q, deps);
    communicator
        .allreduce(host_results.get_slice(stat::min * column_count, column_count),
                   spmd::reduce_op::min)
        .wait();
    communicator
        .allreduce(host_results.get_slice(stat::max * column_count, column_count),
                   spmd::reduce_op::max)
        .wait();
    communicator
        .allreduce(host_results.get_slice(stat::sum * column_count, column_count),
                   spmd::reduce_op::sum)
        .wait();
    communicator
        .allreduce(host_results.get_slice(stat::sum2 * column_count, column_count),
                   spmd::reduce_op::sum)
        .wait();
    communicator
        .allreduce(host_results.get_slice(stat::moment2 * column_count, column_count),
                   spmd::reduce_op::sum)
        .wait();

    results.assign_from_host(q, host_results.get_data(), res_opt_count_ * column_count)
        .wait_and_throw();

    auto csr_data_ptr = csr_data.get_data();
    auto column_indices_ptr = column_indices.get_data();
    auto distr_range = sycl::range<1>(column_count);
    auto calc_s2c_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(distr_range, [=](std::int64_t col_idx) {
            auto mean_val = result_ptr[stat::sum * column_count + col_idx] / row_count;
            result_ptr[stat::mean * column_count + col_idx] = mean_val;
            Float sum2_cent = Float(0);
            std::int32_t nnz_row_count = 0;
            for (std::int32_t data_idx = 0; data_idx < nonzero_count; ++data_idx) {
                if (col_idx == column_indices_ptr[data_idx]) {
                    auto val = csr_data_ptr[data_idx];
                    sum2_cent += (val - mean_val) * (val - mean_val);
                    nnz_row_count += 1;
                }
            }
            // For zero values sum2_cent is just square of mean value
            sum2_cent += (row_count - nnz_row_count) * mean_val * mean_val;

            result_ptr[stat::sum2_cent * column_count + col_idx] = sum2_cent;
        });
    });

    host_results = results.flatten(q, { calc_s2c_event });
    communicator
        .allreduce(host_results.get_slice(stat::sum2_cent * column_count, column_count),
                   spmd::reduce_op::sum)
        .wait();
    auto allreduce_event =
        results.assign_from_host(q, host_results.get_data(), res_opt_count_ * column_count);

    auto final_event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on({ allreduce_event });
        cgh.parallel_for(distr_range, [=](std::int64_t col_idx) {
            auto mean_val = result_ptr[stat::mean * column_count + col_idx];
            result_ptr[stat::variance * column_count + col_idx] =
                result_ptr[stat::sum2_cent * column_count + col_idx] / (row_count - 1);
            result_ptr[stat::stddev * column_count + col_idx] =
                sycl::sqrt(result_ptr[stat::variance * column_count + col_idx]);
            result_ptr[stat::variation * column_count + col_idx] =
                result_ptr[stat::stddev * column_count + col_idx] / mean_val;
        });
    });

    final_event.wait_and_throw();
}

template <typename Float>
result_t compute_kernel_csr_impl<Float>::operator()(const bk::context_gpu& ctx,
                                                    const descriptor_t& desc,
                                                    const input_t& input) {
    auto queue = ctx.get_queue();
    const auto table = input.get_data();
    ONEDAL_ASSERT(table.get_kind() == csr_table::kind());
    const csr_table csr_tdata = static_cast<const csr_table&>(table);
    comm_t comm = ctx.get_communicator();
    const bool distr_mode = comm.get_rank_count() > 1;
    const auto column_count = csr_tdata.get_column_count();
    const auto row_count = csr_tdata.get_row_count();
    auto result_options = desc.get_result_options();
    const auto nonzero_count = csr_tdata.get_non_zero_count();
    auto [csr_data, column_indices, row_offsets] =
        csr_accessor<const Float>(csr_tdata).pull(queue,
                                                  { 0, -1 },
                                                  sparse_indexing::zero_based,
                                                  sycl::usm::alloc::device);
    auto csr_data_ptr = csr_data.get_data();
    auto column_indices_ptr = column_indices.get_data();

    using limits_t = std::numeric_limits<Float>;
    constexpr Float maximum = limits_t::max();

    auto result_data = pr::ndarray<Float, 2>::empty(queue,
                                                    { res_opt_count_, column_count },
                                                    sycl::usm::alloc::device);
    auto result_data_ptr = result_data.get_mutable_data();

    const auto local_size =
        bk::device_max_wg_size(queue); // number of columns processed in one group

    const auto nd_range = bk::make_multiple_nd_range_1d(local_size, local_size);
    auto event = queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<Float, 1> local_res_buf(local_size * res_opt_count_, cgh);
        sycl::local_accessor<std::int32_t, 1> local_hist_buf(local_size, cgh);
        cgh.parallel_for(nd_range, [=](auto item) {
            auto local_id = item.get_local_id();
            Float* work_group_buf =
                local_res_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            std::int32_t* hist_buf =
                local_hist_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
            for (std::int64_t col_ofs = 0; col_ofs < column_count; col_ofs += local_size) {
                auto local_buf = work_group_buf + local_id * res_opt_count_;
                local_buf[stat::min] = maximum;
                local_buf[stat::max] = -maximum;
                local_buf[stat::sum] = Float(0);
                local_buf[stat::sum2] = Float(0);
                local_buf[stat::sum2_cent] = Float(0);
                hist_buf[local_id] = 0;
                for (std::int64_t idx = local_id; idx < nonzero_count; idx += local_size) {
                    auto col_idx = column_indices_ptr[idx] - col_ofs;
                    auto val = csr_data_ptr[idx];
                    if (col_idx >= 0 && col_idx < local_size) {
                        sycl::atomic_ref<Float,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_device,
                                         sycl::access::address_space::local_space>
                            col_min(local_res_buf[col_idx * res_opt_count_ + stat::min]);
                        col_min.fetch_min(val);
                        sycl::atomic_ref<Float,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_device,
                                         sycl::access::address_space::local_space>
                            col_max(local_res_buf[col_idx * res_opt_count_ + stat::max]);
                        col_max.fetch_max(val);
                        sycl::atomic_ref<Float,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_device,
                                         sycl::access::address_space::local_space>
                            col_sum(local_res_buf[col_idx * res_opt_count_ + stat::sum]);
                        col_sum.fetch_add(val);
                        sycl::atomic_ref<Float,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_device,
                                         sycl::access::address_space::local_space>
                            col_sum2(local_res_buf[col_idx * res_opt_count_ + stat::sum2]);
                        col_sum2.fetch_add(val * val);
                        sycl::atomic_ref<std::int32_t,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_device,
                                         sycl::access::address_space::local_space>
                            hist_buf(local_hist_buf[col_idx]);
                        hist_buf.fetch_add(1);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                local_buf[stat::mean] = local_buf[stat::sum] / row_count;
                for (std::int64_t idx = local_id; idx < nonzero_count; idx += local_size) {
                    auto col_idx = column_indices_ptr[idx] - col_ofs;
                    auto val = csr_data_ptr[idx];
                    if (col_idx >= 0 && col_idx < local_size) {
                        sycl::atomic_ref<Float,
                                         sycl::memory_order_relaxed,
                                         sycl::memory_scope_device,
                                         sycl::access::address_space::local_space>
                            col_sum2_cent(
                                local_res_buf[col_idx * res_opt_count_ + stat::sum2_cent]);
                        auto mean_val = work_group_buf[col_idx * res_opt_count_ + stat::mean];
                        col_sum2_cent.fetch_add((val - mean_val) * (val - mean_val));
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
                if ((local_id + col_ofs) >= column_count) {
                    continue;
                }
                auto mean_val = local_buf[stat::mean];
                if (hist_buf[local_id] != row_count) {
                    local_buf[stat::min] = sycl::min<Float>(0, local_buf[stat::min]);
                    local_buf[stat::max] = sycl::max<Float>(0, local_buf[stat::max]);
                    local_buf[stat::sum2_cent] +=
                        Float(row_count - hist_buf[local_id]) * mean_val * mean_val;
                }
                result_data_ptr[stat::min * column_count + col_ofs + local_id] =
                    local_buf[stat::min];
                result_data_ptr[stat::max * column_count + col_ofs + local_id] =
                    local_buf[stat::max];
                result_data_ptr[stat::sum * column_count + col_ofs + local_id] =
                    local_buf[stat::sum];
                result_data_ptr[stat::sum2 * column_count + col_ofs + local_id] =
                    local_buf[stat::sum2];
                result_data_ptr[stat::mean * column_count + col_ofs + local_id] = mean_val;
                result_data_ptr[stat::moment2 * column_count + col_ofs + local_id] =
                    local_buf[stat::sum2] / row_count;
                result_data_ptr[stat::sum2_cent * column_count + col_ofs + local_id] =
                    local_buf[stat::sum2_cent];
                result_data_ptr[stat::variance * column_count + col_ofs + local_id] =
                    local_buf[stat::sum2_cent] / (row_count - 1);
                result_data_ptr[stat::stddev * column_count + col_ofs + local_id] =
                    sycl::sqrt(result_data_ptr[stat::variance * column_count + col_ofs + local_id]);
                result_data_ptr[stat::variation * column_count + col_ofs + local_id] =
                    result_data_ptr[stat::stddev * column_count + col_ofs + local_id] / mean_val;
            }
        });
    });

    event.wait_and_throw();
    if (distr_mode) {
        finalize_for_distr(queue, comm, result_data, input, { event });
    }
    return get_result(queue, result_data, result_options);
}

template class compute_kernel_csr_impl<float>;
template class compute_kernel_csr_impl<double>;

} // namespace oneapi::dal::basic_statistics::backend
#endif // ONEDAL_DATA_PARALLEL
