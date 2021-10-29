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

#include <CL/sycl.hpp>

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/algo/dbscan/backend/gpu/kernels_fp.hpp"

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::dbscan::backend {

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;

template <typename Float>
inline auto make_results(sycl::queue& queue,
                         const descriptor_t& desc,
                         const pr::ndarray<Float, 2> data,
                         const pr::ndarray<std::int32_t, 1> responses,
                         const pr::ndarray<std::int32_t, 1> cores,

                         std::int64_t cluster_count,
                         std::int64_t core_count = -1) {
    const std::int64_t column_count = data.get_dimension(1);
    ONEDAL_ASSERT(column_count > 0);
    const std::int64_t block_size = cores.get_dimension(0);
    ONEDAL_ASSERT(block_size == responses.get_dimension(0));
    auto results =
        result_t().set_cluster_count(cluster_count).set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        results.set_responses(dal::homogen_table::wrap(responses.flatten(queue), block_size, 1));
    }
    if (desc.get_result_options().test(result_options::core_flags)) {
        results.set_core_flags(dal::homogen_table::wrap(cores.flatten(queue), block_size, 1));
    }

    bool return_core_indices =
        desc.get_result_options().test(result_options::core_observation_indices);
    bool return_core_observations =
        desc.get_result_options().test(result_options::core_observations);

    if (return_core_indices || return_core_observations) {
        if (core_count == -1) {
            core_count = count_cores(queue, cores);
        }
        ONEDAL_ASSERT(block_size >= core_count);
        if (core_count == 0) {
            if (return_core_indices) {
                results.set_core_observation_indices(dal::homogen_table{});
            }
            if (return_core_observations) {
                results.set_core_observations(dal::homogen_table{});
            }
        }
        if (return_core_indices) {
            auto host_indices = array<std::int32_t>::empty(core_count);
            auto host_indices_ptr = host_indices.get_mutable_data();
            std::int64_t pos = 0;
            auto host_cores = cores.to_host(queue);
            auto host_cores_ptr = host_cores.get_data();
            for (std::int64_t i = 0; i < block_size; i++) {
                if (host_cores_ptr[i] > 0) {
                    ONEDAL_ASSERT(pos < core_count);
                    host_indices_ptr[pos] = i;
                    pos++;
                }
            }
            auto device_indices =
                pr::ndarray<std::int32_t, 1>::empty(queue, core_count, sycl::usm::alloc::device);
            dal::detail::memcpy_host2usm(queue,
                                         device_indices.get_mutable_data(),
                                         host_indices_ptr,
                                         core_count * sizeof(std::int32_t));
            results.set_core_observation_indices(
                dal::homogen_table::wrap(device_indices.flatten(queue), core_count, 1));
        }
        if (return_core_observations) {
            auto observations = pr::ndarray<Float, 1>::empty(queue, core_count * column_count);
            auto observations_ptr = observations.get_mutable_data();
            std::int64_t pos = 0;
            auto host_cores = cores.to_host(queue);
            auto host_cores_ptr = host_cores.get_data();
            for (std::int64_t i = 0; i < block_size; i++) {
                if (host_cores_ptr[i] > 0) {
                    ONEDAL_ASSERT(pos < core_count * column_count);
                    bk::memcpy(queue,
                               observations_ptr + pos * column_count,
                               data.get_data() + i * column_count,
                               std::size_t(column_count) * sizeof(Float));
                    pos += column_count;
                }
            }
            results.set_core_observations(
                dal::homogen_table::wrap(observations.flatten(queue), core_count, column_count));
        }
    }
    return results;
}

} // namespace oneapi::dal::dbscan::backend
