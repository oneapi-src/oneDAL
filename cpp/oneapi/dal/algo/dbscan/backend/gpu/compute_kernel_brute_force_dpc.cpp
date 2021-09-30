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

#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/algo/dbscan/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/dbscan/backend/fill_core_flags.hpp"

#include <daal/src/algorithms/dbscan/oneapi/dbscan_kernel_ucapi.h>

namespace oneapi::dal::dbscan::backend {

using dal::backend::context_gpu;
using dal::backend::to_host_sync;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

namespace daal_dbscan = daal::algorithms::dbscan;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_dbscan_t = daal_dbscan::internal::DBSCANBatchKernelUCAPI<Float>;

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data,
                                 const table& weights) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const int64_t row_count = data.get_row_count();
    const int64_t column_count = data.get_column_count();

    const double epsilon = desc.get_epsilon();
    const int64_t min_observations = desc.get_min_observations();

    daal_dbscan::Parameter par(epsilon, dal::detail::integral_cast<std::size_t>(min_observations));
    par.memorySavingMode = desc.get_mem_save_mode();
    if (desc.get_result_options().test(result_options::core_observation_indices)) {
        par.resultsToCompute = daal_dbscan::computeCoreIndices;
    }
    else if (desc.get_result_options().test(result_options::core_observations)) {
        par.resultsToCompute = daal_dbscan::computeCoreObservations;
    }
    if (desc.get_result_options().test(result_options::core_observations)) {
        par.resultsToCompute |= daal_dbscan::computeCoreObservations;
    }

    const auto daal_data = interop::convert_to_daal_table(queue, data);
    const auto daal_weights = interop::convert_to_daal_table(queue, weights);

    array<std::int32_t> arr_responses =
        array<std::int32_t>::empty(queue, row_count * 1, sycl::usm::alloc::device);
    array<std::int32_t> arr_cluster_count =
        array<std::int32_t>::empty(queue, 1, sycl::usm::alloc::device);

    const auto daal_responses = interop::convert_to_daal_table(queue, arr_responses, row_count, 1);
    const auto daal_cluster_count = interop::convert_to_daal_table(queue, arr_cluster_count, 1, 1);

    /* Tables for core observation indices and core observations are allocated inside the kernel */
    const auto daal_core_observation_indices = interop::empty_daal_homogen_table<std::int32_t>(1);
    const auto daal_core_observations = interop::empty_daal_homogen_table<Float>(column_count);

    interop::status_to_exception(daal_dbscan_t<Float>{}.compute(daal_data.get(),
                                                                daal_weights.get(),
                                                                daal_responses.get(),
                                                                daal_cluster_count.get(),
                                                                daal_core_observation_indices.get(),
                                                                daal_core_observations.get(),
                                                                &par));
    auto core_observation_indices =
        interop::convert_from_daal_homogen_table<std::int32_t>(daal_core_observation_indices);
    auto core_observations =
        interop::convert_from_daal_homogen_table<Float>(daal_core_observations);

    auto results = result_t()
                       .set_cluster_count(to_host_sync(arr_cluster_count)[0])
                       .set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        results.set_responses(dal::homogen_table::wrap(arr_responses, row_count, 1));
    }
    if (desc.get_result_options().test(result_options::core_flags)) {
        auto arr_core_flags = fill_core_flags(core_observation_indices, row_count);
        results.set_core_flags(dal::homogen_table::wrap(arr_core_flags, row_count, 1));
    }

    if (desc.get_result_options().test(result_options::core_observation_indices)) {
        results.set_core_observation_indices(core_observation_indices);
    }

    if (desc.get_result_options().test(result_options::core_observations)) {
        results.set_core_observations(core_observations);
    }

    return results;
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_weights());
}

template <typename Float>
struct compute_kernel_gpu<Float, method::brute_force, task::clustering> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::brute_force, task::clustering>;
template struct compute_kernel_gpu<double, method::brute_force, task::clustering>;

} // namespace oneapi::dal::dbscan::backend
