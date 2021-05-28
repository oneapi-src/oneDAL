/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <daal/src/algorithms/kmeans/kmeans_lloyd_kernel.h>

#include "oneapi/dal/algo/kmeans/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

namespace oneapi::dal::kmeans::backend {

using dal::backend::context_cpu;
using input_t = train_input<task::clustering>;
using result_t = train_result<task::clustering>;
using descriptor_t = detail::descriptor_base<task::clustering>;

namespace daal_dm = daal::data_management;
namespace daal_serv = daal::services;
namespace daal_kmeans = daal::algorithms::kmeans;
// namespace daal_kmeans_init = daal::algorithms::kmeans::init;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using daal_kmeans_step1_kernel_t =
    daal_kmeans::internal::KMeansDistributedStep1Kernel<daal_kmeans::lloydDense, Float, Cpu>;

template <typename Float, daal::CpuType Cpu>
using daal_kmeans_step2_kernel_t =
    daal_kmeans::internal::KMeansDistributedStep2Kernel<daal_kmeans::lloydDense, Float, Cpu>;

struct step1_input {
    daal_dm::NumericTablePtr local_data;
    daal_dm::NumericTablePtr global_centroids;
    bool need_final_assignments = false;
};

struct step1_result {
    daal_dm::NumericTablePtr local_observation_count;
    daal_dm::NumericTablePtr local_partial_sums;
    daal_dm::NumericTablePtr local_partial_objective;
    daal_dm::NumericTablePtr local_partial_candidate_distances;
    daal_dm::NumericTablePtr local_partial_candidate_centroids;
    daal_dm::NumericTablePtr local_partial_assignments;
    daal_dm::NumericTablePtr local_assignments;
};

struct step2_input {
    std::vector<step1_result> master_step1_results;
};

struct step2_result {
    daal_dm::NumericTablePtr master_centroids;
    daal_dm::NumericTablePtr master_objective;
};

// class cluster_updater {
// public:
// private:
//     // daal_dm::NumericTablePtr clusters_;
//     // daal_dm::NumericTablePtr labels_;
// };

static step1_result train_step1_local(const context_cpu& ctx,
                                      const descriptor_t& desc,
                                      const step1_input& input) {}

static void train_step2(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {}

static result_t train(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {}

template <typename Float>
struct train_kernel_cpu_spmd<Float, method::lloyd_dense, task::clustering> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        // auto& comm = ctx.get_communicator();

        // comm

        return {};
    }
};

template struct train_kernel_cpu_spmd<float, method::lloyd_dense, task::clustering>;
template struct train_kernel_cpu_spmd<double, method::lloyd_dense, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
