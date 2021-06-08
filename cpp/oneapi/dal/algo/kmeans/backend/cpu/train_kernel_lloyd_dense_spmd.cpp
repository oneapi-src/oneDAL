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
#include "oneapi/dal/backend/interop/archive.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

namespace oneapi::dal::kmeans::backend {

using dal::backend::context_cpu;
using model_t = model<task::clustering>;
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

template <typename Float, daal::CpuType Cpu>
using daal_kmeans_step3_kernel_t =
    daal_kmeans::internal::KMeansBatchKernel<daal_kmeans::lloydDense, Float, Cpu>;

template <typename Float>
inline void allocate_daal_nt(daal_dm::NumericTablePtr& table,
                             std::int64_t row_count,
                             std::int64_t column_count) {
    table = interop::allocate_daal_homogen_table<Float>(row_count, column_count);
}

struct step1_input {
    step1_input& set_local_data(const daal_dm::NumericTablePtr& value) {
        local_data = value;
        return *this;
    }

    step1_input& set_global_centroids(const daal_dm::NumericTablePtr& value) {
        global_centroids = value;
        return *this;
    }

    daal_dm::NumericTablePtr local_data;
    daal_dm::NumericTablePtr global_centroids;
};

struct step1_result {
    template <typename Float>
    void allocate(std::int64_t cluster_count, std::int64_t feature_count) {
        allocate_daal_nt<Float>(local_observation_count, cluster_count, 1);
        allocate_daal_nt<Float>(local_partial_sums, cluster_count, feature_count);
        allocate_daal_nt<Float>(local_partial_objective, 1, 1);
        allocate_daal_nt<Float>(local_partial_candidate_distances, cluster_count, 1);
        allocate_daal_nt<Float>(local_partial_candidate_centroids, cluster_count, feature_count);
    }

    daal_dm::NumericTablePtr local_observation_count;
    daal_dm::NumericTablePtr local_partial_sums;
    daal_dm::NumericTablePtr local_partial_objective;
    daal_dm::NumericTablePtr local_partial_candidate_distances;
    daal_dm::NumericTablePtr local_partial_candidate_centroids;

    void serialize(dal::detail::output_archive& ar) const {
        interop::daal_output_data_archive daal_ar{ ar };

        using nt_ptr_t = daal_dm::NumericTablePtr;
        daal_ar.setSharedPtrObj(const_cast<nt_ptr_t&>(local_observation_count));
        daal_ar.setSharedPtrObj(const_cast<nt_ptr_t&>(local_partial_sums));
        daal_ar.setSharedPtrObj(const_cast<nt_ptr_t&>(local_partial_objective));
        daal_ar.setSharedPtrObj(const_cast<nt_ptr_t&>(local_partial_candidate_distances));
        daal_ar.setSharedPtrObj(const_cast<nt_ptr_t&>(local_partial_candidate_centroids));
    }

    void deserialize(dal::detail::input_archive& ar) {
        interop::daal_input_data_archive daal_ar{ ar };
        daal_ar.setSharedPtrObj(local_observation_count);
        daal_ar.setSharedPtrObj(local_partial_sums);
        daal_ar.setSharedPtrObj(local_partial_objective);
        daal_ar.setSharedPtrObj(local_partial_candidate_distances);
        daal_ar.setSharedPtrObj(local_partial_candidate_centroids);
    }
};

struct step2_input {
    step2_input& set_master_step1_results(const std::vector<step1_result>& value) {
        master_step1_results = value;
        return *this;
    }

    std::vector<step1_result> master_step1_results;
};

struct step2_result {
    template <typename Float>
    void allocate(std::int64_t cluster_count, std::int64_t feature_count) {
        allocate_daal_nt<Float>(local_observation_count, cluster_count, 1);
        allocate_daal_nt<Float>(local_partial_sums, cluster_count, feature_count);
        allocate_daal_nt<Float>(local_partial_objective, 1, 1);
        allocate_daal_nt<Float>(local_partial_candidate_distances, cluster_count, 1);
        allocate_daal_nt<Float>(local_partial_candidate_centroids, cluster_count, feature_count);
        allocate_daal_nt<Float>(master_centroids, cluster_count, feature_count);
        allocate_daal_nt<Float>(master_objective, 1, 1);
    }

    daal_dm::NumericTablePtr local_observation_count;
    daal_dm::NumericTablePtr local_partial_sums;
    daal_dm::NumericTablePtr local_partial_objective;
    daal_dm::NumericTablePtr local_partial_candidate_distances;
    daal_dm::NumericTablePtr local_partial_candidate_centroids;
    daal_dm::NumericTablePtr master_centroids;
    daal_dm::NumericTablePtr master_objective;
};

struct step3_input {
    step3_input& set_local_data(const daal_dm::NumericTablePtr& value) {
        local_data = value;
        return *this;
    }

    step3_input& set_global_centroids(const daal_dm::NumericTablePtr& value) {
        global_centroids = value;
        return *this;
    }

    daal_dm::NumericTablePtr local_data;
    daal_dm::NumericTablePtr global_centroids;
};

struct step3_result {
    template <typename Float>
    void allocate(std::int64_t row_count, std::int64_t feature_count, std::int64_t cluster_count) {
        allocate_daal_nt<int>(local_assignments, row_count, 1);
        allocate_daal_nt<Float>(local_centroids, cluster_count, feature_count);
        allocate_daal_nt<Float>(local_objective, 1, 1);
        allocate_daal_nt<int>(local_iteration_count, 1, 1);
    }

    daal_dm::NumericTablePtr local_assignments;
    daal_dm::NumericTablePtr local_centroids;
    daal_dm::NumericTablePtr local_objective;
    daal_dm::NumericTablePtr local_iteration_count;
};

inline daal_kmeans::Parameter get_daal_parameter(const descriptor_t& desc,
                                                 bool compute_assignments) {
    daal_kmeans::Parameter daal_param(
        dal::detail::integral_cast<std::size_t>(desc.get_cluster_count()),
        dal::detail::integral_cast<std::size_t>(desc.get_max_iteration_count()));
    daal_param.accuracyThreshold = desc.get_accuracy_threshold();
    daal_param.resultsToEvaluate = daal_kmeans::computeCentroids | //
                                   daal_kmeans::computeExactObjectiveFunction;
    if (compute_assignments) {
        daal_param.resultsToEvaluate |= daal_kmeans::computeAssignments;
    }
    return daal_param;
}

template <typename Float>
static void run_step1_local(const context_cpu& ctx,
                            const descriptor_t& desc,
                            const step1_input& input,
                            step1_result& result) {
    constexpr std::size_t input_count = 2;
    constexpr std::size_t result_count = 6;

    const daal_dm::NumericTable* input_raw[input_count] = //
        { input.local_data.get(), input.global_centroids.get() };

    const daal_dm::NumericTable* result_raw[result_count] = {
        result.local_observation_count.get(),
        result.local_partial_sums.get(),
        result.local_partial_objective.get(),
        result.local_partial_candidate_distances.get(),
        result.local_partial_candidate_centroids.get(),
        nullptr, // Do not compute partial assignments
    };

    // Do not compute assignments at the first step. They need to be computed
    // once final centroids are determined, i.e., at the third step.
    constexpr bool compute_assignments = false;
    auto daal_param = get_daal_parameter(desc, compute_assignments);

    interop::status_to_exception( //
        interop::call_daal_kernel<Float, daal_kmeans_step1_kernel_t>( //
            ctx,
            input_count,
            input_raw,
            result_count - 1,
            result_raw,
            &daal_param));
}

template <typename Float>
static void run_step2_root(const context_cpu& ctx,
                           const descriptor_t& desc,
                           const step2_input& input,
                           step2_result& result) {
    constexpr std::size_t input_count_per_rank = 5;
    constexpr std::size_t result_count = 5;
    constexpr std::size_t final_result_count = 2;

    const std::size_t collected_result_count = input.master_step1_results.size();
    const std::size_t input_count =
        dal::detail::check_mul_overflow(input_count_per_rank, collected_result_count);

    std::vector<daal_dm::NumericTable*> input_raw;
    input_raw.reserve(input_count);

    for (std::size_t i = 0; i < collected_result_count; i++) {
        const auto& res = input.master_step1_results[i];
        input_raw.push_back(res.local_observation_count.get());
        input_raw.push_back(res.local_partial_sums.get());
        input_raw.push_back(res.local_partial_objective.get());
        input_raw.push_back(res.local_partial_candidate_distances.get());
        input_raw.push_back(res.local_partial_candidate_centroids.get());
    }
    ONEDAL_ASSERT(input_raw.size() == input_count);

    const daal_dm::NumericTable* result_raw[result_count] = {
        result.local_observation_count.get(),
        result.local_partial_sums.get(),
        result.local_partial_objective.get(),
        result.local_partial_candidate_distances.get(),
        result.local_partial_candidate_centroids.get(),
    };

    // Do not compute assignments at the second step. They need to be computed
    // once final centroids are determined, i.e., at the third step.
    constexpr bool compute_assignments = false;
    auto daal_param = get_daal_parameter(desc, compute_assignments);

    interop::status_to_exception( //
        interop::call_daal_kernel<Float, daal_kmeans_step2_kernel_t>( //
            ctx,
            input_count,
            input_raw.data(),
            result_count,
            result_raw,
            &daal_param));

    const daal_dm::NumericTable* final_result_raw[final_result_count] = {
        result.master_centroids.get(),
        result.master_objective.get(),
    };

    interop::status_to_exception( //
        interop::call_daal_kernel_finalize<Float, daal_kmeans_step2_kernel_t>( //
            ctx,
            result_count,
            result_raw,
            final_result_count,
            final_result_raw,
            &daal_param));
}

template <typename Float>
static void run_step3_local(const context_cpu& ctx,
                            const descriptor_t& desc,
                            const step3_input& input,
                            step3_result& result) {
    constexpr std::size_t input_count = 2;
    constexpr std::size_t result_count = 4;

    const daal_dm::NumericTable* input_raw[input_count] = {
        input.local_data.get(),
        input.global_centroids.get(),
    };

    const daal_dm::NumericTable* result_raw[result_count] = {
        result.local_centroids.get(),
        result.local_assignments.get(),
        result.local_objective.get(),
        result.local_iteration_count.get(),
    };

    constexpr bool compute_assignments = true;
    auto daal_param = get_daal_parameter(desc, compute_assignments);

    // We want only to update assignments but keep centroids
    daal_param.maxIterations = 0;

    interop::status_to_exception( //
        interop::call_daal_kernel<Float, daal_kmeans_step3_kernel_t>( //
            ctx,
            input_raw,
            result_raw,
            &daal_param));
}

template <typename Float>
static result_t train(const context_cpu& ctx, const descriptor_t& desc, const input_t& input) {
    auto& comm = ctx.get_communicator();

    const std::int64_t row_count = input.get_data().get_row_count();
    const std::int64_t cluster_count = desc.get_cluster_count();
    const std::int64_t feature_count = input.get_data().get_column_count();

    const daal_dm::NumericTablePtr daal_local_data =
        interop::copy_to_daal_homogen_table<Float>(input.get_data());

    double current_objective = std::numeric_limits<double>::infinity();
    [[maybe_unused]] double previous_objective = std::numeric_limits<double>::infinity();
    auto current_centroids = row_accessor<const Float>{ input.get_initial_centroids() }.pull();
    current_centroids.need_mutable_data();

    step1_result local_step1_result;
    step2_result root_step2_result;

    local_step1_result.allocate<Float>(cluster_count, feature_count);
    if (comm.is_root()) {
        root_step2_result.allocate<Float>(cluster_count, feature_count);
    }

    std::int64_t iteration_counter = 0;
    for (std::int64_t i = 0; i < desc.get_max_iteration_count(); i++) {
        previous_objective = current_objective;
        iteration_counter++;

        {
            const auto daal_centroids = interop::convert_to_daal_homogen_table(current_centroids,
                                                                               cluster_count,
                                                                               feature_count);
            const auto local_step1_input = //
                step1_input{} //
                    .set_global_centroids(daal_centroids)
                    .set_local_data(daal_local_data);

            run_step1_local<Float>(ctx, desc, local_step1_input, local_step1_result);
        }

        const auto gathered_results = comm.gather(local_step1_result);

        if (comm.is_root()) {
            {
                const auto root_step2_input = //
                    step2_input{} //
                        .set_master_step1_results(gathered_results);

                run_step2_root<Float>(ctx, desc, root_step2_input, root_step2_result);
            }

            current_objective = root_step2_result.master_objective->getValue<double>(0, 0);
            interop::copy_from_daal_table<Float>(current_centroids,
                                                 root_step2_result.master_centroids);
        }

        comm.bcast(current_objective);
        comm.bcast(current_centroids);
    }

    table local_labels;
    {
        const auto daal_centroids =
            interop::convert_to_daal_homogen_table(current_centroids, cluster_count, feature_count);

        step3_result local_step3_result;
        local_step3_result.allocate<Float>(row_count, feature_count, cluster_count);

        const auto local_step3_input = //
            step3_input{} //
                .set_local_data(daal_local_data)
                .set_global_centroids(daal_centroids);

        run_step3_local<Float>(ctx, desc, local_step3_input, local_step3_result);

        // TODO: Memory must be allocated on new interfaces side
        local_labels =
            interop::convert_from_daal_table<std::int32_t>(local_step3_result.local_assignments);
    }

    const auto model = model_t{}.set_centroids(
        homogen_table::wrap(current_centroids, cluster_count, feature_count));

    return result_t{} //
        .set_model(model)
        .set_iteration_count(iteration_counter)
        .set_labels(local_labels)
        .set_objective_function_value(current_objective);
}

template <typename Float>
struct train_kernel_cpu_spmd<Float, method::lloyd_dense, task::clustering> {
    result_t operator()(const context_cpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_cpu_spmd<float, method::lloyd_dense, task::clustering>;
template struct train_kernel_cpu_spmd<double, method::lloyd_dense, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
