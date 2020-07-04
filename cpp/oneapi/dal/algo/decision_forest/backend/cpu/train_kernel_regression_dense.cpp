/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <daal/include/services/error_handling.h>
#include <daal/src/algorithms/dtrees/forest/regression/df_regression_model_impl.h>
#include <daal/src/services/service_algo_utils.h>

#include <daal/include/algorithms/decision_forest/decision_forest_regression_training_batch.h>
#include <daal/include/algorithms/decision_forest/decision_forest_regression_training_types.h>

#include <daal/src/algorithms/dtrees/forest/regression/df_regression_train_kernel.h>
//to prevent reordering by clang-format
#include <daal/src/algorithms/dtrees/forest/regression/df_regression_train_dense_default_kernel.h>

#include "oneapi/dal/algo/decision_forest/backend/cpu/train_kernel.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/decision_forest/model_impl.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_cpu;

namespace df      = daal::algorithms::decision_forest;
namespace rgr     = daal::algorithms::decision_forest::regression;
namespace interop = dal::backend::interop;

template <typename Float, daal::CpuType Cpu>
using rgr_default_dense_kernel_t =
    rgr::training::internal::RegressionTrainBatchKernel<Float, rgr::training::defaultDense, Cpu>;

using rgr_model_p = rgr::ModelPtr;

template <typename Float, typename Task>
static train_result<Task> call_daal_kernel(const context_cpu& ctx,
                                           const descriptor_base<Task>& desc,
                                           const table& data,
                                           const table& labels) {
    const int64_t row_count    = data.get_row_count();
    const int64_t column_count = data.get_column_count();

    auto arr_data  = row_accessor<const Float>{ data }.pull();
    auto arr_label = row_accessor<const Float>{ labels }.pull();

    const auto daal_data =
        interop::convert_to_daal_homogen_table(arr_data, row_count, column_count);
    const auto daal_labels = interop::convert_to_daal_homogen_table(arr_label, row_count, 1);

    /* init param for daal kernel */
    auto daal_input = rgr::training::Input();
    daal_input.set(rgr::training::data, daal_data);
    daal_input.set(rgr::training::dependentVariable, daal_labels);

    auto daal_parameter                        = rgr::training::Parameter();
    daal_parameter.nTrees                      = desc.get_tree_count();
    daal_parameter.observationsPerTreeFraction = desc.get_observations_per_tree_fraction();
    daal_parameter.featuresPerNode             = desc.get_features_per_node();
    daal_parameter.maxTreeDepth                = desc.get_max_tree_depth();
    daal_parameter.minObservationsInLeafNode   = desc.get_min_observations_in_leaf_node();
    // TODO take engines from desc
    daal_parameter.engine            = daal::algorithms::engines::mt2203::Batch<>::create();
    daal_parameter.impurityThreshold = desc.get_impurity_threshold();
    daal_parameter.memorySavingMode  = desc.get_memory_saving_mode();
    daal_parameter.bootstrap         = desc.get_bootstrap();
    daal_parameter.minObservationsInSplitNode     = desc.get_min_observations_in_split_node();
    daal_parameter.minWeightFractionInLeafNode    = desc.get_min_weight_fraction_in_leaf_node();
    daal_parameter.minImpurityDecreaseInSplitNode = desc.get_min_impurity_decrease_in_split_node();
    daal_parameter.maxLeafNodes                   = desc.get_max_leaf_nodes();

    daal_parameter.resultsToCompute = desc.get_train_results_to_compute();

    auto vimp                    = desc.get_variable_importance_mode();
    daal_parameter.varImportance = variable_importance_mode::mdi == vimp
                                       ? df::training::MDI
                                       : variable_importance_mode::mda_raw == vimp
                                             ? df::training::MDA_Raw
                                             : variable_importance_mode::mda_scaled == vimp
                                                   ? df::training::MDA_Scaled
                                                   : df::training::none;

    train_result<Task> res;

    auto daal_result = rgr::training::Result();

    /* init daal result's objects */
    if (desc.get_train_results_to_compute() &
        static_cast<std::uint64_t>(train_result_to_compute::compute_out_of_bag_error)) {
        array<Float> arr_oob_err{ 1 * 1 };
        res.set_oob_err(homogen_table_builder(1, arr_oob_err).build());

        const auto res_oob_err = interop::convert_to_daal_homogen_table(arr_oob_err, 1, 1);
        daal_result.set(rgr::training::outOfBagError, res_oob_err);
    }

    if (desc.get_train_results_to_compute() &
        static_cast<std::uint64_t>(
            train_result_to_compute::compute_out_of_bag_error_per_observation)) {
        array<Float> arr_oob_per_obs_err{ row_count * 1 };
        res.set_oob_per_observation_err(homogen_table_builder(1, arr_oob_per_obs_err).build());

        const auto res_oob_per_obs_err =
            interop::convert_to_daal_homogen_table(arr_oob_per_obs_err, row_count, 1);
        daal_result.set(rgr::training::outOfBagErrorPerObservation, res_oob_per_obs_err);
    }
    if (variable_importance_mode::none != vimp) {
        array<Float> arr_var_imp{ 1 * column_count };
        res.set_var_importance(homogen_table_builder(column_count, arr_var_imp).build());

        const auto res_var_imp =
            interop::convert_to_daal_homogen_table(arr_var_imp, 1, column_count);
        daal_result.set(rgr::training::variableImportance, res_var_imp);
    }

    rgr::ModelPtr mptr = rgr::ModelPtr(new rgr::internal::ModelImpl(column_count));

    interop::call_daal_kernel<Float, rgr_default_dense_kernel_t>(
        ctx,
        daal::services::internal::hostApp(daal_input),
        daal_data.get(),
        daal_labels.get(),
        *mptr,
        daal_result,
        daal_parameter);

    /* extract results from daal objects */
    if (desc.get_train_results_to_compute() &
        static_cast<std::uint64_t>(train_result_to_compute::compute_out_of_bag_error)) {
        auto table_oob_err = interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(rgr::training::outOfBagError));
        res.set_oob_err(table_oob_err);
    }

    if (desc.get_train_results_to_compute() &
        static_cast<std::uint64_t>(
            train_result_to_compute::compute_out_of_bag_error_per_observation)) {
        auto table_oob_per_obs_err = interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(rgr::training::outOfBagErrorPerObservation));
        res.set_oob_per_observation_err(table_oob_per_obs_err);
    }

    if (variable_importance_mode::none != vimp) {
        auto table_var_imp = interop::convert_from_daal_homogen_table<Float>(
            daal_result.get(rgr::training::variableImportance));
        res.set_var_importance(table_var_imp);
    }

    return res.set_model(dal::detail::pimpl_accessor().make_from_pimpl<model<Task>>(
        std::make_shared<interop::decision_forest::interop_model_impl<Task, rgr_model_p>>(mptr)));
}

template <typename Float, typename Task>
static train_result<Task> train(const context_cpu& ctx,
                                const descriptor_base<Task>& desc,
                                const train_input<Task>& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_labels());
}

template <typename Float, typename Task>
struct train_kernel_cpu<Float, Task, method::default_dense> {
    train_result<Task> operator()(const context_cpu& ctx,
                                  const descriptor_base<Task>& desc,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, input);
    }
};

template struct train_kernel_cpu<float, task::regression, method::default_dense>;
template struct train_kernel_cpu<double, task::regression, method::default_dense>;

} // namespace oneapi::dal::decision_forest::backend
