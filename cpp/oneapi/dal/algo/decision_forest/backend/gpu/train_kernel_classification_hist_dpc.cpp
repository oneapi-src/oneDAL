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

#define DAAL_SYCL_INTERFACE
#define DAAL_SYCL_INTERFACE_USM
#define DAAL_SYCL_INTERFACE_REVERSED_RANGE

#include <daal/src/algorithms/dtrees/forest/classification/df_classification_model_impl.h>
#include <daal/src/services/service_algo_utils.h>

#include <daal/include/algorithms/decision_forest/decision_forest_classification_training_batch.h>
#include <daal/include/algorithms/decision_forest/decision_forest_classification_training_types.h>

#include <daal/src/algorithms/dtrees/forest/classification/oneapi/df_classification_train_hist_kernel_oneapi.h>

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/decision_forest/backend/interop_helpers.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_gpu;

namespace df  = daal::algorithms::decision_forest;
namespace cls = daal::algorithms::decision_forest::classification;

namespace interop    = dal::backend::interop;
namespace df_interop = dal::backend::interop::decision_forest;

template <typename Float>
using cls_hist_kernel_t =
    cls::training::internal::ClassificationTrainBatchKernelOneAPI<Float, cls::training::hist>;

using cls_model_p = cls::ModelPtr;

template <typename Float, typename Task>
static train_result<Task> call_daal_kernel(const context_gpu& ctx,
                                           const descriptor_base<Task>& desc,
                                           const train_input<Task>& input) {
    const table& data   = input.get_data();
    const table& labels = input.get_labels();
    auto& queue         = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const int64_t row_count    = data.get_row_count();
    const int64_t column_count = data.get_column_count();

    auto arr_data  = row_accessor<const Float>{ data }.pull(queue);
    auto arr_label = row_accessor<const Float>{ labels }.pull(queue);

    const auto daal_data =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_data, row_count, column_count);
    const auto daal_labels =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_label, row_count, 1);

    /* init param for daal kernel */
    auto daal_input = daal::algorithms::classifier::training::Input();
    daal_input.set(daal::algorithms::classifier::training::data, daal_data);
    daal_input.set(daal::algorithms::classifier::training::labels, daal_labels);

    auto daal_parameter                        = cls::training::Parameter(desc.get_class_count());
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

    daal_parameter.resultsToCompute = input.get_results_to_compute();

    auto vimp = desc.get_variable_importance_mode();

    daal_parameter.varImportance = df_interop::convert_to_daal_variable_importance_mode(vimp);

    train_result<Task> res;

    auto daal_result = cls::training::Result();

    /* init daal result's objects */
    array<Float> arr_oob_err;
    if (input.get_results_to_compute() &
        static_cast<std::uint64_t>(train_result_to_compute::compute_out_of_bag_error)) {
        arr_oob_err = array<Float>::empty(1 * 1);

        const auto res_oob_err =
            interop::convert_to_daal_sycl_homogen_table(queue, arr_oob_err, 1, 1);
        daal_result.set(cls::training::outOfBagError, res_oob_err);
    }

    array<Float> arr_oob_per_obs_err;
    if (input.get_results_to_compute() &
        static_cast<std::uint64_t>(
            train_result_to_compute::compute_out_of_bag_error_per_observation)) {
        arr_oob_per_obs_err = array<Float>::empty(queue, row_count * 1);

        const auto res_oob_per_obs_err =
            interop::convert_to_daal_sycl_homogen_table(queue, arr_oob_per_obs_err, row_count, 1);
        daal_result.set(cls::training::outOfBagErrorPerObservation, res_oob_per_obs_err);
    }

    array<Float> arr_var_imp;
    if (variable_importance_mode::none != vimp) {
        arr_var_imp = array<Float>::empty(queue, 1 * column_count);

        const auto res_var_imp =
            interop::convert_to_daal_sycl_homogen_table(queue, arr_var_imp, 1, column_count);
        daal_result.set(cls::training::variableImportance, res_var_imp);
    }

    cls::ModelPtr mptr = cls::ModelPtr(new cls::internal::ModelImpl(column_count));

    interop::status_to_exception(
        cls_hist_kernel_t<Float>().compute(daal::services::internal::hostApp(daal_input),
                                           daal_data.get(),
                                           daal_labels.get(),
                                           *mptr,
                                           daal_result,
                                           daal_parameter));

    /* extract results from daal objects */

    if (input.get_results_to_compute() &
        static_cast<std::uint64_t>(train_result_to_compute::compute_out_of_bag_error)) {
        res.set_oob_err(homogen_table_builder{}.reset(arr_oob_err, 1, 1).build());
    }
    if (input.get_results_to_compute() &
        static_cast<std::uint64_t>(
            train_result_to_compute::compute_out_of_bag_error_per_observation)) {
        res.set_oob_per_observation_err(
            homogen_table_builder{}.reset(arr_oob_per_obs_err, row_count, 1).build());
    }
    if (variable_importance_mode::none != vimp) {
        res.set_var_importance(homogen_table_builder{}.reset(arr_var_imp, 1, column_count).build());
    }

    return res.set_model(dal::detail::pimpl_accessor().make_from_pimpl<model<Task>>(
        std::make_shared<interop::decision_forest::interop_model_impl<Task, cls_model_p>>(mptr)));
}

template <typename Float, typename Task>
static train_result<Task> train(const context_gpu& ctx,
                                const descriptor_base<Task>& desc,
                                const train_input<Task>& input) {
    return call_daal_kernel<Float>(ctx, desc, input);
}

template <typename Float, typename Task>
struct train_kernel_gpu<Float, Task, method::hist> {
    train_result<Task> operator()(const context_gpu& ctx,
                                  const descriptor_base<Task>& desc,
                                  const train_input<Task>& input) const {
        return train<Float, Task>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, task::classification, method::hist>;
template struct train_kernel_gpu<double, task::classification, method::hist>;

} // namespace oneapi::dal::decision_forest::backend
