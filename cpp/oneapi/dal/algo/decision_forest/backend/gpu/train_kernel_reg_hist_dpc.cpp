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

#include <daal/src/services/service_algo_utils.h>
#include <daal/src/algorithms/dtrees/forest/regression/df_regression_model_impl.h>
#include <daal/include/algorithms/decision_forest/decision_forest_regression_training_batch.h>
#include <daal/include/algorithms/decision_forest/decision_forest_regression_training_types.h>
#include <daal/src/algorithms/dtrees/forest/regression/oneapi/df_regression_train_hist_kernel_oneapi.h>

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_kernel.hpp"

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/algo/decision_forest/backend/model_impl.hpp"

namespace oneapi::dal::decision_forest::backend {

using dal::backend::context_gpu;
using model_t = model<task::regression>;
using input_t = train_input<task::regression>;
using result_t = train_result<task::regression>;
using descriptor_t = detail::descriptor_base<task::regression>;

namespace daal_df = daal::algorithms::decision_forest;
namespace daal_df_reg_train = daal_df::regression::training;
namespace interop = dal::backend::interop;

template <typename Float>
using reg_hist_kernel_t =
    daal_df_reg_train::internal::RegressionTrainBatchKernelOneAPI<Float, daal_df_reg_train::hist>;

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data,
                                 const table& labels) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const int64_t row_count = data.get_row_count();
    const int64_t column_count = data.get_column_count();

    auto arr_data = row_accessor<const Float>{ data }.pull(queue);
    auto arr_label = row_accessor<const Float>{ labels }.pull(queue);

    const auto daal_data =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_data, row_count, column_count);
    const auto daal_labels =
        interop::convert_to_daal_sycl_homogen_table(queue, arr_label, row_count, 1);

    /* init param for daal kernel */
    auto daal_input = daal_df_reg_train::Input();
    daal_input.set(daal_df_reg_train::data, daal_data);
    daal_input.set(daal_df_reg_train::dependentVariable, daal_labels);

    auto daal_parameter = daal_df_reg_train::Parameter();
    daal_parameter.nTrees = dal::detail::integral_cast<std::size_t>(desc.get_tree_count());
    daal_parameter.observationsPerTreeFraction = desc.get_observations_per_tree_fraction();
    daal_parameter.featuresPerNode =
        dal::detail::integral_cast<std::size_t>(desc.get_features_per_node());
    daal_parameter.maxTreeDepth =
        dal::detail::integral_cast<std::size_t>(desc.get_max_tree_depth());
    daal_parameter.minObservationsInLeafNode =
        dal::detail::integral_cast<std::size_t>(desc.get_min_observations_in_leaf_node());
    // TODO take engines from desc
    daal_parameter.engine = daal::algorithms::engines::mt2203::Batch<>::create();
    daal_parameter.impurityThreshold = desc.get_impurity_threshold();
    daal_parameter.memorySavingMode = desc.get_memory_saving_mode();
    daal_parameter.bootstrap = desc.get_bootstrap();
    daal_parameter.minObservationsInSplitNode =
        dal::detail::integral_cast<std::size_t>(desc.get_min_observations_in_split_node());
    daal_parameter.minWeightFractionInLeafNode = desc.get_min_weight_fraction_in_leaf_node();
    daal_parameter.minImpurityDecreaseInSplitNode = desc.get_min_impurity_decrease_in_split_node();
    daal_parameter.maxLeafNodes =
        dal::detail::integral_cast<std::size_t>(desc.get_max_leaf_nodes());
    daal_parameter.maxBins = dal::detail::integral_cast<std::size_t>(desc.get_max_bins());
    daal_parameter.minBinSize = dal::detail::integral_cast<std::size_t>(desc.get_min_bin_size());

    daal_parameter.resultsToCompute = static_cast<std::uint64_t>(desc.get_error_metric_mode());

    auto vimp = desc.get_variable_importance_mode();

    daal_parameter.varImportance = convert_to_daal_variable_importance_mode(vimp);

    result_t res;

    auto daal_result = daal_df_reg_train::Result();

    /* init daal result's objects */
    array<Float> arr_oob_err;
    if (check_mask_flag(desc.get_error_metric_mode(), error_metric_mode::out_of_bag_error)) {
        arr_oob_err = array<Float>::empty(queue, 1 * 1);

        const auto res_oob_err =
            interop::convert_to_daal_sycl_homogen_table(queue, arr_oob_err, 1, 1);
        daal_result.set(daal_df_reg_train::outOfBagError, res_oob_err);
    }

    array<Float> arr_oob_per_obs_err;
    if (check_mask_flag(desc.get_error_metric_mode(),
                        error_metric_mode::out_of_bag_error_per_observation)) {
        arr_oob_per_obs_err = array<Float>::empty(queue, row_count * 1);

        const auto res_oob_per_obs_err =
            interop::convert_to_daal_sycl_homogen_table(queue, arr_oob_per_obs_err, row_count, 1);
        daal_result.set(daal_df_reg_train::outOfBagErrorPerObservation, res_oob_per_obs_err);
    }

    array<Float> arr_var_imp;
    if (variable_importance_mode::none != vimp) {
        arr_var_imp = array<Float>::empty(queue, 1 * column_count);

        const auto res_var_imp =
            interop::convert_to_daal_sycl_homogen_table(queue, arr_var_imp, 1, column_count);
        daal_result.set(daal_df_reg_train::variableImportance, res_var_imp);
    }

    daal_df::regression::ModelPtr mptr =
        daal_df::regression::ModelPtr(new daal_df::regression::internal::ModelImpl(column_count));

    interop::status_to_exception(
        reg_hist_kernel_t<Float>().compute(daal::services::internal::hostApp(daal_input),
                                           daal_data.get(),
                                           daal_labels.get(),
                                           *mptr,
                                           daal_result,
                                           daal_parameter));

    /* extract results from daal objects */

    if (check_mask_flag(desc.get_error_metric_mode(), error_metric_mode::out_of_bag_error)) {
        res.set_oob_err(dal::detail::homogen_table_builder{}.reset(arr_oob_err, 1, 1).build());
    }
    if (check_mask_flag(desc.get_error_metric_mode(),
                        error_metric_mode::out_of_bag_error_per_observation)) {
        res.set_oob_err_per_observation(
            dal::detail::homogen_table_builder{}.reset(arr_oob_per_obs_err, row_count, 1).build());
    }
    if (variable_importance_mode::none != vimp) {
        res.set_var_importance(
            dal::detail::homogen_table_builder{}.reset(arr_var_imp, 1, column_count).build());
    }

    const auto model_impl = std::make_shared<model_impl_reg>(new model_interop_impl{ mptr });
    model_impl->tree_count = mptr->getNumberOfTrees();

    return res.set_model(dal::detail::make_private<model_t>(model_impl));
}

template <typename Float>
static result_t train(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_labels());
}

template <typename Float, typename Task>
struct train_kernel_gpu<Float, method::hist, Task> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return train<Float>(ctx, desc, input);
    }
};

template struct train_kernel_gpu<float, method::hist, task::regression>;
template struct train_kernel_gpu<double, method::hist, task::regression>;

} // namespace oneapi::dal::decision_forest::backend
