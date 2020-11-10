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

#pragma once

#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::detail {
namespace v1 {

#define MSG(id) static const char* id() noexcept

// -----------------------------------
// Guide for adding new error messages
// -----------------------------------
// - Before adding a new message check if there is a similar message
//   that can be reused. If you see some algorithm-specific message
//   can be reused in other algorithm/module, consider moving this to
//   common groups like `Common` or `General Algorithms`.
//
// - Select appropriate group among `Common`, `Tables`, ...
//   for a new message.
//
// - Keep alphabtical order of messages within each group.
//   If a new algorithm-specific group is needed, add it
//   in alphabetical order
//
// - Use LaTeX-like shortening for mathematical operations,
//   for example:
//
//   =    eq
//   !=   neq
//   >    gt
//   >=   geq
//   <    lt
//   <=   leq
//
// - If a new mathematical operator introduced, stick to LaTeX
//   conventions as well.
//
// - For oneDAL specific terms use the following shortening:
//
//   row count      rc
//   column count   cc
//   descriptor     desc

class ONEDAL_EXPORT error_messages {
public:
    error_messages() = delete;

    /* Common */
    MSG(array_does_not_contain_mutable_data);
    MSG(feature_index_is_out_of_range);
    MSG(only_homogen_table_is_supported);
    MSG(overflow_found_in_multiplication_of_two_values);
    MSG(overflow_found_in_sum_of_two_values);
    MSG(unknown_status_code);
    MSG(unsupported_data_layout);
    MSG(unsupported_data_type);
    MSG(unsupported_device_type);
    MSG(small_data_block);
    MSG(invalid_data_block_size);
    MSG(method_not_implemented);

    /* Tables */
    MSG(cannot_get_data_type_from_empty_metadata);
    MSG(cannot_get_feature_type_from_empty_metadata);
    MSG(element_count_in_data_type_and_feature_type_arrays_does_not_match);
    MSG(pulling_column_is_not_supported_for_dpc);
    MSG(pulling_column_is_not_supported);
    MSG(pulling_rows_is_not_supported_for_dpc);
    MSG(pulling_rows_is_not_supported);
    MSG(pushing_column_is_not_supported_for_dpc);
    MSG(pushing_column_is_not_supported);
    MSG(pushing_rows_is_not_supported_for_dpc);
    MSG(pushing_rows_is_not_supported);
    MSG(unsupported_conversion_types);
    MSG(rc_leq_zero);
    MSG(cc_leq_zero);

    /* Ranges */
    MSG(invalid_range_of_rows);
    MSG(column_index_out_of_range);

    /* Graphs */
    MSG(vertex_index_out_of_range_expect_from_zero_to_vertex_count);
    MSG(negative_vertex_id);

    /* General Algorithms */
    MSG(accuracy_threshold_lt_zero);
    MSG(class_count_leq_one);
    MSG(input_data_is_empty);
    MSG(input_data_rc_neq_input_labels_rc);
    MSG(input_data_rc_neq_input_weights_rc);
    MSG(input_labels_are_empty);
    MSG(input_labels_contain_only_one_unique_value_expect_two);
    MSG(input_labels_contain_wrong_unique_values_count_expect_two);
    MSG(input_labels_table_has_wrong_cc_expect_one);
    MSG(iteration_count_lt_zero);
    MSG(max_iteration_count_leq_zero);
    MSG(max_iteration_count_lt_zero);

    /* I/O */
    MSG(file_not_found);

    /* Decision Forest */
    MSG(bootstrap_is_incompatible_with_error_metric);
    MSG(bootstrap_is_incompatible_with_variable_importance_mode);
    MSG(decision_forest_train_dense_method_is_not_implemented_for_gpu);
    MSG(decision_forest_train_hist_method_is_not_implemented_for_cpu);

    /* Jaccard */
    MSG(column_begin_gt_column_end);
    MSG(empty_edge_list);
    MSG(interval_gt_vertex_count);
    MSG(negative_interval);
    MSG(row_begin_gt_row_end);
    MSG(range_idx_gt_max_int32);

    /* K-Means and K-Means Init */
    MSG(cluster_count_leq_zero);
    MSG(input_initial_centroids_are_empty);
    MSG(input_initial_centroids_cc_neq_input_data_cc);
    MSG(input_initial_centroids_rc_neq_desc_cluster_count);
    MSG(input_model_centroids_are_empty);
    MSG(input_model_centroids_cc_neq_input_data_cc);
    MSG(input_model_centroids_rc_neq_desc_cluster_count);
    MSG(kmeans_init_parallel_plus_dense_method_is_not_implemented_for_gpu);
    MSG(kmeans_init_plus_plus_dense_method_is_not_implemented_for_gpu);
    MSG(objective_function_value_lt_zero);

    /* k-NN */
    MSG(knn_brute_force_method_is_not_implemented_for_cpu);
    MSG(knn_kd_tree_method_is_not_implemented_for_gpu);
    MSG(neighbor_count_lt_one);

    /* Linear and RBF Kernels */
    MSG(input_x_cc_neq_y_cc);
    MSG(input_x_is_empty);
    MSG(input_y_is_empty);

    /* PCA */
    MSG(component_count_lt_zero);
    MSG(input_data_cc_lt_desc_component_count);
    MSG(input_model_eigenvectors_cc_neq_input_data_cc);
    MSG(input_model_eigenvectors_rc_neq_desc_component_count);
    MSG(pca_svd_based_method_is_not_implemented_for_gpu);

    /* SVM */
    MSG(c_leq_zero);
    MSG(cache_size_leq_zero);
    MSG(input_model_coeffs_are_empty);
    MSG(input_model_coeffs_rc_neq_input_model_support_vector_count);
    MSG(input_model_does_not_match_kernel_function);
    MSG(input_model_support_vectors_are_empty);
    MSG(input_model_support_vectors_cc_neq_input_data_cc);
    MSG(input_model_support_vectors_rc_neq_input_model_support_vector_count);
    MSG(sigma_leq_zero);
    MSG(svm_smo_method_is_not_implemented_for_gpu);
    MSG(tau_leq_zero);
    MSG(unknown_kernel_function_type);
};

#undef MSG

} // namespace v1

using v1::error_messages;

} // namespace oneapi::dal::detail
