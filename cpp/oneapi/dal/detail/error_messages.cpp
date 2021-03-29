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

#include "oneapi/dal/detail/error_messages.hpp"

#define MSG(id, text)                           \
    const char* error_messages::id() noexcept { \
        return text;                            \
    }

namespace oneapi::dal::detail {
namespace v1 {

/* Common */
MSG(array_does_not_contain_mutable_data, "Array does not contain mutable data")
MSG(feature_index_is_out_of_range, "Feature index is out of range")
MSG(incompatible_array_reinterpret_cast_types,
    "Cannot reinterpret array to provided type, "
    "because resulting array size would not match source array size")
MSG(only_homogen_table_is_supported, "Only homogen table is supported")
MSG(overflow_found_in_multiplication_of_two_values,
    "Overflow found in multiplication of two values")
MSG(overflow_found_in_sum_of_two_values, "Overflow found in sum of two values")
MSG(unknown_status_code, "Unknown status code")
MSG(unsupported_data_layout, "Unsupported data layout")
MSG(unsupported_data_type, "Requested data type is not supported")
MSG(unsupported_device_type, "Requested device type is not supported")
MSG(small_data_block, "Data block size is smaller than expected")
MSG(invalid_data_block_size, "Invalid data block size")
MSG(method_not_implemented, "Method is not implemented")
MSG(unsupported_feature_type, "Feature type is not supported")
MSG(unknown_memcpy_error, "Unknown error during memory copying")
MSG(unknown_usm_pointer_type, "USM pointer type is unknown in the current context")
MSG(queues_in_different_contexts, "Provided queues are in different contexts")
MSG(unsupported_usm_alloc, "Requested USM alloc type is not supported")

/* Primitives */
MSG(invalid_number_of_elements_to_sort, "Invalid number of elements to sort")

/* Tables */
MSG(allocated_memory_size_is_not_enough_to_copy_data,
    "Allocated memory size is not enough to copy the data")
MSG(cannot_get_data_type_from_empty_metadata, "Cannot get data type from empty metadata")
MSG(cannot_get_feature_type_from_empty_metadata, "Cannot get feature type from empty metadata")
MSG(cc_leq_zero, "Column count is lower than or equal to zero")
MSG(element_count_in_data_type_and_feature_type_arrays_does_not_match,
    "Element count in data type and feature type array does not match")
MSG(pulling_column_is_not_supported_for_dpc, "Pulling column is not supported for DPC++")
MSG(pulling_column_is_not_supported, "Pulling column is not supported")
MSG(pulling_rows_is_not_supported_for_dpc, "Pulling rows is not supported for DPC++")
MSG(pulling_rows_is_not_supported, "Pulling rows is not supported")
MSG(pushing_column_is_not_supported_for_dpc, "Pushing column is not supported for DPC++")
MSG(pushing_column_is_not_supported, "Pushing column is not supported")
MSG(pushing_rows_is_not_supported_for_dpc, "Pushing rows is not supported for DPC++")
MSG(pushing_rows_is_not_supported, "Pushing rows is not supported")
MSG(rc_and_cc_do_not_match_element_count_in_array,
    "Row count and column count do not match element count in array")
MSG(rc_leq_zero, "Row count is lower than or equal to zero")
MSG(unsupported_conversion_types, "Unsupported conversion types")

/* Ranges */
MSG(invalid_range_of_rows, "Invalid range of rows")
MSG(invalid_range_of_columns, "Invalid range of columns")
MSG(column_index_out_of_range, "Column index out of range")

/* Graphs */
MSG(vertex_index_out_of_range_expect_from_zero_to_vertex_count,
    "Vertex index is out of range, expect index in [0, vertex_count)")
MSG(negative_vertex_id, "Negative vertex ID")
MSG(unimplemented_sorting_procedure, "Unimplemented sorting procedure")

/* General algorithms */
MSG(accuracy_threshold_lt_zero, "Accuracy_threshold is lower than zero")
MSG(class_count_leq_one, "Class count is lower than or equal to one")
MSG(input_data_is_empty, "Input data is empty")
MSG(input_data_rc_neq_input_labels_rc,
    "Input data row count is not equal to input labels row count")
MSG(input_data_rc_neq_input_weights_rc,
    "Input data row count is not equal to input weights row count")
MSG(input_labels_are_empty, "Labels are empty")
MSG(input_labels_contain_only_one_unique_value_expect_two,
    "Input labels contain only one unique value, two unique values are expected")
MSG(input_labels_contain_wrong_unique_values_count_expect_two,
    "Input labels contain wrong number of unique values, two unique values are expected")
MSG(input_labels_table_has_wrong_cc_expect_one,
    "Input labels table has wrong column count, one column is expected")
MSG(iteration_count_lt_zero, "Iteration count is lower than zero")
MSG(max_iteration_count_leq_zero, "Max iteration count lower than or equal to zero")
MSG(max_iteration_count_lt_zero, "Max iteration count lower than zero")

/* IO */
MSG(file_not_found, "File not found")

/* K-Means */
MSG(cluster_count_leq_zero, "Cluster count is lower than or equal to zero")
MSG(input_initial_centroids_are_empty, "Input initial centroids are empty")
MSG(input_initial_centroids_cc_neq_input_data_cc,
    "Input initial centroids column count is not equal to input data column count")
MSG(input_initial_centroids_rc_neq_desc_cluster_count,
    "Input initial centroids row count is not equal to descriptor cluster count")
MSG(input_model_centroids_are_empty, "Input model centroids are empty")
MSG(input_model_centroids_cc_neq_input_data_cc,
    "Input model centroids column count is not equal to input data column count")
MSG(input_model_centroids_rc_neq_desc_cluster_count,
    "Input model centroids row count is not equal to descriptor cluster count")
MSG(kmeans_init_parallel_plus_dense_method_is_not_implemented_for_gpu,
    "K-Means init++ parallel dense method is not implemented for GPU")
MSG(kmeans_init_plus_plus_dense_method_is_not_implemented_for_gpu,
    "K-Means init++ dense method is not implemented for GPU")
MSG(objective_function_value_lt_zero, "Objective function value is lower than zero")

/* k-NN */
MSG(knn_brute_force_method_is_not_implemented_for_cpu,
    "k-NN brute force method is not implemented for CPU")
MSG(knn_kd_tree_method_is_not_implemented_for_gpu,
    "k-NN k-d tree method is not implemented for GPU")
MSG(neighbor_count_lt_one, "Neighbor count lower than one")

/* Jaccard */
MSG(column_begin_gt_column_end, "Column begin is greater than column end")
MSG(empty_edge_list, "Empty edge list")
MSG(interval_gt_vertex_count, "Interval is greater than vertex count")
MSG(negative_interval, "Negative interval")
MSG(row_begin_gt_row_end, "Row begin is greater than row end")
MSG(range_idx_gt_max_int32, "Range indexes are greater than max of int32")

/* PCA */
MSG(component_count_lt_zero, "Component count is lower than zero")
MSG(input_data_cc_lt_desc_component_count,
    "Input data column count is lower than component count provided in descriptor")
MSG(input_model_eigenvectors_cc_neq_input_data_cc,
    "Input model eigenvectors column count is not equal to input data column count")
MSG(input_model_eigenvectors_rc_neq_desc_component_count,
    "Eigenvectors' row count in input model is not equal to component count provided in descriptor")
MSG(input_model_eigenvectors_rc_neq_input_data_cc,
    "Eigenvectors' row count in input model is not equal to input data column count")
MSG(pca_svd_based_method_is_not_implemented_for_gpu,
    "PCA SVD-based method is not implemented for GPU")

/* SVM */
MSG(c_leq_zero, "C is lower than or equal to zero")
MSG(cache_size_lt_zero, "Cache size is lower than zero")
MSG(degree_lt_zero, "Degree lower than zero")
MSG(input_model_coeffs_are_empty, "Input model coeffs are empty")
MSG(input_model_coeffs_rc_neq_input_model_support_vector_count,
    "Input model coeffs row count is not equal to support vector count provided in input model")
MSG(input_model_does_not_match_kernel_function, "Input model does not match kernel function type")
MSG(input_model_support_vectors_are_empty, "Input model support vectors are empty")
MSG(input_model_support_vectors_cc_neq_input_data_cc,
    "Input model support vectors column count is not equal to input data column count")
MSG(input_model_support_vectors_rc_neq_input_model_support_vector_count,
    "Support vectors row count is not equal to support vector count in input model")
MSG(polynomial_kenrel_is_not_implemented_for_gpu, "Polynomial kernel is not implemented for GPU")
MSG(sigma_leq_zero, "Sigma lower than or equal to zero")
MSG(svm_smo_method_is_not_implemented_for_gpu, "SVM SMO method is not implemented for GPU")
MSG(svm_regression_task_is_not_implemented_for_gpu, "Regression SVM is not implemented for GPU")
MSG(tau_leq_zero, "Tau is lower than or equal to zero")
MSG(epsilon_lt_zero, "Epsilon is lower than zero")
MSG(unknown_kernel_function_type, "Unknown kernel function type")

/* Kernel Functions */
MSG(input_x_cc_neq_y_cc, "Input x column count is not qual to y column count")
MSG(input_x_is_empty, "Input x is empty")
MSG(input_y_is_empty, "Input y is empty")

/* Decision Forest */
MSG(bootstrap_is_incompatible_with_error_metric,
    "Values of bootstrap and error metric parameters provided "
    "in descriptor are incompatible to each other")
MSG(bootstrap_is_incompatible_with_variable_importance_mode,
    "Values of bootstrap and variable importance mode parameters provided "
    "in descriptor are incompatible to each other")
MSG(decision_forest_train_dense_method_is_not_implemented_for_gpu,
    "Decision forest train dense method is not implemented for GPU")
MSG(decision_forest_train_hist_method_is_not_implemented_for_cpu,
    "Decision forest train hist method is not implemented for CPU")

} // namespace v1
} // namespace oneapi::dal::detail
