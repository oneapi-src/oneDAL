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

#include "oneapi/dal/algo/decision_forest/common.hpp"

namespace oneapi::dal::decision_forest {

class detail::descriptor_impl : public base {
  public:
    double observations_per_tree_fraction;
    double impurity_threshold;
    double min_weight_fraction_in_leaf_node;
    double min_impurity_decrease_in_split_node;

    std::int64_t n_classes;
    std::int64_t n_trees;
    std::int64_t features_per_node;
    std::int64_t max_tree_depth;
    std::int64_t min_observations_in_leaf_node;
    std::int64_t seed;
    std::int64_t min_observations_in_split_node;
    std::int64_t max_leaf_nodes;

    std::uint64_t results_to_compute;

    bool memory_saving_mode;
    bool bootstrap;

    // engine field

    variable_importance_mode variable_importance_mode_value;
    voting_method voting_method_value;
};

class detail::model_impl : public base {
  public:
    table eigenvectors;
};

using detail::descriptor_impl;
using detail::model_impl;

descriptor_base::descriptor_base()
    : impl_(new descriptor_impl{}) {}

/*getters implementation*/
double descriptor_base::get_observations_per_tree_fraction() const {
    return impl_->observations_per_tree_fraction;
}
double descriptor_base::get_impurity_threshold() const {
    return impl_->impurity_threshold;
}    
double descriptor_base::get_min_weight_fraction_in_leaf_node() const {
    return impl_->min_weight_fraction_in_leaf_node;
}    
double descriptor_base::get_min_impurity_decrease_in_split_node() const {
    return impl_->min_impurity_decrease_in_split_node;
}    

std::int64_t descriptor_base::get_n_classes() const {
    return impl_->n_classes;
}    
std::int64_t descriptor_base::get_n_trees() const {
    return impl_->n_trees;
}    
std::int64_t descriptor_base::get_features_per_node() const {
    return impl_->features_per_node;
}    
std::int64_t descriptor_base::get_max_tree_depth() const {
    return impl_->max_tree_depth;
}    
std::int64_t descriptor_base::get_min_observations_in_leaf_node() const {
    return impl_->min_observations_in_leaf_node;
}    
std::int64_t descriptor_base::get_seed() const {
    return impl_->seed;
}    
std::int64_t descriptor_base::get_min_observations_in_split_node() const {
    return impl_->min_observations_in_split_node;
}    
std::int64_t descriptor_base::get_max_leaf_nodes() const {
    return impl_->max_leaf_nodes;
}    

std::uint64_t descriptor_base::get_results_to_compute() const {
    return impl_->results_to_compute;
}    

bool descriptor_base::get_memory_saving_mode() const {
    return impl_->memory_saving_mode;
}
bool descriptor_base::get_bootstrap() const {
    return impl_->bootstrap;
}

variable_importance_mode descriptor_base::get_variable_importance_mode() const {
    return impl_->variable_importance_mode_value;
}
voting_method descriptor_base::get_voting_method() const {
    return impl_->voting_method_value;
}
/*setters implementation*/

void descriptor_base::set_observations_per_tree_fraction_impl(const double value) {
    impl_->observations_per_tree_fraction = value;
}    
void descriptor_base::set_impurity_threshold_impl(const double value) {
    impl_->impurity_threshold = value;
}    
void descriptor_base::set_min_weight_fraction_in_leaf_node_impl(const double value) {
    impl_->min_weight_fraction_in_leaf_node = value;
}    
void descriptor_base::set_min_impurity_decrease_in_split_node_impl(const double value) {
    impl_->min_impurity_decrease_in_split_node = value;
}    

void descriptor_base::set_n_trees_impl(const std::int64_t value) {
    impl_->n_trees = value;
}    
void descriptor_base::set_features_per_node_impl(const std::int64_t value) {
    impl_->features_per_node = value;
}    
void descriptor_base::set_max_tree_depth_impl(const std::int64_t value) {
    impl_->max_tree_depth = value;
}    
void descriptor_base::set_min_observations_in_leaf_node_impl(const std::int64_t value) {
    impl_->min_observations_in_leaf_node = value;
}    
void descriptor_base::set_seed_impl(const std::int64_t value) {
    impl_->seed = value;
}    
void descriptor_base::set_min_observations_in_split_node_impl(const std::int64_t value) {
    impl_->min_observations_in_split_node = value;
}    
void descriptor_base::set_max_leaf_nodes_impl(const std::int64_t value) {
    impl_->max_leaf_nodes = value;
}    

void descriptor_base::set_results_to_compute_impl(const std::uint64_t value) {
    impl_->results_to_compute = value;
}    

void descriptor_base::set_memory_saving_mode_impl(const bool value) {
    impl_->memory_saving_mode = value;
}    
void descriptor_base::set_bootstrap_impl(const bool value) {
    impl_->bootstrap = value;
}    

void descriptor_base::set_variable_importance_mode_impl(const variable_importance_mode value) {
    impl_->variable_importance_mode_value = value;
}    
void descriptor_base::set_voting_method_impl(const voting_method value) {
    impl_->voting_method_value = value;
}    
/* model */
model::model() : impl_(new model_impl{}) {}

} // namespace oneapi::dal::decision_forest
