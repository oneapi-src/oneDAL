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
#include "oneapi/dal/algo/decision_forest/detail/model_impl.hpp"

namespace oneapi::dal::decision_forest {

class detail::descriptor_impl : public base {
  public:
    double observations_per_tree_fraction = 1.0;
    double impurity_threshold = 0.0;
    double min_weight_fraction_in_leaf_node = 0.0;
    double min_impurity_decrease_in_split_node = 0.0;

    std::int64_t class_count = 1;
    std::int64_t tree_count = 100;
    std::int64_t features_per_node = 0;
    std::int64_t max_tree_depth = 0;
    std::int64_t min_observations_in_leaf_node = 0;
    std::int64_t min_observations_in_split_node = 2;
    std::int64_t max_leaf_nodes = 0;

    std::uint64_t train_results_to_compute = 0;
    std::uint64_t infer_results_to_compute = 0;

    bool memory_saving_mode = false;
    bool bootstrap = true;

    // engine field

    variable_importance_mode variable_importance_mode_value = variable_importance_mode::none;
    voting_method voting_method_value = voting_method::weighted ;
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

std::int64_t descriptor_base::get_class_count() const {
    return impl_->class_count;
}    
std::int64_t descriptor_base::get_tree_count() const {
    return impl_->tree_count;
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
std::int64_t descriptor_base::get_min_observations_in_split_node() const {
    return impl_->min_observations_in_split_node;
}    
std::int64_t descriptor_base::get_max_leaf_nodes() const {
    return impl_->max_leaf_nodes;
}    

std::uint64_t descriptor_base::get_train_results_to_compute() const {
    return impl_->train_results_to_compute;
}    
std::uint64_t descriptor_base::get_infer_results_to_compute() const {
    return impl_->infer_results_to_compute;
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

void descriptor_base::set_observations_per_tree_fraction_impl(double value) {
    impl_->observations_per_tree_fraction = value;
}    
void descriptor_base::set_impurity_threshold_impl(double value) {
    impl_->impurity_threshold = value;
}    
void descriptor_base::set_min_weight_fraction_in_leaf_node_impl(double value) {
    impl_->min_weight_fraction_in_leaf_node = value;
}    
void descriptor_base::set_min_impurity_decrease_in_split_node_impl(double value) {
    impl_->min_impurity_decrease_in_split_node = value;
}    

void descriptor_base::set_class_count_impl(std::int64_t value) {
    impl_->class_count = value;
}    
void descriptor_base::set_tree_count_impl(std::int64_t value) {
    impl_->tree_count = value;
}    
void descriptor_base::set_features_per_node_impl(std::int64_t value) {
    impl_->features_per_node = value;
}    
void descriptor_base::set_max_tree_depth_impl(std::int64_t value) {
    impl_->max_tree_depth = value;
}    
void descriptor_base::set_min_observations_in_leaf_node_impl(std::int64_t value) {
    impl_->min_observations_in_leaf_node = value;
}    
void descriptor_base::set_min_observations_in_split_node_impl(std::int64_t value) {
    impl_->min_observations_in_split_node = value;
}    
void descriptor_base::set_max_leaf_nodes_impl(std::int64_t value) {
    impl_->max_leaf_nodes = value;
}    

void descriptor_base::set_train_results_to_compute_impl(std::uint64_t value) {
    impl_->train_results_to_compute = value;
}    
void descriptor_base::set_infer_results_to_compute_impl(std::uint64_t value) {
    impl_->infer_results_to_compute = value;
}    

void descriptor_base::set_memory_saving_mode_impl(bool value) {
    impl_->memory_saving_mode = value;
}    
void descriptor_base::set_bootstrap_impl(bool value) {
    impl_->bootstrap = value;
}    

void descriptor_base::set_variable_importance_mode_impl(variable_importance_mode value) {
    impl_->variable_importance_mode_value = value;
}    
void descriptor_base::set_voting_method_impl(voting_method value) {
    impl_->voting_method_value = value;
}    

/* model implementation */
model::model() : impl_(new detail::model_impl{}) {}
model::model(const model::pimpl& impl) : impl_(impl) {}

std::int64_t model::get_tree_count() const { return impl_->get_tree_count(); }
std::int64_t model::get_class_count() const { return impl_->get_class_count(); }
void model::clear() { impl_->clear(); }

} // namespace oneapi::dal::decision_forest
