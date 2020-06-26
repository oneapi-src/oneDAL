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

#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::decision_forest {

namespace detail {
struct tag {};
class descriptor_impl;
class model_impl;
} // namespace detail

namespace task {
struct classification {};
struct regression {};
using by_default = classification;
} // namespace task

namespace method {
struct default_dense {};
struct hist {};
using by_default = default_dense;
} // namespace method

enum class variable_importance_mode
{
    none,      /* Do not compute */
    mdi,       /* Mean Decrease Impurity.
                       Computed as the sum of weighted impurity decreases for all nodes where the variable is used,
                       averaged over all trees in the forest */
    mda_raw,   /* Mean Decrease Accuracy (permutation importance).
                       For each tree, the prediction error on the out-of-bag portion of the data is computed
                       (error rate for classification, MSE for regression).
                       The same is done after permuting each predictor variable.
                       The difference between the two are then averaged over all trees. */
    mda_scaled /* Mean Decrease Accuracy (permutation importance).
                       This is MDA_Raw value scaled by its standard deviation. */
};

enum class result_to_compute_id
{
    compute_out_of_bag_error                 = 0x00000001ULL,
    compute_out_of_bag_error_per_observation = 0x00000002ULL
};

enum class voting_method
{
    weighted = 0,
    unweighted,
    last_result_id = unweighted
};

class descriptor_base : public base {
  public:
    using tag_t = detail::tag;
    using float_t = float;
    using task_t   = task::by_default;
    using method_t = method::by_default;

    descriptor_base();

    //engines::EnginePtr engine;             /*!< Engine for the random numbers generator used by the algorithms */

    double get_observations_per_tree_fraction() const;
    double get_impurity_threshold() const;
    double get_min_weight_fraction_in_leaf_node() const;
    double get_min_impurity_decrease_in_split_node() const;

    std::int64_t get_n_classes() const;
    std::int64_t get_n_trees() const;
    std::int64_t get_features_per_node() const;
    std::int64_t get_max_tree_depth() const;
    std::int64_t get_min_observations_in_leaf_node() const;
    std::int64_t get_seed() const;
    std::int64_t get_min_observations_in_split_node() const;
    std::int64_t get_max_leaf_nodes() const;

    bool get_memory_saving_mode() const;
    bool get_bootstrap() const;

    std::uint64_t get_results_to_compute() const;

    variable_importance_mode get_variable_importance_mode() const;
    voting_method get_voting_method() const;

  protected:
    void set_observations_per_tree_fraction_impl(const double value);
    void set_impurity_threshold_impl(const double value);
    void set_min_weight_fraction_in_leaf_node_impl(const double value);
    void set_min_impurity_decrease_in_split_node_impl(const double value);

    void set_n_classes_impl(const std::int64_t value);
    void set_n_trees_impl(const std::int64_t value);
    void set_features_per_node_impl(const std::int64_t value);
    void set_max_tree_depth_impl(const std::int64_t value);
    void set_min_observations_in_leaf_node_impl(const std::int64_t value);
    void set_seed_impl(const std::int64_t value);
    void set_min_observations_in_split_node_impl(const std::int64_t value);
    void set_max_leaf_nodes_impl(const std::int64_t value);

    void set_results_to_compute_impl(const std::uint64_t value);

    void set_memory_saving_mode_impl(const bool value);
    void set_bootstrap_impl(const bool value);

    void set_variable_importance_mode_impl(const variable_importance_mode value);
    void set_voting_method_impl(const voting_method value);

    dal::detail::pimpl<detail::descriptor_impl> impl_;
};

template <typename Float  = descriptor_base::float_t,
          typename Task   = descriptor_base::task_t,
          typename Method = descriptor_base::method_t>
class descriptor : public descriptor_base {
public:
    using float_t  = Float;
    using task_t   = Task;
    using method_t = Method;

    auto& set_observations_per_tree_fraction(const double value) {
        set_observations_per_tree_fraction_impl(value);
        return *this;
    }
    auto& set_impurity_threshold(const double value) {
        set_impurity_threshold_impl(value);
        return *this;
    }
    auto& set_min_weight_fraction_in_leaf_node(const double value) {
        set_min_weight_fraction_in_leaf_node_impl(value);
        return *this;
    }
    auto& set_min_impurity_decrease_in_split_node(const double value) {
        set_min_impurity_decrease_in_split_node_impl(value);
        return *this;
    }

    auto& set_n_classes(const std::int64_t value) {
        set_n_classes_impl(value);
        return *this;
    }
    auto& set_n_trees(const std::int64_t value) {
        set_n_trees_impl(value);
        return *this;
    }
    auto& set_features_per_node(const std::int64_t value) {
        set_features_per_node_impl(value);
        return *this;
    }
    auto& set_max_tree_depth(const std::int64_t value) {
        set_max_tree_depth_impl(value);
        return *this;
    }
    auto& set_min_observations_in_leaf_node(const std::int64_t value) {
        set_min_observations_in_leaf_node_impl(value);
        return *this;
    }
    auto& set_seed(const std::int64_t value) {
        set_seed_impl(value);
        return *this;
    }
    auto& set_min_observations_in_split_node(const std::int64_t value) {
        set_min_observations_in_split_node_impl(value);
        return *this;
    }
    auto& set_max_leaf_nodes(const std::int64_t value) {
        set_max_leaf_nodes_impl(value);
        return *this;
    }

    auto& set_results_to_compute(const std::uint64_t value) {
        set_results_to_compute_impl(value);
        return *this;
    }

    auto& set_memory_saving_mode(const bool value) {
        set_memory_saving_mode_impl(value);
        return *this;
    }
    auto& set_bootstrap(const bool value) {
        set_bootstrap_impl(value);
        return *this;
    }

    auto& set_variable_importance_mode(const variable_importance_mode value) {
        set_variable_importance_mode_impl(value);
        return *this;
    }
    auto& set_voting_method(const voting_method value) {
        set_voting_method_impl(value);
        return *this;
    }
};

class model : public base {
  friend dal::detail::pimpl_accessor;
  public:
    model();

    dal::detail::pimpl<detail::model_impl> impl_;
};
} // namespace oneapi::dal::decision_forest
