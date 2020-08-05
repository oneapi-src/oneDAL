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

namespace task {
struct classification {};
struct regression {};
using by_default = classification;
} // namespace task

namespace detail {
struct tag {};

template <typename Task = task::by_default>
class descriptor_impl;

template <typename Task = task::by_default>
class model_impl;
} // namespace detail

namespace method {
struct dense {};
struct hist {};
using by_default = dense;
} // namespace method

enum class variable_importance_mode {
    none, /* Do not compute */
    mdi, /* Mean Decrease Impurity.
                       Computed as the sum of weighted impurity decreases for all nodes where the variable is used,
                       averaged over all trees in the forest */
    mda_raw, /* Mean Decrease Accuracy (permutation importance).
                       For each tree, the prediction error on the out-of-bag portion of the data is computed
                       (error rate for classification, MSE for regression).
                       The same is done after permuting each predictor variable.
                       The difference between the two are then averaged over all trees. */
    mda_scaled /* Mean Decrease Accuracy (permutation importance).
                       This is MDA_Raw value scaled by its standard deviation. */
};

enum class train_result_to_compute : std::uint64_t {
    compute_out_of_bag_error                 = 0x00000001ULL,
    compute_out_of_bag_error_per_observation = 0x00000002ULL
};

enum class infer_result_to_compute : std::uint64_t {
    compute_class_labels =
        0x00000001ULL, /*!< Numeric table of size n x 1 with the predicted labels >*/
    compute_class_probabilities =
        0x00000002ULL /*!< Numeric table of size n x p with the predicted class probabilities for each observation >*/
};

enum class voting_method { weighted, unweighted };

template <typename Task = task::by_default>
class descriptor_base : public base {
    friend dal::detail::pimpl_accessor;

public:
    using tag_t    = detail::tag;
    using float_t  = float;
    using task_t   = Task;
    using method_t = method::by_default;
    using pimpl    = typename dal::detail::pimpl<detail::descriptor_impl<task_t>>;
    template <typename T>
    using is_classification_t =
        std::enable_if_t<std::is_same_v<T, std::decay_t<task::classification>>>;

    descriptor_base();

    double get_observations_per_tree_fraction() const;
    double get_impurity_threshold() const;
    double get_min_weight_fraction_in_leaf_node() const;
    double get_min_impurity_decrease_in_split_node() const;

    std::int64_t get_tree_count() const;
    std::int64_t get_features_per_node() const;
    std::int64_t get_max_tree_depth() const;
    std::int64_t get_min_observations_in_leaf_node() const;
    std::int64_t get_min_observations_in_split_node() const;
    std::int64_t get_max_leaf_nodes() const;

    bool get_memory_saving_mode() const;
    bool get_bootstrap() const;

    variable_importance_mode get_variable_importance_mode() const;

    /* classification specific methods */
    template <typename T = Task, typename = is_classification_t<T>>
    std::int64_t get_class_count() const {
        return get_class_count_impl();
    }

    template <typename T = Task, typename = is_classification_t<T>>
    voting_method get_voting_method() const {
        return get_voting_method_impl();
    }

protected:
    void set_observations_per_tree_fraction_impl(double value);
    void set_impurity_threshold_impl(double value);
    void set_min_weight_fraction_in_leaf_node_impl(double value);
    void set_min_impurity_decrease_in_split_node_impl(double value);

    void set_tree_count_impl(std::int64_t value);
    void set_features_per_node_impl(std::int64_t value);
    void set_max_tree_depth_impl(std::int64_t value);
    void set_min_observations_in_leaf_node_impl(std::int64_t value);
    void set_min_observations_in_split_node_impl(std::int64_t value);
    void set_max_leaf_nodes_impl(std::int64_t value);

    void set_memory_saving_mode_impl(bool value);
    void set_bootstrap_impl(bool value);

    void set_variable_importance_mode_impl(variable_importance_mode value);

    /* classification specific methods */
    std::int64_t get_class_count_impl() const;
    voting_method get_voting_method_impl() const;

    void set_class_count_impl(std::int64_t value);
    void set_voting_method_impl(voting_method value);

private:
    pimpl impl_;
};
/* task descriptor */
template <typename Float  = descriptor_base<task::by_default>::float_t,
          typename Task   = task::by_default,
          typename Method = descriptor_base<task::by_default>::method_t>
class descriptor : public descriptor_base<Task> {
public:
    using float_t  = Float;
    using task_t   = Task;
    using method_t = Method;
    using parent   = descriptor_base<Task>;

    template <typename T>
    using is_classification_t =
        std::enable_if_t<std::is_same_v<T, std::decay_t<task::classification>>>;

    auto& set_observations_per_tree_fraction(double value) {
        parent::set_observations_per_tree_fraction_impl(value);
        return *this;
    }
    auto& set_impurity_threshold(double value) {
        parent::set_impurity_threshold_impl(value);
        return *this;
    }
    auto& set_min_weight_fraction_in_leaf_node(double value) {
        parent::set_min_weight_fraction_in_leaf_node_impl(value);
        return *this;
    }
    auto& set_min_impurity_decrease_in_split_node(double value) {
        parent::set_min_impurity_decrease_in_split_node_impl(value);
        return *this;
    }

    auto& set_tree_count(std::int64_t value) {
        parent::set_tree_count_impl(value);
        return *this;
    }
    auto& set_features_per_node(std::int64_t value) {
        parent::set_features_per_node_impl(value);
        return *this;
    }
    auto& set_max_tree_depth(std::int64_t value) {
        parent::set_max_tree_depth_impl(value);
        return *this;
    }
    auto& set_min_observations_in_leaf_node(std::int64_t value) {
        parent::set_min_observations_in_leaf_node_impl(value);
        return *this;
    }
    auto& set_min_observations_in_split_node(std::int64_t value) {
        parent::set_min_observations_in_split_node_impl(value);
        return *this;
    }
    auto& set_max_leaf_nodes(std::int64_t value) {
        parent::set_max_leaf_nodes_impl(value);
        return *this;
    }

    auto& set_memory_saving_mode(bool value) {
        parent::set_memory_saving_mode_impl(value);
        return *this;
    }
    auto& set_bootstrap(bool value) {
        parent::set_bootstrap_impl(value);
        return *this;
    }

    auto& set_variable_importance_mode(variable_importance_mode value) {
        parent::set_variable_importance_mode_impl(value);
        return *this;
    }
    /* classification specific methods */

    template <typename T = Task, typename = is_classification_t<T>>
    auto& set_class_count(std::int64_t value) {
        parent::set_class_count_impl(value);
        return *this;
    }

    template <typename T = Task, typename = is_classification_t<T>>
    auto& set_voting_method(voting_method value) {
        parent::set_voting_method_impl(value);
        return *this;
    }
};

/* model declaration */
template <typename Task = task::by_default>
class model : public base {
    friend dal::detail::pimpl_accessor;

public:
    using task_t = Task;
    using pimpl  = typename dal::detail::pimpl<detail::model_impl<Task>>;
    template <typename T>
    using is_classification_t =
        std::enable_if_t<std::is_same_v<T, std::decay_t<task::classification>>>;

    model();

    std::int64_t get_tree_count() const;
    void clear();

    template <typename T = Task, typename = is_classification_t<T>>
    std::int64_t get_class_count() const {
        return get_class_count_impl();
    }

protected:
    std::int64_t get_class_count_impl() const;

private:
    explicit model(const pimpl& impl);
    pimpl impl_;
};

} // namespace oneapi::dal::decision_forest

ONEAPI_DAL_EXPORT std::uint64_t operator|(
    oneapi::dal::decision_forest::train_result_to_compute value_left,
    oneapi::dal::decision_forest::train_result_to_compute value_right);
ONEAPI_DAL_EXPORT std::uint64_t operator|(
    oneapi::dal::decision_forest::infer_result_to_compute value_left,
    oneapi::dal::decision_forest::infer_result_to_compute value_right);
