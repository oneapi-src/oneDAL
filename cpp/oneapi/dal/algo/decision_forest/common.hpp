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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/util/common.hpp"

namespace oneapi::dal::decision_forest {

namespace task {
namespace v1 {
struct classification {};
struct regression {};
using by_default = classification;
} // namespace v1

using v1::classification;
using v1::regression;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
struct dense {};
struct hist {};
using by_default = dense;
} // namespace v1

using v1::dense;
using v1::hist;
using v1::by_default;

} // namespace method

namespace v1 {

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

enum class error_metric_mode : std::uint64_t {
    none = 0x00000000ULL,
    out_of_bag_error = 0x00000001ULL,
    out_of_bag_error_per_observation = 0x00000002ULL
};

enum class infer_mode : std::uint64_t {
    class_labels = 0x00000001ULL, /*!< Numeric table of size n x 1 with the predicted labels >*/
    class_probabilities =
        0x00000002ULL /*!< Numeric table of size n x p with the predicted class probabilities for each observation >*/
};

enum class voting_mode { weighted, unweighted };

inline infer_mode operator|(infer_mode value_left, infer_mode value_right) {
    return bitwise_or(value_left, value_right);
}

inline infer_mode& operator|=(infer_mode& value_left, infer_mode value_right) {
    value_left = value_left | value_right;
    return value_left;
}

inline infer_mode operator&(infer_mode value_left, infer_mode value_right) {
    return bitwise_and(value_left, value_right);
}

inline infer_mode& operator&=(infer_mode& value_left, infer_mode value_right) {
    value_left = value_left & value_right;
    return value_left;
}

inline error_metric_mode operator&(error_metric_mode value_left, error_metric_mode value_right) {
    return bitwise_and(value_left, value_right);
}

inline error_metric_mode& operator&=(error_metric_mode& value_left, error_metric_mode value_right) {
    value_left = value_left & value_right;
    return value_left;
}

inline error_metric_mode operator|(error_metric_mode value_left, error_metric_mode value_right) {
    return bitwise_or(value_left, value_right);
}

inline error_metric_mode& operator|=(error_metric_mode& value_left, error_metric_mode value_right) {
    value_left = value_left | value_right;
    return value_left;
}

} // namespace v1

using v1::variable_importance_mode;
using v1::error_metric_mode;
using v1::infer_mode;
using v1::voting_mode;

namespace detail {
namespace v1 {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Task>
class model_impl;

template <typename T>
using enable_if_classification_t =
    std::enable_if_t<std::is_same_v<std::decay_t<T>, task::classification>>;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::dense, method::hist>;

template <typename Task>
constexpr bool is_valid_task_v =
    dal::detail::is_one_of_v<Task, task::classification, task::regression>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

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
    std::int64_t get_max_bins() const;
    std::int64_t get_min_bin_size() const;

    bool get_memory_saving_mode() const;
    bool get_bootstrap() const;

    error_metric_mode get_error_metric_mode() const;
    variable_importance_mode get_variable_importance_mode() const;

    template <typename T = Task, typename = enable_if_classification_t<T>>
    std::int64_t get_class_count() const {
        return get_class_count_impl();
    }

    template <typename T = Task, typename = enable_if_classification_t<T>>
    infer_mode get_infer_mode() const {
        return get_infer_mode_impl();
    }

    template <typename T = Task, typename = enable_if_classification_t<T>>
    voting_mode get_voting_mode() const {
        return get_voting_mode_impl();
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
    void set_max_bins_impl(std::int64_t value);
    void set_min_bin_size_impl(std::int64_t value);

    void set_memory_saving_mode_impl(bool value);
    void set_bootstrap_impl(bool value);

    void set_error_metric_mode_impl(error_metric_mode value);
    void set_variable_importance_mode_impl(variable_importance_mode value);

    void set_class_count_impl(std::int64_t value);
    void set_infer_mode_impl(infer_mode value);
    void set_voting_mode_impl(voting_mode value);

    std::int64_t get_class_count_impl() const;
    infer_mode get_infer_mode_impl() const;
    voting_mode get_voting_mode_impl() const;

private:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor_tag;
using v1::descriptor_impl;
using v1::model_impl;
using v1::descriptor_base;

using v1::enable_if_classification_t;
using v1::is_valid_float_v;
using v1::is_valid_method_v;
using v1::is_valid_task_v;

} // namespace detail

namespace v1 {

template <typename Float = detail::descriptor_base<>::float_t,
          typename Method = detail::descriptor_base<>::method_t,
          typename Task = detail::descriptor_base<>::task_t>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

    auto& set_observations_per_tree_fraction(double value) {
        base_t::set_observations_per_tree_fraction_impl(value);
        return *this;
    }

    auto& set_impurity_threshold(double value) {
        base_t::set_impurity_threshold_impl(value);
        return *this;
    }

    auto& set_min_weight_fraction_in_leaf_node(double value) {
        base_t::set_min_weight_fraction_in_leaf_node_impl(value);
        return *this;
    }

    auto& set_min_impurity_decrease_in_split_node(double value) {
        base_t::set_min_impurity_decrease_in_split_node_impl(value);
        return *this;
    }

    auto& set_tree_count(std::int64_t value) {
        base_t::set_tree_count_impl(value);
        return *this;
    }

    auto& set_features_per_node(std::int64_t value) {
        base_t::set_features_per_node_impl(value);
        return *this;
    }

    auto& set_max_tree_depth(std::int64_t value) {
        base_t::set_max_tree_depth_impl(value);
        return *this;
    }

    auto& set_min_observations_in_leaf_node(std::int64_t value) {
        base_t::set_min_observations_in_leaf_node_impl(value);
        return *this;
    }

    auto& set_min_observations_in_split_node(std::int64_t value) {
        base_t::set_min_observations_in_split_node_impl(value);
        return *this;
    }

    auto& set_max_leaf_nodes(std::int64_t value) {
        base_t::set_max_leaf_nodes_impl(value);
        return *this;
    }

    auto& set_max_bins(std::int64_t value) {
        base_t::set_max_bins_impl(value);
        return *this;
    }

    auto& set_min_bin_size(std::int64_t value) {
        base_t::set_min_bin_size_impl(value);
        return *this;
    }

    auto& set_error_metric_mode(error_metric_mode value) {
        base_t::set_error_metric_mode_impl(value);
        return *this;
    }

    auto& set_memory_saving_mode(bool value) {
        base_t::set_memory_saving_mode_impl(value);
        return *this;
    }

    auto& set_bootstrap(bool value) {
        base_t::set_bootstrap_impl(value);
        return *this;
    }

    auto& set_variable_importance_mode(variable_importance_mode value) {
        base_t::set_variable_importance_mode_impl(value);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_class_count(std::int64_t value) {
        base_t::set_class_count_impl(value);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_infer_mode(infer_mode value) {
        base_t::set_infer_mode_impl(value);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_voting_mode(voting_mode value) {
        base_t::set_voting_mode_impl(value);
        return *this;
    }
};

template <typename Task = task::by_default>
class model : public base {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;

public:
    using task_t = Task;

    model();

    std::int64_t get_tree_count() const;

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    std::int64_t get_class_count() const {
        return get_class_count_impl();
    }

protected:
    std::int64_t get_class_count_impl() const;

private:
    explicit model(const std::shared_ptr<detail::model_impl<Task>>& impl);
    dal::detail::pimpl<detail::model_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor;
using v1::model;

} // namespace oneapi::dal::decision_forest
