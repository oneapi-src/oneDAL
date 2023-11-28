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

#include "oneapi/dal/util/result_option_id.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::pca {

namespace task {
namespace v1 {

/// Tag-type that parameterizes entities used for solving
/// :capterm:`dimensionality reduction problem <dimensionality reduction>`.
struct dim_reduction {};
/// Alias tag-type for dimensionality reduction task.
using by_default = dim_reduction;
} // namespace v1

using v1::dim_reduction;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {
/// Tag-type that denotes :ref:`Covariance <pca_t_math_cov>` computational
/// method.
struct cov {};

struct precomputed {};

/// Tag-type that denotes :ref:`SVD <pca_t_math_svd>` computational method.
struct svd {};

/// Alias tag-type for :ref:`Covariance <pca_t_math_cov>` computational
/// method.
using by_default = cov;
} // namespace v1

using v1::cov;
using v1::precomputed;
using v1::svd;
using v1::by_default;

} // namespace method

/// Represents result option flag
/// Behaves like a regular :expr`enum`.
class result_option_id : public result_option_id_base {
public:
    constexpr result_option_id() = default;
    constexpr explicit result_option_id(const result_option_id_base& base)
            : result_option_id_base{ base } {}
};

namespace detail {

ONEDAL_EXPORT result_option_id get_eigenvectors_id();
ONEDAL_EXPORT result_option_id get_eigenvalues_id();
ONEDAL_EXPORT result_option_id get_variances_id();
ONEDAL_EXPORT result_option_id get_means_id();

} // namespace detail

/// Result options are used to define
/// what should an algorithm return
namespace result_options {

/// Return eigenvectors
const inline result_option_id eigenvectors = detail::get_eigenvectors_id();
/// Return eigenvalues
const inline result_option_id eigenvalues = detail::get_eigenvalues_id();
/// Return variances
const inline result_option_id vars = detail::get_variances_id();
/// Return means
const inline result_option_id means = detail::get_means_id();

} // namespace result_options

namespace detail {
namespace v1 {
struct descriptor_tag {};

template <typename Task>
class descriptor_impl;

template <typename Task>
class model_impl;

template <typename Float>
constexpr bool is_valid_float_v = dal::detail::is_one_of_v<Float, float, double>;

template <typename Method>
constexpr bool is_valid_method_v =
    dal::detail::is_one_of_v<Method, method::cov, method::svd, method::precomputed>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::dim_reduction>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;

    descriptor_base();
    bool whiten() const;
    bool do_mean_centering() const;
    bool is_scaled() const;
    bool do_scale() const;
    bool is_mean_centered() const;
    bool get_deterministic() const;
    std::int64_t get_component_count() const;

    result_option_id get_result_options() const;

protected:
    void set_whiten_impl(bool value);
    void set_do_mean_centering_impl(bool value);
    void set_is_scaled_impl(bool value);
    void set_do_scale_impl(bool value);
    void set_is_mean_centered_impl(bool value);
    void set_deterministic_impl(bool value);
    void set_component_count_impl(std::int64_t value);
    void set_result_options_impl(const result_option_id& value);

private:
    dal::detail::pimpl<descriptor_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor_tag;
using v1::descriptor_impl;
using v1::model_impl;
using v1::descriptor_base;

using v1::is_valid_float_v;
using v1::is_valid_method_v;
using v1::is_valid_task_v;

} // namespace detail

namespace v1 {

/// @tparam Float  The floating-point type that the algorithm uses for
///                intermediate computations. Can be :expr:`float` or
///                :expr:`double`.
/// @tparam Method Tag-type that specifies an implementation of algorithm. Can
///                be :expr:`method::cov` or :expr:`method::svd`.
/// @tparam Task   Tag-type that specifies type of the problem to solve. Can
///                be :expr:`task::dim_reduction`.
template <typename Float = float,
          typename Method = method::by_default,
          typename Task = task::by_default>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;

    /// Creates a new instance of the class with the given :literal:`component_count`
    /// property value
    explicit descriptor(std::int64_t component_count = 0) {
        set_component_count(component_count);
    }

    /// The number of principal components $r$. If it is zero, the algorithm
    /// computes the eigenvectors for all features, $r = p$.
    /// @remark default = 0
    /// @invariant :expr:`component_count >= 0`
    std::int64_t get_component_count() const {
        return base_t::get_component_count();
    }

    auto& set_component_count(std::int64_t value) {
        base_t::set_component_count_impl(value);
        return *this;
    }

    /// Specifies whether the algorithm applies the sign-flip technique.
    /// If it is `true`, the directions of the eigenvectors must be deterministic.
    /// @remark default = true
    bool get_deterministic() const {
        return base_t::get_deterministic();
    }

    auto& set_deterministic(bool value) {
        base_t::set_deterministic_impl(value);
        return *this;
    }
    bool whiten() const {
        return base_t::whiten();
    }

    auto& set_whiten(bool value) {
        base_t::set_whiten_impl(value);
        return *this;
    }

    bool do_mean_centering() const {
        return base_t::do_mean_centering();
    }

    auto& set_do_mean_centering(bool value) {
        base_t::set_do_mean_centering_impl(value);
        return *this;
    }

    bool is_scaled() const {
        return base_t::is_scaled();
    }

    auto& set_is_scaled(bool value) {
        base_t::set_is_scaled_impl(value);
        return *this;
    }
    bool do_scale() const {
        return base_t::do_scale();
    }

    auto& set_do_scale(bool value) {
        base_t::set_do_scale_impl(value);
        return *this;
    }
    bool is_mean_centered() const {
        return base_t::is_mean_centered();
    }

    auto& set_is_mean_centered(bool value) {
        base_t::set_is_mean_centered_impl(value);
        return *this;
    }
    /// Choose which results should be computed and returned.
    result_option_id get_result_options() const {
        return base_t::get_result_options();
    }

    auto& set_result_options(const result_option_id& value) {
        base_t::set_result_options_impl(value);
        return *this;
    }
};

/// @tparam Task Tag-type that specifies type of the problem to solve. Can
///              be :expr:`task::dim_reduction`.
template <typename Task = task::by_default>
class model : public base {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;
    friend dal::detail::serialization_accessor;

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    model();

    /// An $r \\times p$ table with the eigenvectors. Each row contains one
    /// eigenvector.
    /// @remark default = table{}
    const table& get_eigenvectors() const;

    auto& set_eigenvectors(const table& value) {
        set_eigenvectors_impl(value);
        return *this;
    }
    const table& get_means() const;

    auto& set_means(const table& value) {
        set_means_impl(value);
        return *this;
    }

protected:
    void set_eigenvectors_impl(const table&);
    void set_means_impl(const table&);
    void set_variances_impl(const table&);
    void set_eigenvalues_impl(const table&);
    void set_singular_values_impl(const table&);

private:
    void serialize(dal::detail::output_archive& ar) const;
    void deserialize(dal::detail::input_archive& ar);

    dal::detail::pimpl<detail::model_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor;
using v1::model;

} // namespace oneapi::dal::pca
