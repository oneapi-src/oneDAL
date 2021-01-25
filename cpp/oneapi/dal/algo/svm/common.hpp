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

#pragma once

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/algo/svm/detail/kernel_function.hpp"

namespace oneapi::dal::svm {

namespace task {
namespace v1 {

/// Tag-type that parameterizes entities that are used for solving
/// :capterm:`classification problem <classification>`.
struct classification {};

/// Alias tag-type for classification task.
using by_default = classification;
} // namespace v1

using v1::classification;
using v1::by_default;

} // namespace task

namespace method {
namespace v1 {

/// Tag-type that denotes `Thunder <svm_t_math_thunder_>`_ computational
/// method.
struct thunder {};

/// Tag-type that denotes `SMO <svm_t_math_smo_>`_ computational
/// method.
struct smo {};

/// Alias tag-type for `Thunder <svm_t_math_thunder_>`_ computational
/// method.
using by_default = thunder;
} // namespace v1

using v1::thunder;
using v1::smo;
using v1::by_default;

} // namespace method

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
constexpr bool is_valid_method_v = dal::detail::is_one_of_v<Method, method::smo, method::thunder>;

template <typename Task>
constexpr bool is_valid_task_v = dal::detail::is_one_of_v<Task, task::classification>;

template <typename Kernel>
constexpr bool is_valid_kernel_v =
    dal::detail::is_tag_one_of_v<Kernel,
                                 linear_kernel::detail::descriptor_tag,
                                 rbf_kernel::detail::descriptor_tag>;

template <typename Task = task::by_default>
class descriptor_base : public base {
    static_assert(is_valid_task_v<Task>);
    friend detail::kernel_function_accessor;

public:
    using tag_t = descriptor_tag;
    using float_t = float;
    using method_t = method::by_default;
    using task_t = Task;
    using kernel_t = linear_kernel::descriptor<float_t>;

    /// The upper bound in constraints of the quadratic optimization problem. $C$
    /// @invariant :expr:`c > 0`
    /// @remark default = 1.0
    double get_c() const;

    /// The maximum number of iterations $T$
    /// @invariant :expr:`max_iteration_count >= 0`
    /// @remark default = 100000
    std::int64_t get_max_iteration_count() const;

    /// The threshold $\\varepsilon$ for the stop condition
    /// @invariant :expr:`accuracy_threshold >= 0.0`
    /// @remark default = 0.0
    double get_accuracy_threshold() const;

    /// The size of cache in megabytes for storing values of the kernel matrix.
    /// @invariant :expr:`cache_size >= 0.0`
    /// @remark default = 200.0
    double get_cache_size() const;

    /// The parameter of the WSS scheme $\\tau$.
    /// @invariant :expr:`tau >= 0.0`
    /// @remark default = 1e-6
    double get_tau() const;

    /// A flag that enables the use of a shrinking optimization technique. Used with :expr:`oneapi::dal::svm::method::v1::thunder` split-finding method only.
    /// @remark default = true
    bool get_shrinking() const;

protected:
    explicit descriptor_base(const detail::kernel_function_ptr& kernel);

    void set_c_impl(double);
    void set_accuracy_threshold_impl(double);
    void set_max_iteration_count_impl(std::int64_t);
    void set_cache_size_impl(double);
    void set_tau_impl(double);
    void set_shrinking_impl(bool);

    void set_kernel_impl(const detail::kernel_function_ptr&);
    const detail::kernel_function_ptr& get_kernel_impl() const;

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
using v1::is_valid_kernel_v;

} // namespace detail

namespace v1 {

/// @tparam Float  The floating-point type that the algorithm uses for
///                intermediate computations. Can be :expr:`float` or
///                :expr:`double`.
/// @tparam Method Tag-type that specifies an implementation of algorithm. Can
///                be :expr:`method::v1::thunder` or :expr:`method::v1::smo`.
/// @tparam Task   Tag-type that specifies the type of the problem to solve. Can
///                be :expr:`task::v1::classification`.
template <typename Float = detail::descriptor_base<>::float_t,
          typename Method = detail::descriptor_base<>::method_t,
          typename Task = detail::descriptor_base<>::task_t,
          typename Kernel = detail::descriptor_base<>::kernel_t>
class descriptor : public detail::descriptor_base<Task> {
    static_assert(detail::is_valid_float_v<Float>);
    static_assert(detail::is_valid_method_v<Method>);
    static_assert(detail::is_valid_task_v<Task>);
    static_assert(detail::is_valid_kernel_v<Kernel>,
                  "Custom kernel for SVM is not supported. "
                  "Use one of the predefined kernels.");

    using base_t = detail::descriptor_base<Task>;

public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;
    using kernel_t = Kernel;

    /// Creates a new instance of the class with the given descriptor of the kernel function
    /// @remark default = :literal:`kernel`
    explicit descriptor(const Kernel& kernel = kernel_t{})
            : base_t(std::make_shared<detail::kernel_function<Kernel>>(kernel)) {}

    /// The descriptor of kernel function `K(x,y)`. Can be :expr:`linear_kernel::desc` or
    /// :expr:`rbf_kernel::desc`.
    /// @remark default = :literal:`kernel`
    const Kernel& get_kernel() const {
        using kf_t = detail::kernel_function<Kernel>;
        const auto kf = std::static_pointer_cast<kf_t>(base_t::get_kernel_impl());
        return kf->get_kernel();
    }

    auto& set_kernel(const Kernel& kernel) {
        base_t::set_kernel_impl(std::make_shared<detail::kernel_function<Kernel>>(kernel));
        return *this;
    }

    auto& set_c(double value) {
        base_t::set_c_impl(value);
        return *this;
    }

    auto& set_accuracy_threshold(double value) {
        base_t::set_accuracy_threshold_impl(value);
        return *this;
    }

    auto& set_max_iteration_count(std::int64_t value) {
        base_t::set_max_iteration_count_impl(value);
        return *this;
    }

    auto& set_cache_size(double value) {
        base_t::set_cache_size_impl(value);
        return *this;
    }

    auto& set_tau(double value) {
        base_t::set_tau_impl(value);
        return *this;
    }

    auto& set_shrinking(bool value) {
        base_t::set_shrinking_impl(value);
        return *this;
    }
};

/// @tparam Task Tag-type that specifies the type of the problem to solve. Can
///              be :expr:`task::v1::classification`.
template <typename Task = task::by_default>
class model : public base {
    static_assert(detail::is_valid_task_v<Task>);
    friend dal::detail::pimpl_accessor;

public:
    using task_t = Task;

    /// Creates a new instance of the class with the default property values.
    model();

    /// The number of support vectors
    /// @remark default = 0
    /// @invariant :expr:`support_vector_count >= 0`
    std::int64_t get_support_vector_count() const;

    /// A $nsv \\times p$ table containing support vectors.
    /// Where $nsv$ - number of support vectors.
    /// @remark default = table{}
    const table& get_support_vectors() const;

    auto& set_support_vectors(const table& value) {
        set_support_vectors_impl(value);
        return *this;
    }

    /// A $nsv \\times 1$ table containing coefficients of Lagrange multiplier
    /// @remark default = table{}
    const table& get_coeffs() const;

    auto& set_coeffs(const table& value) {
        set_coeffs_impl(value);
        return *this;
    }

    /// The bias
    /// @remark default = 0.0
    double get_bias() const;

    auto& set_bias(double value) {
        set_bias_impl(value);
        return *this;
    }

    /// The first unique value in class labels
    /// @remark default = 0
    std::int64_t get_first_class_label() const;

    auto& set_first_class_label(std::int64_t value) {
        set_first_class_label_impl(value);
        return *this;
    }

    /// The second unique value in class labels
    /// @remark default = 0
    std::int64_t get_second_class_label() const;

    auto& set_second_class_label(std::int64_t value) {
        set_second_class_label_impl(value);
        return *this;
    }

protected:
    void set_support_vectors_impl(const table&);
    void set_coeffs_impl(const table&);
    void set_bias_impl(double);
    void set_first_class_label_impl(std::int64_t);
    void set_second_class_label_impl(std::int64_t);

private:
    dal::detail::pimpl<detail::model_impl<Task>> impl_;
};

} // namespace v1

using v1::descriptor;
using v1::model;

} // namespace oneapi::dal::svm
