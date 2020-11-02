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

#include "oneapi/dal/algo/linear_kernel.hpp"
#include "oneapi/dal/algo/rbf_kernel.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::svm {

namespace task {
struct classification {};
using by_default = classification;
} // namespace task

namespace detail {
struct tag {};

template <typename Task = task::by_default>
class descriptor_impl;

template <typename Task = task::by_default>
class model_impl;

class kernel_function_impl;

class kernel_function_iface {
public:
    virtual ~kernel_function_iface() {}
    virtual kernel_function_impl *get_impl() const = 0;
};

using kf_iface_ptr = std::shared_ptr<kernel_function_iface>;

template <typename Kernel>
class ONEDAL_EXPORT kernel_function : public base, public kernel_function_iface {
public:
    explicit kernel_function(const Kernel &kernel) : kernel_(kernel) {}

    kernel_function_impl *get_impl() const override {
        return nullptr;
    }

    const Kernel &get_kernel() const {
        return kernel_;
    }

private:
    Kernel kernel_;
    dal::detail::pimpl<kernel_function_impl> impl_;
};

template <typename Float, typename Method>
class kernel_function<linear_kernel::descriptor<Float, Method>> : public base,
                                                                  public kernel_function_iface {
public:
    using kernel_t = linear_kernel::descriptor<Float, Method>;
    explicit kernel_function(const kernel_t &kernel);
    kernel_function_impl *get_impl() const override;

private:
    kernel_t kernel_;
    dal::detail::pimpl<kernel_function_impl> impl_;
};

template <typename Float, typename Method>
class kernel_function<rbf_kernel::descriptor<Float, Method>> : public base,
                                                               public kernel_function_iface {
public:
    using kernel_t = rbf_kernel::descriptor<Float, Method>;
    explicit kernel_function(const kernel_t &kernel);
    kernel_function_impl *get_impl() const override;

private:
    kernel_t kernel_;
    dal::detail::pimpl<kernel_function_impl> impl_;
};

} // namespace detail

namespace method {
struct thunder {};
struct smo {};
using by_default = thunder;
} // namespace method

template <typename Task = task::by_default>
class ONEDAL_EXPORT descriptor_base : public base {
public:
    using tag_t = detail::tag;
    using float_t = float;
    using task_t = Task;
    using method_t = method::by_default;
    using kernel_t = linear_kernel::descriptor<float_t>;

    double get_c() const;
    double get_accuracy_threshold() const;
    std::int64_t get_max_iteration_count() const;
    double get_cache_size() const;
    double get_tau() const;
    bool get_shrinking() const;
    const detail::kf_iface_ptr &get_kernel_impl() const;

protected:
    explicit descriptor_base(const detail::kf_iface_ptr &kernel);

    void set_c_impl(double);
    void set_accuracy_threshold_impl(double);
    void set_max_iteration_count_impl(std::int64_t);
    void set_cache_size_impl(double);
    void set_tau_impl(double);
    void set_shrinking_impl(bool);
    void set_kernel_impl(const detail::kf_iface_ptr &);

    dal::detail::pimpl<detail::descriptor_impl<Task>> impl_;
};

template <typename Float = descriptor_base<task::by_default>::float_t,
          typename Method = descriptor_base<task::by_default>::method_t,
          typename Task = task::by_default,
          typename Kernel = descriptor_base<task::by_default>::kernel_t>
class descriptor : public descriptor_base<Task> {
public:
    using float_t = Float;
    using method_t = Method;
    using task_t = Task;
    using kernel_t = Kernel;

    explicit descriptor(const Kernel &kernel = kernel_t{})
            : descriptor_base<Task>(std::make_shared<detail::kernel_function<Kernel>>(kernel)) {}

    const Kernel &get_kernel() const {
        using kf_t = detail::kernel_function<Kernel>;
        const auto kf = std::static_pointer_cast<kf_t>(descriptor_base<Task>::get_kernel_impl());
        return kf->get_kernel();
    }

    auto &set_kernel(const Kernel &kernel) {
        descriptor_base<Task>::set_kernel_impl(
            std::make_shared<detail::kernel_function<Kernel>>(kernel));
        return *this;
    }

    auto &set_c(double value) {
        descriptor_base<Task>::set_c_impl(value);
        return *this;
    }

    auto &set_accuracy_threshold(double value) {
        descriptor_base<Task>::set_accuracy_threshold_impl(value);
        return *this;
    }

    auto &set_max_iteration_count(std::int64_t value) {
        descriptor_base<Task>::set_max_iteration_count_impl(value);
        return *this;
    }

    auto &set_cache_size(double value) {
        descriptor_base<Task>::set_cache_size_impl(value);
        return *this;
    }

    auto &set_tau(double value) {
        descriptor_base<Task>::set_tau_impl(value);
        return *this;
    }

    auto &set_shrinking(bool value) {
        descriptor_base<Task>::set_shrinking_impl(value);
        return *this;
    }
};

template <typename Task = task::by_default>
class ONEDAL_EXPORT model : public base {
    friend dal::detail::pimpl_accessor;

public:
    using task_t = Task;
    model();

    table get_support_vectors() const;

    auto &set_support_vectors(const table &value) {
        set_support_vectors_impl(value);
        return *this;
    }

    table get_coeffs() const;

    auto &set_coeffs(const table &value) {
        set_coeffs_impl(value);
        return *this;
    }

    double get_bias() const;

    auto &set_bias(double value) {
        set_bias_impl(value);
        return *this;
    }

    std::int64_t get_support_vector_count() const;

    auto &set_support_vector_count(std::int64_t value) {
        set_support_vector_count_impl(value);
        return *this;
    }

    std::int64_t get_first_class_label() const;

    auto &set_first_class_label(std::int64_t value) {
        set_first_class_label_impl(value);
        return *this;
    }

    std::int64_t get_second_class_label() const;

    auto &set_second_class_label(std::int64_t value) {
        set_second_class_label_impl(value);
        return *this;
    }

private:
    void set_support_vectors_impl(const table &);
    void set_coeffs_impl(const table &);
    void set_bias_impl(double);
    void set_support_vector_count_impl(std::int64_t);
    void set_first_class_label_impl(std::int64_t);
    void set_second_class_label_impl(std::int64_t);

    dal::detail::pimpl<detail::model_impl<task_t>> impl_;
};

} // namespace oneapi::dal::svm
