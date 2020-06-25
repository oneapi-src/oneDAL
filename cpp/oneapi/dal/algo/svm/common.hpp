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

namespace oneapi::dal::svm {

namespace detail {
struct tag {};
class descriptor_impl;
class model_impl;

enum class kernel_function_kind {
    linear,
    rbf,
    unknown
}

class kernel_function_iface {
public:
    virtual ~kernel_function_iface()                   = default;
    virtual operator()(const table &x, const table &y) = 0;
    virtual kernel_function_kind get_kind() const      = 0;
};

using kf_iface_ptr = std::shared_ptr<kernel_function_iface>;

template <typename Kernel>
class kernel_function : public base, public kernel_function_iface {
public:
    explicit kernel_function(const Kernel &kernel) : kernel_(kernel) {}

    table operator()(const table &x,
                     const table &y) override { // return dal::compute(kernel, x, y).ger
        return kernel_(x, y);
    }

    kernel_function_kind get_kind() const override {
        using kernel_tag_t = typename Kernel::tag_t;
        if constexpr (std::is_same_v<kernel_tag_t, linear_kernel::detail::tag>) {
            return kernel_function_kind::linear;
        }

        if constexpr (std::is_same_v<kernel_tag_t, rbf_kernel::detail::tag>) {
            return kernel_function_kind::linear;
        }

        return kernel_function_kind::unknown;
    }

private:
    Kernel kernel_;
};

template <typename Float, typename Method>
class kernel_function<linear_kernel::descriptor<Float, Method>> : public base,
                                                                  public kernel_function_iface {
public:
    explicit kernel_function(const linear_kernel<Float, Method> &kernel) : kernel_(kernel) {}

private:
    linear_kernel::descriptor<Float, Method>> desc_;
};

} // namespace detail

namespace method {
struct thunder {};
struct smo {};
using by_default = thunder;
} // namespace method

namespace task {
struct classification {};
struct regression {};
using by_default = classification;
} // namespace task

class descriptor_base : public base {
public:
    using tag_t    = detail::tag;
    using float_t  = float;
    using task_t   = task::by_default;
    using method_t = method::by_default;
    using kernel_t = method::by_default;

    // descriptor_base();
    // explicit descriptor_base(detail::object_wrapper_iface &kernel);
    explicit descriptor_base(detail::kf_iface_ptr &kernel);

    double get_c() const;
    double get_accuracy_threshold() const;
    std::int64_t get_max_iteration_count() const;
    double get_cache_size() const;
    double get_tau() const;
    bool get_shrinking() const;

protected:
    void set_c_impl(const double value);
    void set_accuracy_threshold_impl(const double value);
    void set_max_iteration_count_impl(const std::int64_t value);
    void set_cache_size_impl(const double value);
    void set_tau_impl(const double value);
    void set_shrinking_impl(const bool value);

    void set_kernel_impl(detail::kf_iface_ptr &kernel);

    const detail::kf_iface_ptr &get_kernel_impl() const;

    dal::detail::pimpl<detail::descriptor_impl> impl_;
};

template <typename Float  = descriptor_base::float_t,
          typename Task   = descriptor_base::task_t,
          typename Method = descriptor_base::method_t>
typename Kernel = descriptor_base::method_t > class descriptor : public descriptor_base {
public:
    using float_t  = Float;
    using task_t   = Task;
    using method_t = Method;
    using kernel_t = Method;

    const Kernel &get_kernel() const {
        const auto kf =
            std::static_pointer_cast<detail::kernel_function<Kernel>>(get_kernel_impl());
        // static_cast<detail::kernel_function<Kernel>*>(get_kernel_impl());
        return get_kernel_impl().get_ref<Kernel>();
    }

    descriptor(const Kernel &kernel = kernel_t{})
            : descriptor_base(std::make_shared<detail::kernel_function<Kernel>>{ kernel })

    {}
    // {
    // set_kernel_impl(detail::object_wrapper<Kernel>(value));
    // set_kernel_impl(kernel);
    // }

    // descriptor(const Kernel &kernel = kernel_t{}) {
    //     // set_kernel_impl(detail::object_wrapper<Kernel>(value));
    //     set_kernel_impl(kernel);
    // }

    auto &set_c(const double value) {
        set_c_impl(value);
        return *this;
    }

    auto &set_accuracy_threshold(const double value) {
        set_accuracy_threshold_impl(value);
        return *this;
    }

    auto &set_max_iteration_count(const std::int64_t value) {
        set_max_iteration_count_impl(value);
        return *this;
    }

    auto &set_cache_size(const double value) {
        set_cache_size_impl(value);
        return *this;
    }

    auto &set_tau(const double value) {
        set_tau_impl(value);
        return *this;
    }

    auto &set_shrinking(const bool value) {
        set_shrinking_impl(value);
        return *this;
    }

    auto &set_kernel(const Kernel &value) {
        set_kernel_impl(detail::object_wrapper<Kernel>(value));
        return *this;
    }
};

class model : public base {
    friend dal::detail::pimpl_accessor;

public:
    model();

    table get_support_vectors() const;

    auto &set_support_vectors(const table &value) {
        set_support_vectors_impl(value);
        return *this;
    }

    table get_coefficients() const;

    auto &set_coefficients(const table &value) {
        set_coefficients_impl(value);
        return *this;
    }

    double get_bias() const;

    auto &set_bias(const double value) {
        set_bias_impl(value);
        return *this;
    }

    std::int64_t get_support_vectors_count() const;

    auto &set_support_vectors_count(const std::int64_t value) {
        set_support_vectors_count_impl(value);
        return *this;
    }

private:
    void set_support_vectors_impl(const table &);
    void set_coefficients_impl(const table &);
    void set_bias_impl(const double);
    void set_support_vectors_count_impl(const std::int64_t);

    dal::detail::pimpl<detail::model_impl> impl_;
};

} // namespace oneapi::dal::svm
