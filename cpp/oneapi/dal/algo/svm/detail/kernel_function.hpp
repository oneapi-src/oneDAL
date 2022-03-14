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
#include "oneapi/dal/algo/polynomial_kernel.hpp"
#include "oneapi/dal/algo/rbf_kernel.hpp"
#include "oneapi/dal/algo/sigmoid_kernel.hpp"

namespace oneapi::dal::svm::detail {
namespace v1 {

class kernel_function_impl;

class kernel_function_iface {
public:
    virtual ~kernel_function_iface() {}
    virtual kernel_function_impl* get_impl() const = 0;
#ifdef ONEDAL_DATA_PARALLEL
    virtual void compute_kernel_function(const dal::detail::data_parallel_policy& policy,
                                         const table& x,
                                         const table& y,
                                         homogen_table& res) = 0;
#endif
};

using kernel_function_ptr = std::shared_ptr<kernel_function_iface>;

template <typename Kernel>
class kernel_function : public base, public kernel_function_iface {
public:
    explicit kernel_function(const Kernel& kernel) : kernel_(kernel) {}

    kernel_function_impl* get_impl() const override {
        return nullptr;
    }

    const Kernel& get_kernel() const {
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
    explicit kernel_function(const kernel_t& kernel);
    kernel_function_impl* get_impl() const override;
#ifdef ONEDAL_DATA_PARALLEL
    void compute_kernel_function(const dal::detail::data_parallel_policy& policy,
                                 const table& x,
                                 const table& y,
                                 homogen_table& res) override;
#endif

private:
    kernel_t kernel_;
    dal::detail::pimpl<kernel_function_impl> impl_;
};

template <typename Float, typename Method>
class kernel_function<polynomial_kernel::descriptor<Float, Method>> : public base,
                                                                      public kernel_function_iface {
public:
    using kernel_t = polynomial_kernel::descriptor<Float, Method>;
    explicit kernel_function(const kernel_t& kernel);
    kernel_function_impl* get_impl() const override;
#ifdef ONEDAL_DATA_PARALLEL
    void compute_kernel_function(const dal::detail::data_parallel_policy& policy,
                                 const table& x,
                                 const table& y,
                                 homogen_table& res) override;
#endif

private:
    kernel_t kernel_;
    dal::detail::pimpl<kernel_function_impl> impl_;
};

template <typename Float, typename Method>
class kernel_function<rbf_kernel::descriptor<Float, Method>> : public base,
                                                               public kernel_function_iface {
public:
    using kernel_t = rbf_kernel::descriptor<Float, Method>;
    explicit kernel_function(const kernel_t& kernel);
    kernel_function_impl* get_impl() const override;
#ifdef ONEDAL_DATA_PARALLEL
    void compute_kernel_function(const dal::detail::data_parallel_policy& policy,
                                 const table& x,
                                 const table& y,
                                 homogen_table& res) override;
#endif

private:
    kernel_t kernel_;
    dal::detail::pimpl<kernel_function_impl> impl_;
};

template <typename Float, typename Method>
class kernel_function<sigmoid_kernel::descriptor<Float, Method>> : public base,
                                                                   public kernel_function_iface {
public:
    using kernel_t = sigmoid_kernel::descriptor<Float, Method>;
    explicit kernel_function(const kernel_t& kernel);
    kernel_function_impl* get_impl() const override;
#ifdef ONEDAL_DATA_PARALLEL
    void compute_kernel_function(const dal::detail::data_parallel_policy& policy,
                                 const table& x,
                                 const table& y,
                                 homogen_table& res) override;
#endif

private:
    kernel_t kernel_;
    dal::detail::pimpl<kernel_function_impl> impl_;
};

struct kernel_function_accessor {
    template <typename Descriptor>
    const kernel_function_ptr& get_kernel_impl(Descriptor&& desc) const {
        return desc.get_kernel_impl();
    }
};

template <typename Descriptor>
kernel_function_impl* get_kernel_function_impl(Descriptor&& desc) {
    const auto& kernel = kernel_function_accessor{}.get_kernel_impl(std::forward<Descriptor>(desc));
    return kernel ? kernel->get_impl() : nullptr;
}

template <typename Descriptor>
const kernel_function_ptr& get_kernel_ptr(Descriptor&& desc) {
    return kernel_function_accessor{}.get_kernel_impl(std::forward<Descriptor>(desc));
}

} // namespace v1

using v1::kernel_function_impl;
using v1::kernel_function_iface;
using v1::kernel_function_ptr;
using v1::kernel_function;
using v1::kernel_function_accessor;
using v1::get_kernel_function_impl;
using v1::get_kernel_ptr;

} // namespace oneapi::dal::svm::detail
