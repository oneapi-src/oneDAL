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

#include "oneapi/dal/algo/svm/common.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::svm {

namespace detail {

using daal_kf = daal::algorithms::kernel_function::KernelIfacePtr;
namespace daal_linear_kernel = daal::algorithms::kernel_function::linear;
namespace daal_rbf_kernel = daal::algorithms::kernel_function::rbf;

template <typename Float, typename Method>
class daal_interop_linear_kernel_impl : public kernel_function_impl {
public:
    daal_interop_linear_kernel_impl(double scale, double shift) : scale_(scale), shift_(shift) {}

    daal_kf get_daal_kernel_function() override {
        constexpr daal_linear_kernel::Method daal_method = get_daal_method();
        auto alg = new daal_linear_kernel::Batch<Float, daal_method>;
        alg->parameter.k = scale_;
        alg->parameter.b = shift_;
        return daal_kf(alg);
    }

private:
    static constexpr daal_linear_kernel::Method get_daal_method() {
        if constexpr (std::is_same_v<Method, linear_kernel::method::dense>)
            return daal_linear_kernel::Method::defaultDense;
        else if constexpr (std::is_same_v<Method, linear_kernel::method::csr>)
            return daal_linear_kernel::Method::fastCSR;
        return daal_linear_kernel::Method::defaultDense;
    }

    double scale_;
    double shift_;
};

template <typename F, typename M>
using linear_kernel_t = linear_kernel::descriptor<F, M>;

template <typename F, typename M>
kernel_function<linear_kernel_t<F, M>>::kernel_function(const linear_kernel_t<F, M> &kernel)
        : kernel_(kernel),
          impl_(new daal_interop_linear_kernel_impl<F, M>{ kernel.get_scale(),
                                                           kernel.get_shift() }) {}

template <typename F, typename M>
kernel_function_impl *kernel_function<linear_kernel_t<F, M>>::get_impl() const {
    return impl_.get();
}

#define INSTANTIATE_LINEAR(F, M) template class kernel_function<linear_kernel_t<F, M>>;

INSTANTIATE_LINEAR(float, linear_kernel::method::dense)
INSTANTIATE_LINEAR(float, linear_kernel::method::csr)
INSTANTIATE_LINEAR(double, linear_kernel::method::dense)
INSTANTIATE_LINEAR(double, linear_kernel::method::csr)

#undef INSTANTIATE_LINEAR

template <typename F, typename M>
using rbf_kernel_t = rbf_kernel::descriptor<F, M>;

template <typename Float, typename Method>
class daal_interop_rbf_kernel_impl : public kernel_function_impl {
public:
    daal_interop_rbf_kernel_impl(double sigma) : sigma_(sigma) {}

    daal_kf get_daal_kernel_function() override {
        constexpr daal_rbf_kernel::Method daal_method = get_daal_method();
        auto alg = new daal_rbf_kernel::Batch<Float, daal_method>;
        alg->parameter.sigma = sigma_;
        return daal_kf(alg);
    }

private:
    static constexpr daal_rbf_kernel::Method get_daal_method() {
        if constexpr (std::is_same_v<Method, linear_kernel::method::dense>)
            return daal_rbf_kernel::Method::defaultDense;
        else if constexpr (std::is_same_v<Method, linear_kernel::method::csr>)
            return daal_rbf_kernel::Method::fastCSR;
        return daal_rbf_kernel::Method::defaultDense;
    }

    double sigma_;
};

template <typename F, typename M>
kernel_function<rbf_kernel_t<F, M>>::kernel_function(const rbf_kernel_t<F, M> &kernel)
        : kernel_(kernel),
          impl_(new daal_interop_rbf_kernel_impl<F, M>{ kernel.get_sigma() }) {}

template <typename F, typename M>
kernel_function_impl *kernel_function<rbf_kernel_t<F, M>>::get_impl() const {
    return impl_.get();
}

#define INSTANTIATE_RBF(F, M) template class kernel_function<rbf_kernel_t<F, M>>;

INSTANTIATE_RBF(float, rbf_kernel::method::dense)
INSTANTIATE_RBF(float, rbf_kernel::method::csr)
INSTANTIATE_RBF(double, rbf_kernel::method::dense)
INSTANTIATE_RBF(double, rbf_kernel::method::csr)

#undef INSTANTIATE_RBF

} // namespace detail

class detail::descriptor_impl : public base {
public:
    explicit descriptor_impl(const detail::kf_iface_ptr &kernel) : kernel(kernel) {}

    detail::kf_iface_ptr kernel;
    double c = 1.0;
    double accuracy_threshold = 0.001;
    std::int64_t max_iteration_count = 100000;
    double cache_size = 200.0;
    double tau = 1e-6;
    bool shrinking = true;
};

class detail::model_impl : public base {
public:
    table support_vectors;
    table coeffs;
    double bias;
    std::int64_t support_vector_count;
    double first_class_label;
    double second_class_label;
};

using detail::descriptor_impl;
using detail::model_impl;

descriptor_base::descriptor_base(const detail::kf_iface_ptr &kernel)
        : impl_(new descriptor_impl{ kernel }) {}

double descriptor_base::get_c() const {
    return impl_->c;
}

double descriptor_base::get_accuracy_threshold() const {
    return impl_->accuracy_threshold;
}

std::int64_t descriptor_base::get_max_iteration_count() const {
    return impl_->max_iteration_count;
}

double descriptor_base::get_cache_size() const {
    return impl_->cache_size;
}

double descriptor_base::get_tau() const {
    return impl_->tau;
}

bool descriptor_base::get_shrinking() const {
    return impl_->shrinking;
}

void descriptor_base::set_c_impl(double value) {
    if (value <= 0.0) {
        throw domain_error("c should be > 0");
    }
    impl_->c = value;
}

void descriptor_base::set_accuracy_threshold_impl(double value) {
    if (value < 0.0) {
        throw domain_error("accuracy_threshold should be >= 0.0");
    }
    impl_->accuracy_threshold = value;
}

void descriptor_base::set_max_iteration_count_impl(std::int64_t value) {
    if (value <= 0) {
        throw domain_error("max_iteration_count should be > 0");
    }
    impl_->max_iteration_count = value;
}

void descriptor_base::set_cache_size_impl(double value) {
    if (value <= 0.0) {
        throw domain_error("cache_size should be > 0");
    }
    impl_->cache_size = value;
}

void descriptor_base::set_tau_impl(double value) {
    if (value <= 0.0) {
        throw domain_error("tau should be > 0");
    }
    impl_->tau = value;
}

void descriptor_base::set_shrinking_impl(bool value) {
    impl_->shrinking = value;
}

void descriptor_base::set_kernel_impl(const detail::kf_iface_ptr &kernel) {
    impl_->kernel = kernel;
}

const detail::kf_iface_ptr &descriptor_base::get_kernel_impl() const {
    return impl_->kernel;
}

model::model() : impl_(new model_impl{}) {}

table model::get_support_vectors() const {
    return impl_->support_vectors;
}

table model::get_coeffs() const {
    return impl_->coeffs;
}

double model::get_bias() const {
    return impl_->bias;
}

std::int64_t model::get_support_vector_count() const {
    return impl_->support_vector_count;
}

std::int64_t model::get_first_class_label() const {
    return impl_->first_class_label;
}

std::int64_t model::get_second_class_label() const {
    return impl_->second_class_label;
}

void model::set_support_vectors_impl(const table &value) {
    impl_->support_vectors = value;
}

void model::set_coeffs_impl(const table &value) {
    impl_->coeffs = value;
}

void model::set_bias_impl(double value) {
    impl_->bias = value;
}

void model::set_support_vector_count_impl(std::int64_t value) {
    impl_->support_vector_count = value;
}

void model::set_first_class_label_impl(std::int64_t value) {
    impl_->first_class_label = value;
}

void model::set_second_class_label_impl(std::int64_t value) {
    impl_->second_class_label = value;
}

} // namespace oneapi::dal::svm
