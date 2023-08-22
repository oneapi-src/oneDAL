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

#include "oneapi/dal/algo/svm/detail/kernel_function.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"
#include "daal/src/algorithms/kernel_function/kernel_function_dense_base.h"

namespace oneapi::dal::svm::detail {

using daal_kf_t = daal::algorithms::kernel_function::KernelIfacePtr;
namespace daal_linear_kernel = daal::algorithms::kernel_function::linear;
namespace daal_rbf_kernel = daal::algorithms::kernel_function::rbf;
namespace daal_polynomial_kernel = daal::algorithms::kernel_function::polynomial::internal;

template <typename F, typename M>
using linear_kernel_t = linear_kernel::descriptor<F, M>;

template <typename F, typename M>
using polynomial_kernel_t = polynomial_kernel::descriptor<F, M>;

template <typename F, typename M>
using rbf_kernel_t = rbf_kernel::descriptor<F, M>;

template <typename F, typename M>
using sigmoid_kernel_t = sigmoid_kernel::descriptor<F, M>;

template <typename Float, typename Method>
class daal_interop_linear_kernel_impl : public kernel_function_impl {
public:
    daal_interop_linear_kernel_impl(double scale, double shift) : scale_(scale), shift_(shift) {}

    daal_kf_t get_daal_kernel_function(bool is_dense) override {
        if (is_dense) {
            constexpr daal_linear_kernel::Method daal_method = get_daal_dense_method();
            auto alg = new daal_linear_kernel::Batch<Float, daal_method>;
            alg->parameter.k = scale_;
            alg->parameter.b = shift_;
            return daal_kf_t(alg);
        }
        else {
            constexpr daal_linear_kernel::Method daal_method = get_daal_csr_method();
            auto alg = new daal_linear_kernel::Batch<Float, daal_method>;
            alg->parameter.k = scale_;
            alg->parameter.b = shift_;
            return daal_kf_t(alg);
        }
    }

private:
    static constexpr daal_linear_kernel::Method get_daal_dense_method() {
        return daal_linear_kernel::Method::defaultDense;
    }

    static constexpr daal_linear_kernel::Method get_daal_csr_method() {
        return daal_linear_kernel::Method::fastCSR;
    }

    double scale_;
    double shift_;
};

template <typename Float, typename Method>
class daal_interop_polynomial_kernel_impl : public kernel_function_impl {
public:
    daal_interop_polynomial_kernel_impl(double scale, double shift, std::int64_t degree)
            : scale_(scale),
              shift_(shift),
              degree_(degree) {}

    daal_kf_t get_daal_kernel_function(bool is_dense) override {
        if (is_dense) {
            constexpr daal_polynomial_kernel::Method daal_method = get_daal_dense_method();
            auto alg = new daal_polynomial_kernel::Batch<Float, daal_method>;
            alg->parameter.scale = scale_;
            alg->parameter.shift = shift_;
            alg->parameter.degree = degree_;
            return daal_kf_t(alg);
        }
        else {
            constexpr daal_polynomial_kernel::Method daal_method = get_daal_csr_method();
            auto alg = new daal_polynomial_kernel::Batch<Float, daal_method>;
            alg->parameter.scale = scale_;
            alg->parameter.shift = shift_;
            alg->parameter.degree = degree_;
            return daal_kf_t(alg);
        }
    }

private:
    static constexpr daal_polynomial_kernel::Method get_daal_dense_method() {
        return daal_polynomial_kernel::Method::defaultDense;
    }

    static constexpr daal_polynomial_kernel::Method get_daal_csr_method() {
        return daal_polynomial_kernel::Method::fastCSR;
    }

    double scale_;
    double shift_;
    std::int64_t degree_;
};

template <typename Float, typename Method>
class daal_interop_rbf_kernel_impl : public kernel_function_impl {
public:
    daal_interop_rbf_kernel_impl(double sigma) : sigma_(sigma) {}

    daal_kf_t get_daal_kernel_function(bool is_dense) override {
        if (is_dense) {
            constexpr daal_rbf_kernel::Method daal_method = get_daal_dense_method();
            auto alg = new daal_rbf_kernel::Batch<Float, daal_method>;
            alg->parameter.sigma = sigma_;
            return daal_kf_t(alg);
        }
        else {
            constexpr daal_rbf_kernel::Method daal_method = get_daal_csr_method();
            auto alg = new daal_rbf_kernel::Batch<Float, daal_method>;
            alg->parameter.sigma = sigma_;
            return daal_kf_t(alg);
        }
    }

private:
    static constexpr daal_rbf_kernel::Method get_daal_dense_method() {
        return daal_rbf_kernel::Method::defaultDense;
    }
    static constexpr daal_rbf_kernel::Method get_daal_csr_method() {
        return daal_rbf_kernel::Method::fastCSR;
    }

    double sigma_;
};

template <typename Float, typename Method>
class daal_interop_sigmoid_kernel_impl : public kernel_function_impl {
public:
    daal_interop_sigmoid_kernel_impl(double scale, double shift) : scale_(scale), shift_(shift) {}

    daal_kf_t get_daal_kernel_function(bool is_dense) override {
        if (is_dense) {
            constexpr daal_polynomial_kernel::Method daal_method = get_daal_dense_method();
            auto alg = new daal_polynomial_kernel::Batch<Float, daal_method>;
            alg->parameter.scale = scale_;
            alg->parameter.shift = shift_;
            alg->parameter.kernelType =
                daal::algorithms::kernel_function::internal::KernelType::sigmoid;
            return daal_kf_t(alg);
        }
        else {
            constexpr daal_polynomial_kernel::Method daal_method = get_daal_csr_method();
            auto alg = new daal_polynomial_kernel::Batch<Float, daal_method>;
            alg->parameter.scale = scale_;
            alg->parameter.shift = shift_;
            alg->parameter.kernelType =
                daal::algorithms::kernel_function::internal::KernelType::sigmoid;
            return daal_kf_t(alg);
        }
    }

private:
    static constexpr daal_polynomial_kernel::Method get_daal_dense_method() {
        return daal_polynomial_kernel::Method::defaultDense;
    }

    static constexpr daal_polynomial_kernel::Method get_daal_csr_method() {
        return daal_polynomial_kernel::Method::fastCSR;
    }

    double scale_;
    double shift_;
};

template <typename F, typename M>
kernel_function<linear_kernel_t<F, M>>::kernel_function(const linear_kernel_t<F, M>& kernel)
        : kernel_(kernel),
          impl_(new daal_interop_linear_kernel_impl<F, M>{ kernel.get_scale(),
                                                           kernel.get_shift() }) {}

template <typename F, typename M>
kernel_function_impl* kernel_function<linear_kernel_t<F, M>>::get_impl() const {
    return impl_.get();
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename F, typename M>
void kernel_function<linear_kernel_t<F, M>>::compute_kernel_function(
    const dal::detail::data_parallel_policy& policy,
    const table& x,
    const table& y,
    homogen_table& res) {
    dal::linear_kernel::detail::compute_ops<linear_kernel::descriptor<F, M>> kernel_compute_ops;
    kernel_compute_ops(policy, kernel_, x, y, res);
}
#endif

template <typename F, typename M>
kernel_function<polynomial_kernel_t<F, M>>::kernel_function(const polynomial_kernel_t<F, M>& kernel)
        : kernel_(kernel),
          impl_(new daal_interop_polynomial_kernel_impl<F, M>{ kernel.get_scale(),
                                                               kernel.get_shift(),
                                                               kernel.get_degree() }) {}

template <typename F, typename M>
kernel_function_impl* kernel_function<polynomial_kernel_t<F, M>>::get_impl() const {
    return impl_.get();
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename F, typename M>
void kernel_function<polynomial_kernel_t<F, M>>::compute_kernel_function(
    const dal::detail::data_parallel_policy& policy,
    const table& x,
    const table& y,
    homogen_table& res) {
    dal::polynomial_kernel::detail::compute_ops<polynomial_kernel::descriptor<F, M>>
        kernel_compute_ops;
    kernel_compute_ops(policy, kernel_, x, y, res);
}
#endif

template <typename F, typename M>
kernel_function<rbf_kernel_t<F, M>>::kernel_function(const rbf_kernel_t<F, M>& kernel)
        : kernel_(kernel),
          impl_(new daal_interop_rbf_kernel_impl<F, M>{ kernel.get_sigma() }) {}

template <typename F, typename M>
kernel_function_impl* kernel_function<rbf_kernel_t<F, M>>::get_impl() const {
    return impl_.get();
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename F, typename M>
void kernel_function<rbf_kernel_t<F, M>>::compute_kernel_function(
    const dal::detail::data_parallel_policy& policy,
    const table& x,
    const table& y,
    homogen_table& res) {
    dal::rbf_kernel::detail::compute_ops<rbf_kernel::descriptor<F, M>> kernel_compute_ops;
    kernel_compute_ops(policy, kernel_, x, y, res);
}
#endif

template <typename F, typename M>
kernel_function<sigmoid_kernel_t<F, M>>::kernel_function(const sigmoid_kernel_t<F, M>& kernel)
        : kernel_(kernel),
          impl_(new daal_interop_sigmoid_kernel_impl<F, M>{ kernel.get_scale(),
                                                            kernel.get_shift() }) {}

template <typename F, typename M>
kernel_function_impl* kernel_function<sigmoid_kernel_t<F, M>>::get_impl() const {
    return impl_.get();
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename F, typename M>
void kernel_function<sigmoid_kernel_t<F, M>>::compute_kernel_function(
    const dal::detail::data_parallel_policy& policy,
    const table& x,
    const table& y,
    homogen_table& res) {
    dal::sigmoid_kernel::detail::compute_ops<sigmoid_kernel::descriptor<F, M>> kernel_compute_ops;
    kernel_compute_ops(policy, kernel_, x, y, res);
}
#endif

#define INSTANTIATE_LINEAR(F, M) \
    template class ONEDAL_EXPORT kernel_function<linear_kernel_t<F, M>>;

#define INSTANTIATE_POLYNOMIAL(F, M) \
    template class ONEDAL_EXPORT kernel_function<polynomial_kernel_t<F, M>>;

#define INSTANTIATE_RBF(F, M) template class ONEDAL_EXPORT kernel_function<rbf_kernel_t<F, M>>;

#define INSTANTIATE_SIGMOID(F, M) \
    template class ONEDAL_EXPORT kernel_function<sigmoid_kernel_t<F, M>>;

INSTANTIATE_LINEAR(float, linear_kernel::method::dense)
INSTANTIATE_LINEAR(double, linear_kernel::method::dense)

INSTANTIATE_POLYNOMIAL(float, polynomial_kernel::method::dense)
INSTANTIATE_POLYNOMIAL(double, polynomial_kernel::method::dense)

INSTANTIATE_RBF(float, rbf_kernel::method::dense)
INSTANTIATE_RBF(double, rbf_kernel::method::dense)

INSTANTIATE_SIGMOID(float, sigmoid_kernel::method::dense)
INSTANTIATE_SIGMOID(double, sigmoid_kernel::method::dense)

} // namespace oneapi::dal::svm::detail
