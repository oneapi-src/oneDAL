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

#include "oneapi/dal/algo/svm/detail/kernel_function.hpp"
#include "oneapi/dal/algo/svm/backend/kernel_function_impl.hpp"

namespace oneapi::dal::svm::detail {
namespace v1 {

using daal_kf_t = daal::algorithms::kernel_function::KernelIfacePtr;
namespace daal_linear_kernel = daal::algorithms::kernel_function::linear;
namespace daal_polynomial_kernel = daal::algorithms::kernel_function::polynomial::internal;
namespace daal_rbf_kernel = daal::algorithms::kernel_function::rbf;

template <typename F, typename M>
using linear_kernel_t = linear_kernel::descriptor<F, M>;

template <typename F, typename M>
using polynomial_kernel_t = polynomial_kernel::descriptor<F, M>;

template <typename F, typename M>
using rbf_kernel_t = rbf_kernel::descriptor<F, M>;

template <typename Float, typename Method>
class daal_interop_linear_kernel_impl : public kernel_function_impl {
public:
    daal_interop_linear_kernel_impl(double scale, double shift) : scale_(scale), shift_(shift) {}

    daal_kf_t get_daal_kernel_function() override {
        constexpr daal_linear_kernel::Method daal_method = get_daal_method();
        auto alg = new daal_linear_kernel::Batch<Float, daal_method>;
        alg->parameter.k = scale_;
        alg->parameter.b = shift_;
        return daal_kf_t(alg);
    }

private:
    static constexpr daal_linear_kernel::Method get_daal_method() {
        static_assert(dal::detail::is_one_of_v<Method, linear_kernel::method::dense>);

        if constexpr (std::is_same_v<Method, linear_kernel::method::dense>) {
            return daal_linear_kernel::Method::defaultDense;
        }
        // TODO: Comment out once CSR method is supported
        // else if constexpr (std::is_same_v<Method, linear_kernel::method::csr>) {
        //     return daal_linear_kernel::Method::fastCSR;
        // }
        return daal_linear_kernel::Method::defaultDense;
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

    daal_kf_t get_daal_kernel_function() override {
        constexpr daal_polynomial_kernel::Method daal_method = get_daal_method();
        auto alg = new daal_polynomial_kernel::Batch<Float, daal_method>;
        alg->parameter.scale = scale_;
        alg->parameter.shift = shift_;
        alg->parameter.degree = degree_;
        return daal_kf_t(alg);
    }

private:
    static constexpr daal_polynomial_kernel::Method get_daal_method() {
        static_assert(dal::detail::is_one_of_v<Method, polynomial_kernel::method::dense>);

        if constexpr (std::is_same_v<Method, polynomial_kernel::method::dense>) {
            return daal_polynomial_kernel::Method::defaultDense;
        }
        // TODO: Comment out once CSR method is supported
        // else if constexpr (std::is_same_v<Method, polynomial_kernel::method::csr>) {
        //     return daal_polynomial_kernel::Method::fastCSR;
        // }
        return daal_polynomial_kernel::Method::defaultDense;
    }

    double scale_;
    double shift_;
    std::int64_t degree_;
};

template <typename Float, typename Method>
class daal_interop_rbf_kernel_impl : public kernel_function_impl {
public:
    daal_interop_rbf_kernel_impl(double sigma) : sigma_(sigma) {}

    daal_kf_t get_daal_kernel_function() override {
        constexpr daal_rbf_kernel::Method daal_method = get_daal_method();
        auto alg = new daal_rbf_kernel::Batch<Float, daal_method>;
        alg->parameter.sigma = sigma_;
        return daal_kf_t(alg);
    }

private:
    static constexpr daal_rbf_kernel::Method get_daal_method() {
        static_assert(dal::detail::is_one_of_v<Method, rbf_kernel::method::dense>);

        if constexpr (std::is_same_v<Method, rbf_kernel::method::dense>) {
            return daal_rbf_kernel::Method::defaultDense;
        }
        // TODO: Comment out once CSR method is supported
        // else if constexpr (std::is_same_v<Method, rbf_kernel::method::csr>) {
        //     return daal_rbf_kernel::Method::fastCSR;
        // }
        return daal_rbf_kernel::Method::defaultDense;
    }

    double sigma_;
};

template <typename F, typename M>
kernel_function<linear_kernel_t<F, M>>::kernel_function(const linear_kernel_t<F, M> &kernel)
        : kernel_(kernel),
          impl_(new daal_interop_linear_kernel_impl<F, M>{ kernel.get_scale(),
                                                           kernel.get_shift() }) {}

template <typename F, typename M>
kernel_function_impl *kernel_function<linear_kernel_t<F, M>>::get_impl() const {
    return impl_.get();
}

template <typename F, typename M>
kernel_function<polynomial_kernel_t<F, M>>::kernel_function(const polynomial_kernel_t<F, M> &kernel)
        : kernel_(kernel),
          impl_(new daal_interop_polynomial_kernel_impl<F, M>{ kernel.get_scale(),
                                                               kernel.get_shift(),
                                                               kernel.get_degree() }) {}

template <typename F, typename M>
kernel_function_impl *kernel_function<polynomial_kernel_t<F, M>>::get_impl() const {
    return impl_.get();
}

template <typename F, typename M>
kernel_function<rbf_kernel_t<F, M>>::kernel_function(const rbf_kernel_t<F, M> &kernel)
        : kernel_(kernel),
          impl_(new daal_interop_rbf_kernel_impl<F, M>{ kernel.get_sigma() }) {}

template <typename F, typename M>
kernel_function_impl *kernel_function<rbf_kernel_t<F, M>>::get_impl() const {
    return impl_.get();
}

#define INSTANTIATE_LINEAR(F, M) \
    template class ONEDAL_EXPORT kernel_function<linear_kernel_t<F, M>>;

#define INSTANTIATE_POLYNOMIAL(F, M) \
    template class ONEDAL_EXPORT kernel_function<polynomial_kernel_t<F, M>>;

#define INSTANTIATE_RBF(F, M) template class ONEDAL_EXPORT kernel_function<rbf_kernel_t<F, M>>;

INSTANTIATE_LINEAR(float, linear_kernel::method::dense)
INSTANTIATE_LINEAR(double, linear_kernel::method::dense)

INSTANTIATE_POLYNOMIAL(float, polynomial_kernel::method::dense)
INSTANTIATE_POLYNOMIAL(double, polynomial_kernel::method::dense)

INSTANTIATE_RBF(float, rbf_kernel::method::dense)
INSTANTIATE_RBF(double, rbf_kernel::method::dense)

} // namespace v1
} // namespace oneapi::dal::svm::detail
