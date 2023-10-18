/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/logistic_regression/detail/optimizer.hpp"
#include "oneapi/dal/algo/logistic_regression/backend/optimizer_impl.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include "oneapi/dal/backend/primitives/optimizers.hpp"
#endif

namespace oneapi::dal::logistic_regression::detail {
namespace v1 {

template <typename F, typename M>
using newton_cg_optimizer_t = newton_cg::descriptor<F, M>;

namespace be = dal::backend;
namespace pr = be::primitives;

class newton_cg_optimizer_impl : public optimizer_impl {
public:
    newton_cg_optimizer_impl(std::int64_t max_iter, double tol) : max_iter_(max_iter), tol_(tol) {}

    optimizer_type get_optimizer_type() override {
        return optimizer_type::newton_cg;
    }

    double get_tol() override {
        return tol_;
    }

    std::int64_t get_max_iter() override {
        return max_iter_;
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Float>
    sycl::event minimize_impl(sycl::queue& q,
                              pr::base_function<Float>& f,
                              pr::ndview<Float, 1>& x,
                              const be::event_vector& deps = {}) {
        return pr::newton_cg(q, f, x, Float(tol_), max_iter_, deps);
    }

    sycl::event minimize(sycl::queue& q,
                         pr::base_function<float>& f,
                         pr::ndview<float, 1>& x,
                         const be::event_vector& deps = {}) final {
        return minimize_impl(q, f, x, deps);
    }

    sycl::event minimize(sycl::queue& q,
                         pr::base_function<double>& f,
                         pr::ndview<double, 1>& x,
                         const be::event_vector& deps = {}) final {
        return minimize_impl(q, f, x, deps);
    }
#endif

private:
    std::int64_t max_iter_ = 100;
    double tol_ = 1e-4;
};

template <typename F, typename M>
optimizer<newton_cg_optimizer_t<F, M>>::optimizer(const newton_cg_optimizer_t<F, M>& opt)
        : optimizer_(opt),
          impl_(new newton_cg_optimizer_impl{ opt.get_max_iteration(), opt.get_tolerance() }) {}

template <typename F, typename M>
optimizer_impl* optimizer<newton_cg_optimizer_t<F, M>>::get_impl() const {
    return impl_.get();
}

#define INSTANTIATE_NEWTON_CG(F, M) \
    template class ONEDAL_EXPORT optimizer<newton_cg_optimizer_t<F, M>>;

INSTANTIATE_NEWTON_CG(float, newton_cg::method::dense)
INSTANTIATE_NEWTON_CG(double, newton_cg::method::dense)

} // namespace v1
} // namespace oneapi::dal::logistic_regression::detail
