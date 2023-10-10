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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/backend/primitives/objective_function.hpp"
//#include "oneapi/dal/backend/primitives/optimizers.hpp"

namespace oneapi::dal::logistic_regression::backend {

namespace be = dal::backend;
namespace pr = be::primitives;

template <typename Float>
class optimizer_iface {
public:
    virtual ~optimizer_iface() {}
    virtual sycl::event minimize(sycl::queue& q,
                                 pr::BaseFunction<Float>& f,
                                 pr::ndview<Float, 1>& x,
                                 const be::event_vector& deps = {}) = 0;
};

template <typename Float>
class newton_cg_optimizer : public optimizer_iface<Float> {
public:
    newton_cg_optimizer(double tol, std::int32_t maxiter);
    sycl::event minimize(sycl::queue& q,
                         pr::BaseFunction<Float>& f,
                         pr::ndview<Float, 1>& x,
                         const be::event_vector& deps = {});

private:
    double tol_;
    std::int32_t maxiter_;
};

template <typename Float, typename Task>
std::shared_ptr<optimizer_iface<Float>> get_optimizer(const detail::descriptor_base<Task>& desc) {
    if (desc.get_optimizer() == optimizer_enum::newton_cg) {
        return std::make_shared<newton_cg_optimizer<Float>>(desc.get_tol(), desc.get_max_iter());
    }
    return nullptr;
}

} // namespace oneapi::dal::logistic_regression::backend
