/*******************************************************************************
* Copyright contributors to the oneDAL project
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
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include "oneapi/dal/backend/primitives/optimizers.hpp"
#endif

namespace oneapi::dal::logistic_regression::detail {
namespace v1 {

namespace be = dal::backend;
namespace pr = be::primitives;

enum optimizer_type { newton_cg };

class optimizer_impl : public base {
public:
    virtual ~optimizer_impl() = default;
    virtual optimizer_type get_optimizer_type() = 0;
    virtual double get_tol() = 0;
    virtual std::int64_t get_max_iter() = 0;

    // this function returns meaningful value only for newton_cg optimizer
    // inner iterations value can be accessed after minimize method was called
    virtual std::int64_t get_inner_iter() {
        return -1;
    }

#ifdef ONEDAL_DATA_PARALLEL

    virtual std::pair<sycl::event, std::int64_t> minimize(sycl::queue& q,
                                                          pr::base_function<float>& f,
                                                          pr::ndview<float, 1>& x,
                                                          const be::event_vector& deps = {}) = 0;
    virtual std::pair<sycl::event, std::int64_t> minimize(sycl::queue& q,
                                                          pr::base_function<double>& f,
                                                          pr::ndview<double, 1>& x,
                                                          const be::event_vector& deps = {}) = 0;
#endif
};

} // namespace v1

using v1::optimizer_impl;
using v1::optimizer_type;

} // namespace oneapi::dal::logistic_regression::detail
