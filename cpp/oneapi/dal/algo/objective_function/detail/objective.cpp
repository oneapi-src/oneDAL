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

#include "oneapi/dal/algo/objective_function/detail/objective.hpp"
#include "oneapi/dal/algo/objective_function/backend/objective_impl.hpp"

namespace oneapi::dal::objective_function::detail {

template <typename F, typename M>
using logloss_objective_t = logloss_objective::descriptor<F, M>;

class logloss_objective_impl : public objective_impl {
public:
    logloss_objective_impl(double l1_coef, double l2_coef, bool fit_intercept)
            : l1_coef(l1_coef),
              l2_coef(l2_coef),
              fit_intercept(fit_intercept) {}
    double get_l1_regularization_coefficient() {
        return l1_coef;
    }
    double get_l2_regularization_coefficient() {
        return l2_coef;
    }
    bool get_intercept_flag() {
        return fit_intercept;
    }

private:
    double l1_coef = 0.0;
    double l2_coef = 0.0;
    bool fit_intercept = true;
};

template <typename F, typename M>
objective<logloss_objective_t<F, M>>::objective(const logloss_objective_t<F, M>& obj)
        : objective_(obj),
          impl_(new logloss_objective_impl{ obj.get_l1_regularization_coefficient(),
                                            obj.get_l2_regularization_coefficient(),
                                            obj.get_intercept_flag() }) {}

template <typename F, typename M>
objective_impl* objective<logloss_objective_t<F, M>>::get_impl() const {
    return impl_.get();
}

#define INSTANTIATE_LOGLOSS(F, M) template class ONEDAL_EXPORT objective<logloss_objective_t<F, M>>;

INSTANTIATE_LOGLOSS(float, logloss_objective::method::dense_batch)
INSTANTIATE_LOGLOSS(double, logloss_objective::method::dense_batch)

} // namespace oneapi::dal::objective_function::detail
