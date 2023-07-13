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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event dot_product(sycl::queue& queue,
                        const ndview<Float, 1>& x,
                        const ndview<Float, 1>& y,
                        Float* res_gpu,
                        Float* res_host,
                        const event_vector& deps = {});

template <typename Float>
class matrix_operator {
public:
    matrix_operator(sycl::queue& q, const ndview<Float, 2>& A);

    sycl::event operator()(const ndview<Float, 1>& vec,
                           ndview<Float, 1>& out,
                           const event_vector& deps = {});

private:
    sycl::queue q_;
    const ndview<Float, 2> A_;
};

template <typename Float>
class logloss_function {

logloss_function(sycl::queue& q, const ndview<Float, 2>& data, const ndview<Float, 1>& labels, const Float L2, const bool fit_intercept);

ndview<Float, 1>& get_gradient();

logloss_hessian_product<Float>& get_hessian_product();

sycl::event compute_gradient();

sycl::event compute_hessian_product();



private:
    sycl::queue q_;
    const ndview<Float, 2> data_;
    const ndview<Float, 1> labels_;
    ndview<Float, 1> probabilities_;
    const Float L2_;
    const bool fit_intercept_;
    logloss_hessian_product<Float> hessp_;
};


} // namespace oneapi::dal::backend::primitives
