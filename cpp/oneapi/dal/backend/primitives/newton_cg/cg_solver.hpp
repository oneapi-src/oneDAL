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

namespace oneapi::dal::backend::primitives {

template <typename Float, typename MatrixOperator>
sycl::event cg_solve(sycl::queue& queue,
                     MatrixOperator& mul_operator,
                     const ndview<Float, 1>& b,
                     ndview<Float, 1>& x,
                     ndview<Float, 1>& residual,
                     ndview<Float, 1>& conj_vector,
                     ndview<Float, 1>& buffer,
                     const Float tol = 1e-5,
                     const Float atol = -1,
                     const std::int32_t maxiter = 100,
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

} // namespace oneapi::dal::backend::primitives
