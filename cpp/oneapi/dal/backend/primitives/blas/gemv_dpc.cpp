/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/backend/primitives/blas/gemv.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

#include <mkl_dal_sycl.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float, ndorder ao>
sycl::event gemv(sycl::queue& queue, const ndview<Float, 2, ao>& a, const ndview<Float, 1>& x, ndview<Float, 1>& y, Float alpha, Float beta, const event_vector& deps) {
    ONEDAL_ASSERT(a.get_dimension(0) == y.get_dimension(0));
    ONEDAL_ASSERT(a.get_dimension(1) == x.get_dimension(0));
    ONEDAL_ASSERT(y.has_mutable_data());
    return mkl::blas::gemv(
        queue, 
        mkl::transpose::nontrans, 
        a.get_dimension(0), 
        a.get_dimension(1), 
        alpha, 
        a.get_data(), 
        a.get_leading_stride(),
        x.get_data(),
        x.get_leading_stride(),
        beta,
        y.get_mutable_data(),
        y.get_leading_stride()
        );
}

#define INSTANTIATE(F, ao)                                                              \
    template ONEDAL_EXPORT sycl::event gemv<F, ao>(sycl::queue & queue,                 \
                                                           const ndview<F, 2, ao>& a,   \
                                                           const ndview<F, 1>& x,       \
                                                           ndview<F, 1>& y,             \
                                                           F alpha,                     \
                                                           F beta,                      \
                                                           const event_vector& deps);


#define INSTANTIATE_FLOAT(ao) \
    INSTANTIATE(float, ao)    \
    INSTANTIATE(double, ao)


INSTANTIATE_FLOAT(ndorder::c);
INSTANTIATE_FLOAT(ndorder::f);

} // namespace oneapi::dal::backend::primitives