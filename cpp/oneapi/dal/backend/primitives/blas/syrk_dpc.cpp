/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/primitives/blas/syrk.hpp"
#include "oneapi/dal/backend/primitives/blas/misc.hpp"

#include <mkl_dal_sycl.hpp>

namespace oneapi::dal::backend::primitives {

template <mkl::uplo ul, typename Float, ndorder ao, ndorder co>
sycl::event syrk(sycl::queue& queue,
                 const ndview<Float, 2, ao>& a,
                 ndview<Float, 2, co>& c,
                 Float alpha,
                 Float beta,
                 const event_vector& deps) {
    ONEDAL_ASSERT(a.get_dimension(1) == c.get_dimension(0));
    ONEDAL_ASSERT(c.get_dimension(0) == c.get_dimension(1));
    ONEDAL_ASSERT(c.has_mutable_data());

    const auto nd = c.get_dimension(0);
    const auto kd = a.get_dimension(1);
    const auto* const a_ptr = a.get_data();
    auto* const c_ptr = c.get_mutable_data();
    const auto a_str = a.get_leading_stride();
    const auto c_str = c.get_leading_stride();
    constexpr bool is_c_trans = (co == ndorder::c);
    if constexpr (is_c_trans) {
        constexpr auto tr = f_order_as_transposed(ao);
        return mkl::blas::syrk(queue,
                               ul,
                               tr,
                               nd,
                               kd,
                               alpha,
                               a_ptr,
                               a_str,
                               beta,
                               c_ptr,
                               c_str,
                               deps);
    } else {
        constexpr auto tr = c_order_as_transposed(ao);
        return mkl::blas::syrk(queue,
                               ul,
                               tr,
                               nd,
                               kd,
                               alpha,
                               a_ptr,
                               a_str,
                               beta,
                               c_ptr,
                               c_str,
                               deps);
    }
}

#define INSTANTIATE(ul, F, ao, co)                                                \
    template ONEDAL_EXPORT sycl::event syrk<ul, F, ao, co>(sycl::queue & queue,       \
                                                           const ndview<F, 2, ao>& a, \
                                                           ndview<F, 2, co>& c,       \
                                                           F alpha,                   \
                                                           F beta,                    \
                                                           const event_vector& deps);

#define INSTANTIATE_FLOAT(ul, ao, co) \
    INSTANTIATE(ul, float, ao, co)    \
    INSTANTIATE(ul, double, ao, co)

#define INSTANTIATE_UPLO(ao, co)                   \
    INSTANTIATE_FLOAT(mkl::uplo::upper, ao, co)    \
    INSTANTIATE_FLOAT(mkl::uplo::lower, ao, co)

INSTANTIATE_UPLO(ndorder::c, ndorder::c)
INSTANTIATE_UPLO(ndorder::c, ndorder::f)
INSTANTIATE_UPLO(ndorder::f, ndorder::c)
INSTANTIATE_UPLO(ndorder::f, ndorder::f)

} // namespace oneapi::dal::backend::primitives
