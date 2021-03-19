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

#include <type_traits>

#include "oneapi/dal/backend/primitives/reduction/reduction.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_rw.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <class Float, ndorder Layout, class BinaryOp, class UnaryOp>
sycl::event reduce_rm_rw(sycl::queue& q,
                         ndview<Float, 2, Layout>& input,
                         ndview<Float, 1>& output,
                         const BinaryOp binary,
                         const UnaryOp unary,
                         const event_vector& deps) {
    using kernel_t = reduction_rm_rw<Float, BinaryOp, UnaryOp>;
    const auto width = input.get_dimension(1);
    const auto height = input.get_dimension(0);
    const auto stride = input.get_leading_stride();
    const auto* inp_ptr = input.get_data();
    auto* out_ptr = output.get_mutable_data();
    const kernel_t kernel(q);
    return kernel(inp_ptr, out_ptr, width, height, stride, binary, unary, deps);
}

template <class Float, ndorder Layout, class BinaryOp, class UnaryOp>
sycl::event reduce_rm_cw(sycl::queue& q,
                         ndview<Float, 2, Layout>& input,
                         ndview<Float, 1>& output,
                         const BinaryOp binary,
                         const UnaryOp unary,
                         const event_vector& deps) {
    using kernel_t = reduction_rm_cw<Float, BinaryOp, UnaryOp>;
    const auto width = input.get_dimension(1);
    const auto height = input.get_dimension(0);
    const auto stride = input.get_leading_stride();
    const auto* inp_ptr = input.get_data();
    auto* out_ptr = output.get_mutable_data();
    const kernel_t kernel(q);
    return kernel(inp_ptr, out_ptr, width, height, stride, binary, unary, deps);
}

template <class Float, ndorder Layout, class BinaryOp, class UnaryOp>
sycl::event reduce_rows(sycl::queue& q,
                        ndview<Float, 2, Layout>& input,
                        ndview<Float, 1>& output,
                        const BinaryOp binary,
                        const UnaryOp unary,
                        const event_vector& deps) {
    ONEDAL_ASSERT(input.get_dimension(0) <= output.get_dimension(0));
    if constexpr (Layout == ndorder::c) {
        return reduce_rm_rw(q, input, output, binary, unary, deps);
    }
    else {
        auto input_tr = input.t();
        return reduce_rm_cw(q, input_tr, output, binary, unary, deps);
    }
    ONEDAL_ASSERT(false);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
    });
}

template <class Float, ndorder Layout, class BinaryOp, class UnaryOp>
sycl::event reduce_cols(sycl::queue& q,
                        ndview<Float, 2, Layout>& input,
                        ndview<Float, 1>& output,
                        const BinaryOp binary,
                        const UnaryOp unary,
                        const event_vector& deps) {
    ONEDAL_ASSERT(input.get_dimension(1) <= output.get_dimension(0));
    if constexpr (Layout == ndorder::c) {
        return reduce_rm_cw(q, input, output, binary, unary, deps);
    }
    else {
        auto input_tr = input.t();
        return reduce_rm_rw(q, input_tr, output, binary, unary, deps);
    }
    ONEDAL_ASSERT(false);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
    });
}

#define INSTANTIATE(F, L, B, U)                                        \
    template sycl::event reduce_rows<F, L, B, U>(sycl::queue&,         \
                                                 ndview<F, 2, L>&,     \
                                                 ndview<F, 1>&,        \
                                                 const B,              \
                                                 const U,              \
                                                 const event_vector&); \
    template sycl::event reduce_cols<F, L, B, U>(sycl::queue&,         \
                                                 ndview<F, 2, L>&,     \
                                                 ndview<F, 1>&,        \
                                                 const B,              \
                                                 const U,              \
                                                 const event_vector&);

#define INSTANTIATE_LAYOUT(F, B, U)  \
    INSTANTIATE(F, ndorder::c, B, U) \
    INSTANTIATE(F, ndorder::f, B, U)

#define INSTANTIATE_FLOAT(B, U)                       \
    INSTANTIATE_LAYOUT(double, B<double>, U<double>); \
    INSTANTIATE_LAYOUT(float, B<float>, U<float>);

INSTANTIATE_FLOAT(min, identity)
INSTANTIATE_FLOAT(min, abs)
INSTANTIATE_FLOAT(min, square)

INSTANTIATE_FLOAT(max, identity)
INSTANTIATE_FLOAT(max, abs)
INSTANTIATE_FLOAT(max, square)

INSTANTIATE_FLOAT(sum, identity)
INSTANTIATE_FLOAT(sum, abs)
INSTANTIATE_FLOAT(sum, square)

#undef INSTANTIATE_FLOAT

#undef INSTANTIATE_LAYOUT

#undef INSTANTIATE

#endif

} // namespace oneapi::dal::backend::primitives
