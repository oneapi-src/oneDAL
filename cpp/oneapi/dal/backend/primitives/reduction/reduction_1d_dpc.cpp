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

#include <type_traits>

#include "oneapi/dal/backend/primitives/reduction/reduction_1d.hpp"
#include "oneapi/dal/backend/primitives/common.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduce_1d_impl(sycl::queue& queue,
                           const ndview<Float, 1>& inp,
                           ndview<Float, 1>& out,
                           const BinaryOp& binary,
                           const UnaryOp& unary,
                           const event_vector& deps) {
    ONEDAL_ASSERT(inp.has_data());
    ONEDAL_ASSERT(out.has_mutable_data());
    ONEDAL_ASSERT(out.get_count() == std::int64_t(1));
    return queue.submit([&](sycl::handler& h) {
        using sycl::reduction;
        h.depends_on(deps);
        const auto* const inp_ptr = inp.get_data();
        auto* const out_ptr = out.get_mutable_data();
        auto accum = reduction(out_ptr, binary.native);
        h.parallel_for(make_range_1d(inp.get_count()), accum, [=](sycl::id<1> idx, auto& acc) {
            acc.combine(unary(inp_ptr[idx]));
        });
    });
}

template <typename Float, typename BinaryOp, typename UnaryOp>
Float reduce_1d_impl(sycl::queue& queue,
                     const ndview<Float, 1>& inp,
                     const BinaryOp& binary,
                     const UnaryOp& unary,
                     const event_vector& deps) {
    using oneapi::dal::backend::operator+;
    auto [out, out_event] = ndarray<Float, 1>::full(queue,
                                                    std::int64_t(1),
                                                    binary.init_value,
                                                    sycl::usm::alloc::device);
    const auto full_deps = deps + out_event;
    auto event = reduce_1d_impl(queue, inp, out, binary, unary, full_deps);
    return out.to_host(queue, { event }).at(0);
}

#define INSTANTIATE(F, B, U)                                          \
    template F reduce_1d_impl<F, B, U>(sycl::queue&,                  \
                                       const ndview<F, 1>&,           \
                                       const B&,                      \
                                       const U&,                      \
                                       const event_vector&);          \
    template sycl::event reduce_1d_impl<F, B, U>(sycl::queue&,        \
                                                 const ndview<F, 1>&, \
                                                 ndview<F, 1>&,       \
                                                 const B&,            \
                                                 const U&,            \
                                                 const event_vector&);

#define INSTANTIATE_FLOAT(B, U)             \
    INSTANTIATE(float, B<float>, U<float>); \
    INSTANTIATE(double, B<double>, U<double>);

INSTANTIATE_FLOAT(min, identity)
INSTANTIATE_FLOAT(min, abs)
INSTANTIATE_FLOAT(min, square)

INSTANTIATE_FLOAT(max, identity)
INSTANTIATE_FLOAT(max, abs)
INSTANTIATE_FLOAT(max, square)

INSTANTIATE_FLOAT(sum, identity)
INSTANTIATE_FLOAT(sum, abs)
INSTANTIATE_FLOAT(sum, square)

INSTANTIATE_FLOAT(logical_or, isinfornan)
INSTANTIATE_FLOAT(logical_or, isinf)

INSTANTIATE_FLOAT(logical_or, isinfornan)
INSTANTIATE_FLOAT(logical_or, isinf)

#undef INSTANTIATE_FLOAT

#undef INSTANTIATE_LAYOUT

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
