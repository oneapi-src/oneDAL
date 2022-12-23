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

#include "oneapi/dal/backend/primitives/reduction/reduction_1d.hpp"


namespace oneapi::dal::backend::primitives {

template<typename Float, typename BinaryOp, typename UnaryOp>
Float reduce_1d_impl(sycl::queue& q,
                                const ndview<Float, 1>& input,
                                const BinaryOp& binary,
                                const UnaryOp& unary,
                                const event_vector& deps) {
    // auto [out, out_event] = ndarray<Float, 1>::full(q, std::int64_t(1), binary.init_value, sycl::usm::alloc::device);
    const auto full_deps = deps;
    Float result = Float(binary.init_value);
    sycl::buffer<Float> acc_buf{ &result, 1 };

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        const auto* const inp_ptr = input.get_data();
        // auto out_ptr = out.get_mutable_data();
        
        auto accum = reduction(acc_buf, cgh, binary.native);
        cgh.parallel_for(make_range_1d(input.get_count()), accum, 
            [=](sycl::id<1> idx, auto& acc) { 
                acc.combine(unary(inp_ptr[idx])); 
            }
        );
    });
    event.wait_and_throw();
    return acc_buf.get_host_access()[0];



}


#define INSTANTIATE(F, B, U)   template F reduce_1d_impl<F, B, U>(sycl::queue&, \
                                const ndview<F, 1>&, \
                                const B&, \
                                const U&,\
                                const event_vector&);


#define INSTANTIATE_FLOAT(B, U)                       \
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

#undef INSTANTIATE_FLOAT

#undef INSTANTIATE_LAYOUT

#undef INSTANTIATE

} //