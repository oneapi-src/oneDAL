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

#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance.hpp"

#include "oneapi/dal/backend/primitives/reduction/reduction.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_squared_l2_norms(sycl::queue& q,
                          const ndview<Float, 2>& inp,
                          ndview<Float, 1>& out,
                          const event_vector& deps) {
    return reduce_by_rows(q, inp, out, sum<Float>{}, square<Float>{}, deps);
}

template <typename Float>
std::tuple<array<Float>, sycl::event> compute_squared_l2_norms(sycl::queue& q,
                                                    const ndview<Float, 2>& inp,
                                                    const event_vector& deps,
                                                    const sycl::usm::alloc& alloc) {
    const auto n_samples = inp.get_dimension(0);
    auto res_array = array<Float>::empty(q, n_samples, alloc);
    auto res_wrapper = ndview<Float, 1>::wrap(res_array.get_mutable_data(), { n_samples });
    return { res_array, compute_squared_l2_norms(q, inp, res_wrapper, deps) };
}

template <typename Float>
sycl::event scatter_2d(sycl::queue& q,
                          const ndview<Float, 1>& inp1,
                          const ndview<Float, 1>& inp2,
                          ndview<Float, 2>& out,
                          const event_vector& deps) {
    const auto out_stride = out.get_leading_stride();
    const auto n_samples1 = inp1.get_dimension(0);
    const auto n_samples2 = inp2.get_dimension(0);
    ONEDAL_ASSERT(n_samples1 <= out.get_dimension(1));
    ONEDAL_ASSERT(n_samples2 <= out.get_dimension(0));
    const auto* inp1_ptr = inp1.get_data();
    const auto* inp2_ptr = inp2.get_data();
    auto* out_ptr = out.get_mutable_data();
    sycl::range<2> out_range(n_samples1, n_samples2);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(out_range, [=](sycl::id<2> idx) {
            auto& out_place = *(out_ptr + out_stride * idx[0] + idx[1]);
            out_place = inp1_ptr[idx[0]] + inp2_ptr[idx[1]];
        });
    });
}

/*template<typename Float>
sycl::event l2_distance<Float>::operator()(sycl::queue& q, 
                                           const ndview<Float, 2>& inp1, 
                                           const ndview<Float, 2>& inp2,
                                           ndview<Float, 2>& out,
                                           const event_vector& deps) {
    this->check_inputs(inp1, inp2, out);
    auto [l2_inp1_arr, l2_inp1_event] = get_norms(q, inp1, deps);
    auto [l2_inp2_arr, l2_inp2_event] = get_norms(q, inp2, deps);
    auto scatter_event = scatter_norms(q, l2_inp1_arr, l2_inp2_arr, out,
                                    { l2_inp1_event, l2_inp2_event });
    return perform_gemm(q, inp1, inp2, out, { scatter_event });
}*/

#define INSTANTIATE(F)                                                                              \
    template std::tuple<array<F>, sycl::event> compute_squared_l2_norms<F>(sycl::queue&,            \
                                                                           const ndview<F, 2>&,     \
                                                                           const event_vector&,     \
                                                                           const sycl::usm::alloc&);\
    template sycl::event scatter_2d<F>(sycl::queue& q, \
                                       const ndview<F, 1>&,\
                                       const ndview<F, 1>&,\
                                       ndview<F, 2>&,\
                                       const event_vector&);


INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
