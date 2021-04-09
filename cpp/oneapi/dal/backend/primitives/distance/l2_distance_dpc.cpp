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
#include "oneapi/dal/backend/primitives/reduction/reduction.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event compute_norms(sycl::queue& q,
                          const ndview<Float, 2>& inp,
                          ndview<Float, 1>& out,
                          const event_vector& deps = {}) {
    return reduce_by_rows(q, inp, out, sum<Float>{}, square<Float>{}, deps);
}

template <typename Float>
std::tuple<array<Float>, sycl::event> compute_norms(sycl::queue& q,
                                                    const ndview<Float, 2>& inp,
                                                    const event_vector& deps = {}) {
    const auto n_samples = inp.get_dimension(0);
    auto res_array = array<Float>::empty(q, n_samples, sycl::usm::alloc::device);
    auto res_wrapper = ndview<Float, 1>::wrap(res_array.get_mutable_data(), { n_samples });
    return { res_array, compute_norms(q, inp, res_wrapper, deps) };
}

template <typename Float, bool full = true>
sycl::event scatter_norms(sycl::queue& q,
                          ndview<Float, 1>& norms1,
                          ndview<Float, 1>& norms2,
                          ndview<Float, 2>& out,
                          const event_vector& deps = {}) {
    const auto out_stride = out.get_leading_stride();
    const auto n_samples1 = out.get_dimension(0);
    const auto n_samples2 = out.get_dimension(0);
    ONEDAL_ASSERT(n_samples1 <= out.get_dimension(1));
    ONEDAL_ASSERT(n_samples2 <= out.get_dimension(0));
    //sycl::range<2>
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        /*h.parallel_for<class scatter_full> {

        }*/
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

#define INSTANTIATE(F)                                                               \
    template std::tuple<array<F>, sycl::event> compute_norms<F>(sycl::queue&,        \
                                                                const ndview<F, 2>&, \
                                                                const event_vector&);
//  template class l2_distance<F>;

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
