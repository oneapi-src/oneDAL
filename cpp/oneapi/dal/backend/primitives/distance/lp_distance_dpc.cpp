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

namespace oneapi::dal::backend::primitives {

template<typename Float, typename Metric>
sycl::event distance<Float, Metric>::initialize(const ndview<Float, 2>& inp1, 
                                                const ndview<Float, 2>& inp2,
                                                const event_vector& deps) {
    return q_.submit([&](sycl::handler& h) { h.depends_on(deps); });
}

template<typename Float, typename Metric>
sycl::event distance<Float, Metric>::operator()(const ndview<Float, 2>& inp1, 
                                                const ndview<Float, 2>& inp2,
                                                ndview<Float, 2>& out,
                                                const event_vector& deps) const {
    check_inputs(inp1, inp2, out);
    // Getting raw USM pointers
    const auto* inp1_ptr = inp1.get_data();
    const auto* inp2_ptr = inp2.get_data();
    auto* out_ptr = inp2.get_mutable_data();
    // Getting info about dimensions
    const auto n_features = inp1.get_dimension(1);
    const auto n_samples1 = inp1.get_dimension(0);
    const auto n_samples2 = inp2.get_dimension(0);
    // Getting info about strides
    const auto inp_stride1 = inp1.get_leading_stride(); 
    const auto inp_stride2 = inp2.get_leading_stride(); 
    const auto out_stride = out.get_leading_stride();
    // Constructing colrrect range of size m x n 
    sycl::range<2> out_range(n_samples1, n_samples2);
    // Metric instance 
    const auto& metric = this->m_;
    return q_.submit(
        [&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for<class dist_comp>(
                out_range, 
                [=](sycl::id<2> idx) {
                    const auto* inp1_first = inp1_ptr + inp_stride1 * idx[0];
                    const auto* inp1_last = inp1_first + n_features;
                    const auto* inp2_first = inp2_ptr + inp_stride2 * idx[1];
                    auto& out_place = *(out_ptr + out_stride * idx[0] + idx[1]); 
                    out_place = metric(inp1_first, inp1_last, inp2_first);
                }
            );
        }
    );
}




#define INSTANTIATE(F) template class distance<F, lp_metric<F>>;

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
