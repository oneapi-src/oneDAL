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
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_auxilary.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
auto distance<Float, squared_l2_metric<Float>>::get_norms(const ndview<Float, 2>& inp,
                                                          const event_vector& deps) const
    -> norms_res_t {
    auto [res_array, res_event] = compute_squared_l2_norms(q_, inp, deps);
    return { res_array, res_event };
}

template <typename Float>
sycl::event distance<Float, squared_l2_metric<Float>>::operator()(const ndview<Float, 2>& inp1,
                                                                  const ndview<Float, 2>& inp2,
                                                                  ndview<Float, 2>& out,
                                                                  const array<Float> inp1_norms,
                                                                  const array<Float> inp2_norms,
                                                                  const event_vector& deps) const {
    const auto inp1_norms_view =
        ndview<Float, 1>::wrap(inp1_norms.get_data(), { inp1_norms.get_count() });
    const auto inp2_norms_view =
        ndview<Float, 1>::wrap(inp2_norms.get_data(), { inp2_norms.get_count() });
    auto scatter_event = scatter_2d(q_, inp1_norms_view, inp2_norms_view, out, { deps });
    return compute_inner_product(q_, inp1, inp2, out, { scatter_event });
}

template <typename Float>
sycl::event distance<Float, squared_l2_metric<Float>>::operator()(const ndview<Float, 2>& inp1,
                                                                  const ndview<Float, 2>& inp2,
                                                                  ndview<Float, 2>& out,
                                                                  const event_vector& deps) const {
    auto [norms1_array, norms1_event] = get_norms(inp1, deps);
    auto [norms2_array, norms2_event] = get_norms(inp2, deps);
    return this->operator()(inp1,
                            inp2,
                            out,
                            norms1_array,
                            norms2_array,
                            { norms1_event, norms2_event });
}

#define INSTANTIATE(F) template class distance<F, squared_l2_metric<F>>;

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
