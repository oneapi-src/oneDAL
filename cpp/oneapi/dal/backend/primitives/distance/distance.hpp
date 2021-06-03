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

#pragma once

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/backend/primitives/distance/metrics.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float, typename Metric>
class distance {
public:
    distance(sycl::queue& q, const Metric& m = Metric{}) : q_{ q }, m_{ m } {
        static_assert(dal::detail::is_tag_one_of_v<Metric, distance_metric_tag>,
                      "Metric must be a special operation defined in metrics header");
    }
    sycl::event operator()(const ndview<Float, 2>& inp1,
                           const ndview<Float, 2>& inp2,
                           ndview<Float, 2>& out,
                           const event_vector& deps = {}) const;

private:
    sycl::queue& q_;
    const Metric m_;
};

template <typename Float>
class l2_helper;

template <typename Float>
class distance<Float, squared_l2_metric<Float>> {
public:
    distance(sycl::queue& q) : q_{ q } {};
    sycl::event operator()(const ndview<Float, 2>& inp1,
                           const ndview<Float, 2>& inp2,
                           ndview<Float, 2>& out,
                           const event_vector& deps = {}) const;
    sycl::event operator()(const ndview<Float, 2>& inp1,
                           const ndview<Float, 2>& inp2,
                           ndview<Float, 2>& out,
                           const ndview<Float, 1>& inp1_norms,
                           const ndview<Float, 1>& inp2_norms,
                           const event_vector& deps = {}) const;

protected:
    using norms_res_t = std::tuple<ndarray<Float, 1>, sycl::event>;
    norms_res_t get_norms(const ndview<Float, 2>& inp, const event_vector& deps = {}) const;

private:
    sycl::queue& q_;
};

template <typename Float>
using lp_distance = distance<Float, lp_metric<Float>>;

template <typename Float>
using squared_l2_distance = distance<Float, squared_l2_metric<Float>>;

template <typename Float>
void check_inputs(const ndview<Float, 2>& inp1,
                  const ndview<Float, 2>& inp2,
                  const ndview<Float, 2>& out);

#endif

} // namespace oneapi::dal::backend::primitives
