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
    sycl::event initialize(const ndview<Float, 2>& inp1,
                           const ndview<Float, 2>& inp2,
                           const event_vector& deps = {});
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
    distance(sycl::queue& q);
    sycl::event initialize(const ndview<Float, 2>& inp1,
                           const ndview<Float, 2>& inp2,
                           const event_vector& deps = {});
    sycl::event operator()(const ndview<Float, 2>& inp1,
                           const ndview<Float, 2>& inp2,
                           ndview<Float, 2>& out,
                           const event_vector& deps = {}) const;
    ~distance();
                           
protected:
    using helper_t = l2_helper<Float>;
    using helper_ptr_t = detail::unique<helper_t>;
    using norms_res_t = std::tuple<const array<Float>, sycl::event>;
    norms_res_t get_norms(const helper_ptr_t& helper,
                          const ndview<Float, 2>& inp,
                          const event_vector& deps = {}) const;

private:
    sycl::event initialize_helper(helper_ptr_t& helper, 
                                  const ndview<Float, 2>& inp1,
                                  const event_vector& deps = {});
    sycl::queue& q_;
    helper_ptr_t helper1_;
    helper_ptr_t helper2_;
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
