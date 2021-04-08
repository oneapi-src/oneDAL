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

#include <cmath>

#include "oneapi/dal/backend/primitives/distance/distance.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

struct distance_metric_tag;

template<typename Float>
struct metric_base {
public:
    using tag_t = distance_metric_tag;
    template<typename InputIt1, typename InputIt2>
    Float operator() (InputIt1 first1,
                      InputIt1 last1, 
                      InputIt2 first2) const;
};

template<typename Float>
struct lp_metric : public metric_base<Float> {
public:
    lp_metric(Float p = 1.0) : p_{ p } {}
    template<typename InputIt1, typename InputIt2>
    Float operator() (InputIt1 first1,
                      InputIt1 last1, 
                      InputIt2 first2) const {
        Float acc = 0;
        auto it1 = first1;
        auto it2 = first2;
        for(; it1 != last1; ++it1, ++it2) {
            const Float adiff = std::abs(*it1 - *it2);
            acc += std::pow(adiff, get_p());
        }
        return std::pow(acc, Float(1) / get_p());
    }
    const Float& get_p() const {
        return p_;
    }
private:
    const Float p_;
};

template<typename Float>
struct l2_metric : public lp_metric<Float> {
public:
    l2_metric() : lp_metric<Float>(2.0) {}
};



#endif

} // namespace oneapi::dal::backend::primitives
