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

#include <limits>
#include "oneapi/dal/test/engine/linalg.hpp"

namespace oneapi::dal::test::engine {

template <typename Float>
inline double get_tolerance(double f32_tol, double f64_tol) {
    static_assert(std::is_same_v<Float, float> || std::is_same_v<Float, double>,
                  "Only single or double precision is allowed");

    if constexpr (std::is_same_v<Float, float>) {
        return f32_tol;
    }

    if constexpr (std::is_same_v<Float, double>) {
        return f64_tol;
    }

    return 0.0;
}

template <typename Float>
inline double get_tolerance_for_sum(std::int64_t element_count) {
    constexpr Float eps = std::numeric_limits<Float>::epsilon();
    const Float nu = eps * element_count;
    return nu / (1.0 - nu);
}

template <typename Reference, typename Actual>
inline double l_inf_norm(const Reference& ref, const Actual& actual) {
    const auto ref_mat = linalg::matrix<double>::wrap(ref);
    const auto act_mat = linalg::matrix<double>::wrap(actual);
    return linalg::l_inf_norm(ref_mat, act_mat);
}

template <typename Reference, typename Actual>
inline double abs_error(const Reference& ref, const Actual& actual) {
    const auto ref_mat = linalg::matrix<double>::wrap(ref);
    const auto act_mat = linalg::matrix<double>::wrap(actual);
    return linalg::abs_error(ref_mat, act_mat);
}

template <typename Reference, typename Actual>
inline double rel_error(const Reference& ref, const Actual& actual, double tol) {
    const auto ref_mat = linalg::matrix<double>::wrap(ref);
    const auto act_mat = linalg::matrix<double>::wrap(actual);
    return linalg::rel_error(ref_mat, act_mat, tol);
}

template <typename Container>
inline bool has_nans(const Container& container) {
    const auto container_mat = linalg::matrix<double>::wrap(container);

    const double* mat_ptr = container_mat.get_data();
    for (std::int64_t i = 0; i < container_mat.get_count(); i++) {
        if (mat_ptr[i] != mat_ptr[i]) {
            return true;
        }
    }

    return false;
}

template <typename Container>
inline bool has_no_nans(const Container& container) {
    return !has_nans<Container>(container);
}

} // namespace oneapi::dal::test::engine
