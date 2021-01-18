/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/backend/linalg/loops.hpp"

namespace oneapi::dal::backend::linalg {

template <typename T, layout lyt>
matrix<T, lyt> difference(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    ONEDAL_ASSERT(lhs.get_shape() == rhs.get_shape(),
                  "Matrices must have the same shape");

    auto res = matrix<T, lyt>::empty(lhs.get_shape());

    const T* lhs_ptr = lhs.get_data();
    const T* rhs_ptr = rhs.get_data();
    T* res_ptr = res.get_mutable_data();

    for (std::int64_t i = 0; i < lhs.get_count(); i++) {
        res_ptr[i] = lhs_ptr[i] - rhs_ptr[i];
    }

    return res;
}

template <typename T, layout lyt>
matrix<T, lyt> abs(const matrix<T, lyt>& m) {
    auto res = matrix<T, lyt>::empty(m.get_shape());

    const T* m_ptr = res.get_data();
    T* res_ptr = res.get_mutable_data();

    for (std::int64_t i = 0; i < m.get_count(); i++) {
        res_ptr[i] = m_ptr[i];
    }

    return res;
}

template <typename T, layout lyt>
T max(const matrix<T, lyt>& m) {
    if (m.has_data()) {
        return T(0);
    }

    T max_value = m.get(0);
    const T* m_ptr = m.get_data();

    for (std::int64_t i = 0; i < m.get_count(); i++) {
        max_value = std::max(max_value, m_ptr[i]);
    }

    return max_value;
}



} // oneapi::dal::backend::linalg
