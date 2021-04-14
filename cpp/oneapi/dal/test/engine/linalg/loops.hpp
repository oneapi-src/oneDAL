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

#include "oneapi/dal/test/engine/linalg/matrix.hpp"

namespace oneapi::dal::test::engine::linalg {

template <typename T, layout lyt, typename Op>
inline void enumerate_row_first(const matrix<T, lyt>& m, Op&& op) {
    m.check_if_host_accessible();
    for (std::int64_t i = 0; i < m.get_row_count(); i++) {
        for (std::int64_t j = 0; j < m.get_column_count(); j++) {
            op(i, j, m.get(i, j));
        }
    }
}

template <typename T, layout lyt, typename Op>
inline void enumerate_column_first(const matrix<T, lyt>& m, Op&& op) {
    m.check_if_host_accessible();
    for (std::int64_t j = 0; j < m.get_column_count(); j++) {
        for (std::int64_t i = 0; i < m.get_row_count(); i++) {
            op(i, j, m.get(i, j));
        }
    }
}

template <typename T, layout lyt, typename Op>
inline void enumerate_linear(const matrix<T, lyt>& m, Op&& op) {
    m.check_if_host_accessible();
    const T* m_ptr = m.get_data();
    for (std::int64_t i = 0; i < m.get_count(); i++) {
        op(i, m_ptr[i]);
    }
}

template <typename T, layout lyt, typename Op>
inline void enumerate_linear_mutable(matrix<T, lyt>& m, Op&& op) {
    m.check_if_host_accessible();
    ONEDAL_ASSERT(m.has_mutable_data());
    T* m_ptr = m.get_mutable_data();
    for (std::int64_t i = 0; i < m.get_count(); i++) {
        op(i, m_ptr[i]);
    }
}

template <typename T, layout lyt, typename Op>
inline void enumerate(const matrix<T, lyt>& m, Op&& op) {
    static_assert(lyt == layout::row_major || lyt == layout::column_major,
                  "Only row-major or column-major layouts are supported");

    if constexpr (lyt == layout::row_major) {
        enumerate_row_first(m, std::forward<Op>(op));
    }
    else {
        enumerate_column_first(m, std::forward<Op>(op));
    }
}

template <typename T, layout lyt, typename Op>
inline void for_each(const matrix<T, lyt>& m, Op&& op) {
    m.check_if_host_accessible();
    enumerate_linear(m, [&](std::int64_t i, T x) {
        op(x);
    });
}

} // namespace oneapi::dal::test::engine::linalg
