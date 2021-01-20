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

template <typename T, layout lyt, typename Op>
struct unary_operation_result {
    using element_type = decltype(std::declval<Op>()(std::declval<T>()));
    using type = matrix<element_type, lyt>;
};

template <typename T, layout lyt, typename Op>
struct binary_operation_result {
    using element_type = decltype(std::declval<Op>()(std::declval<T>(), std::declval<T>()));
    using type = matrix<element_type, lyt>;
};

template <typename T, layout lyt, typename Op>
using unary_operation_result_t = typename unary_operation_result<T, lyt, Op>::type;

template <typename T, layout lyt, typename Op>
using binary_operation_result_t = typename binary_operation_result<T, lyt, Op>::type;

template <typename T, layout lyt, typename Op>
unary_operation_result_t<T, lyt, Op> elementwise(const matrix<T, lyt>& m, Op&& op) {
    using result_matrix_t = unary_operation_result_t<T, lyt, Op>;
    auto res = result_matrix_t::empty(m.get_shape());

    const T* m_ptr = m.get_data();
    T* res_ptr = res.get_mutable_data();

    for (std::int64_t i = 0; i < m.get_count(); i++) {
        res_ptr[i] = op(m_ptr[i]);
    }

    return res;
}

template <typename T, layout lyt, typename Op>
binary_operation_result_t<T, lyt, Op> elementwise(const matrix<T, lyt>& lhs,
                                                  const matrix<T, lyt>& rhs,
                                                  Op&& op) {
    ONEDAL_ASSERT(lhs.get_shape() == rhs.get_shape(),
                  "Matrices must have the same shape");

    using result_matrix_t = binary_operation_result_t<T, lyt, Op>;
    auto res = result_matrix_t::empty(lhs.get_shape());

    const T* lhs_ptr = lhs.get_data();
    const T* rhs_ptr = rhs.get_data();
    T* res_ptr = res.get_mutable_data();

    for (std::int64_t i = 0; i < lhs.get_count(); i++) {
        res_ptr[i] = op(lhs_ptr[i], rhs_ptr[i]);
    }

    return res;
}

template <typename T, layout lyt, typename Op>
binary_operation_result_t<T, lyt, Op> elementwise_operation(const matrix<T, lyt>& lhs,
                                                            const matrix<T, lyt>& rhs,
                                                            Op&& op) {
    ONEDAL_ASSERT(lhs.get_shape() == rhs.get_shape(),
                  "Matrices must have the same shape");

    using result_matrix_t = binary_operation_result_t<T, lyt, Op>;
    auto res = result_matrix_t::empty(lhs.get_shape());

    const T* lhs_ptr = lhs.get_data();
    const T* rhs_ptr = rhs.get_data();
    T* res_ptr = res.get_mutable_data();

    for (std::int64_t i = 0; i < lhs.get_count(); i++) {
        res_ptr[i] = op(lhs_ptr[i], rhs_ptr[i]);
    }

    return res;
}

template <typename T, layout lyt>
matrix<T, lyt> add(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::plus<T>{});
}

template <typename T, layout lyt>
matrix<T, lyt> subtract(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::minus<T>{});
}

template <typename T, layout lyt>
matrix<T, lyt> multiply(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::multiplies<T>{});
}

template <typename T, layout lyt>
matrix<T, lyt> divide(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::divides<T>{});
}

template <typename T, layout lyt>
matrix<bool, lyt> equal(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::equal_to<T>{});
}

template <typename T, layout lyt>
matrix<bool, lyt> not_equal(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::not_equal_to<T>{});
}

template <typename T, layout lyt>
matrix<bool, lyt> equal_approx(const matrix<T, lyt>& lhs,
                               const matrix<T, lyt>& rhs,
                               double epsilon) {
    return elementwise(lhs, rhs, [&](T x, T y) {
        return std::abs(double(x) - double(y)) < epsilon;
    });
}

template <typename T, layout lyt>
matrix<T, lyt> abs(const matrix<T, lyt>& m) {
    return elementwise(m, [](T x) {
        return std::abs(x);
    });
}

template <typename T, layout lyt>
T max(const matrix<T, lyt>& m) {
    if (!m.has_data()) {
        return T(0);
    }

    T max_value = m.get(0);
    const T* m_ptr = m.get_data();

    for (std::int64_t i = 0; i < m.get_count(); i++) {
        max_value = std::max(max_value, m_ptr[i]);
    }

    return max_value;
}

template <typename T, layout lyt>
T l_inf_norm(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return max(abs(subtract(lhs, rhs)));
}

} // oneapi::dal::backend::linalg
