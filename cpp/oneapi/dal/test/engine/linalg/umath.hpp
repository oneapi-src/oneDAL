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

#include "oneapi/dal/test/engine/linalg/loops.hpp"

namespace oneapi::dal::test::engine::linalg {

template <typename T, layout lyt, typename Op>
struct unary_op_result {
    using element_type = decltype(std::declval<Op>()(std::declval<T>()));
    using type = matrix<element_type, lyt>;
};

template <typename T, layout lyt, typename Op>
struct binary_op_result {
    using element_type = decltype(std::declval<Op>()(std::declval<T>(), std::declval<T>()));
    using type = matrix<element_type, lyt>;
};

template <typename T, layout lyt, typename Op>
using unary_op_result_t = typename unary_op_result<T, lyt, Op>::type;

template <typename T, layout lyt, typename Op>
using binary_op_result_t = typename binary_op_result<T, lyt, Op>::type;

template <typename T, layout lyt, typename Op>
inline unary_op_result_t<T, lyt, Op> elementwise(const matrix<T, lyt>& m, Op&& op) {
    m.check_if_host_accessible();

    using result_matrix_t = unary_op_result_t<T, lyt, Op>;
    auto res = result_matrix_t::empty(m.get_shape());

    const T* m_ptr = m.get_data();
    T* res_ptr = res.get_mutable_data();

    for (std::int64_t i = 0; i < m.get_count(); i++) {
        res_ptr[i] = op(m_ptr[i]);
    }

    return res;
}

template <typename T, layout lyt, typename Op>
inline binary_op_result_t<T, lyt, Op> elementwise(const matrix<T, lyt>& lhs,
                                                  const matrix<T, lyt>& rhs,
                                                  Op&& op) {
    lhs.check_if_host_accessible();
    rhs.check_if_host_accessible();
    ONEDAL_ASSERT(lhs.get_shape() == rhs.get_shape(), "Matrices must have the same shape");

    using result_matrix_t = binary_op_result_t<T, lyt, Op>;
    auto res = result_matrix_t::empty(lhs.get_shape());

    const T* lhs_ptr = lhs.get_data();
    const T* rhs_ptr = rhs.get_data();
    auto res_ptr = res.get_mutable_data();

    for (std::int64_t i = 0; i < lhs.get_count(); i++) {
        res_ptr[i] = op(lhs_ptr[i], rhs_ptr[i]);
    }

    return res;
}

template <typename T, layout lyt, typename Op, typename U = T>
inline U reduce(const matrix<T, lyt>& m, const U& init, Op&& op) {
    m.check_if_host_accessible();

    if (!m.has_data()) {
        return U(0);
    }

    auto res = matrix<U, lyt>::empty(m.get_shape());

    U reduced_value = init;
    const T* m_ptr = m.get_data();

    for (std::int64_t i = 0; i < m.get_count(); i++) {
        reduced_value = op(reduced_value, m_ptr[i]);
    }

    return reduced_value;
}

template <typename T, layout lyt>
inline matrix<T, lyt> add(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::plus<T>{});
}

template <typename T, layout lyt>
inline matrix<T, lyt> subtract(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::minus<T>{});
}

template <typename T, layout lyt>
inline matrix<T, lyt> multiply(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::multiplies<T>{});
}

template <typename T, layout lyt>
inline matrix<T, lyt> multiply(T scalar, const matrix<T, lyt>& m) {
    return elementwise(m, [&](T x) {
        return scalar * x;
    });
}

template <typename T, layout lyt>
inline matrix<T, lyt> divide(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::divides<T>{});
}

template <typename T, layout lyt>
inline matrix<bool, lyt> equal(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::equal_to<T>{});
}

template <typename T, layout lyt>
inline matrix<bool, lyt> not_equal(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, std::not_equal_to<T>{});
}

template <typename T, layout lyt>
inline matrix<bool, lyt> equal_approx(const matrix<T, lyt>& lhs,
                                      const matrix<T, lyt>& rhs,
                                      double epsilon) {
    return elementwise(lhs, rhs, [&](T x, T y) {
        return std::abs(double(x) - double(y)) < epsilon;
    });
}

template <typename T, layout lyt>
inline matrix<T, lyt> abs(const matrix<T, lyt>& m) {
    return elementwise(m, [](T x) {
        return std::abs(x);
    });
}

template <typename T, layout lyt>
inline T max(const matrix<T, lyt>& m) {
    return reduce(m, std::numeric_limits<T>::min(), [](T x, T y) {
        return std::max(x, y);
    });
}

template <typename T, layout lyt>
inline matrix<T, lyt> max(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return elementwise(lhs, rhs, [&](T x, T y) {
        return std::max(x, y);
    });
}

template <typename T, layout lyt>
inline T l_inf_norm(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return max(abs(subtract(lhs, rhs)));
}

template <typename T, layout lyt>
inline T abs_error(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs) {
    return l_inf_norm(lhs, rhs);
}

template <typename T, layout lyt>
inline T rel_error(const matrix<T, lyt>& lhs, const matrix<T, lyt>& rhs, T tol) {
    return max(elementwise(lhs, rhs, [&](T x, T y) {
        const auto div = std::max(std::abs(x), std::abs(y));
        return (div > tol) ? (std::abs(x - y) / div) : T(0);
    }));
}

} // namespace oneapi::dal::test::engine::linalg
