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

#include <cmath>
#include <ostream>
#include <iomanip>
#include <algorithm>
#include <functional>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::linalg {

enum class layout {
    row_major,
    column_major,
};

class shape {
public:
    shape(std::int64_t row_count = 0, std::int64_t column_count = 0) {
        ONEDAL_ASSERT(row_count >= 0, "Row count must be non-negative");
        ONEDAL_ASSERT(column_count >= 0, "Column count must be non-negative");
        if (row_count == 0 || column_count == 0) {
            ONEDAL_ASSERT(row_count == 0,
                          "Both row count and column count must be zeros, "
                          "but got non-zero row count");
            ONEDAL_ASSERT(column_count == 0,
                          "Both row count and column count must be zeros, "
                          "but got non-zero column count");
        }
        shape_[0] = row_count;
        shape_[1] = column_count;
        ONEDAL_ASSERT(count() / columns() == rows(), "Shape count overflow");
    }

    std::int64_t operator[](std::int64_t i) const {
        ONEDAL_ASSERT(i == 0 || i == 1, "Index can be only 0 or 1");
        return shape_[i];
    }

    std::int64_t rows() const {
        return shape_[0];
    }

    std::int64_t columns() const {
        return shape_[1];
    }

    std::int64_t count() const {
        return rows() * columns();
    }

    shape T() const {
        return shape{ shape_[1], shape_[0] };
    }

    bool operator==(const shape& other) const {
        return shape_[0] == other.shape_[0] && shape_[1] == other.shape_[1];
    }

    bool operator!=(const shape& other) const {
        return !(*this == other);
    }

private:
    std::int64_t shape_[2];
};

class matrix_base {
public:
    std::int64_t get_row_count() const {
        return s_.rows();
    }

    std::int64_t get_column_count() const {
        return s_.columns();
    }

    std::int64_t get_count() const {
        return s_.count();
    }

    const shape& get_shape() const {
        return s_;
    }

    layout get_layout() const {
        return l_;
    }

    std::int64_t get_stride() const {
        return stride_;
    }

protected:
    explicit matrix_base(const shape& s, layout l, std::int64_t stride)
            : s_(s),
              l_(l),
              stride_(stride) {
        if (l == layout::row_major) {
            ONEDAL_ASSERT(stride >= s.columns(),
                          "Stride must be greater than "
                          "column count in row-major layout");
        }
        else if (l == layout::column_major) {
            ONEDAL_ASSERT(stride >= s.rows(),
                          "Stride must be greater than "
                          "row count in column-major layout");
        }
    }

private:
    shape s_;
    layout l_;
    std::int64_t stride_;
};

template <typename Float>
class matrix : public matrix_base {
public:
    static matrix wrap(const array<Float>& x, const shape& s, layout l = layout::row_major) {
        return matrix{ x, s, l, get_default_stride(s, l) };
    }

    static matrix wrap(const table& t, layout l = layout::row_major) {
        if (l != layout::row_major) {
            // TODO: Figure out how to use column-major layout
            throw unimplemented{ dal::detail::error_messages::unsupported_data_layout() };
        }
        const auto t_flat = row_accessor<const Float>{ t }.pull();
        return wrap(t_flat, { t.get_row_count(), t.get_column_count() });
    }

    static matrix empty(const shape& s, layout l = layout::row_major) {
        return wrap(array<Float>::empty(s.count()), s, l);
    }

    template <typename Filler>
    static matrix full(const shape& s, Filler&& filler, layout l = layout::row_major) {
        return matrix<Float>::empty(s, l).fill(std::forward<Filler>(filler));
    }

    static matrix ones(const shape& s, layout l = layout::row_major) {
        return matrix<Float>::full(s, Float(1), l);
    }

    static matrix zeros(const shape& s, layout l = layout::row_major) {
        return matrix<Float>::full(s, Float(0), l);
    }

    static matrix eye(std::int64_t dim, layout l = layout::row_major) {
        auto m = zeros({ dim, dim }, l);
        Float* data = m.get_mutable_data();
        for (std::int64_t i = 0; i < dim; i++) {
            data[i * dim + i] = Float(1);
        }
        return m;
    }

    matrix() : matrix_base({ 0, 0 }, layout::row_major, 0) {}

    const array<Float>& get_array() const {
        return x_;
    }

    array<Float>& get_array() {
        return x_;
    }

    const Float* get_data() const {
        return x_.get_data();
    }

    Float* get_mutable_data() const {
        return x_.get_mutable_data();
    }

    matrix& need_mutable_data() {
        x_.need_mutable_data();
        return *this;
    }

    bool has_mutable_data() const {
        return x_.has_mutable_data();
    }

    matrix T() const {
        const shape t_s = get_shape().T();
        const layout t_l = transpose_layout(get_layout());
        return matrix<Float>{ x_, t_s, t_l, get_stride() };
    }

    template <typename U, typename Op>
    auto binary_op(const matrix<U>& other, Op&& op) const {
        ONEDAL_ASSERT(get_shape() == other.get_shape());
        ONEDAL_ASSERT(get_layout() == other.get_layout());

        using T = Float;
        const T* lhs_data = get_data();
        const U* rhs_data = other.get_data();

        using O = decltype(op(std::declval<T>(), std::declval<U>()));
        auto m = matrix<O>::empty(get_shape(), get_layout());

        return m.mutable_enumerate_linear([&](std::int64_t i, O& o) {
            o = op(lhs_data[i], rhs_data[i]);
        });
    }

    matrix operator+(const matrix& other) const {
        return binary_op(other, std::plus<Float>{});
    }

    matrix operator-(const matrix& other) const {
        return binary_op(other, std::minus<Float>{});
    }

    matrix operator*(const matrix& other) const {
        return binary_op(other, std::multiplies<Float>{});
    }

    matrix operator/(const matrix& other) const {
        return binary_op(other, std::divides<Float>{});
    }

    Float get(std::int64_t i, std::int64_t j) const {
        return get_data()[get_linear_index(i, j)];
    }

    Float& set(std::int64_t i, std::int64_t j) {
        return get_mutable_data()[get_linear_index(i, j)];
    }

    template <typename Op>
    auto& enumerate_linear(Op&& op) const {
        const Float* data = get_data();
        for (std::int64_t i = 0; i < get_count(); i++) {
            op(i, data[i]);
        }
        return *this;
    }

    template <typename Op>
    auto& mutable_enumerate_linear(Op&& op) {
        Float* mutable_data = get_mutable_data();
        for (std::int64_t i = 0; i < get_count(); i++) {
            op(i, mutable_data[i]);
        }
        return *this;
    }

#define ENUMERATE_LOOP_IJ()                            \
    for (std::int64_t i = 0; i < get_row_count(); i++) \
        for (std::int64_t j = 0; j < get_column_count(); j++)

#define ENUMERATE_LOOP_JI()                               \
    for (std::int64_t j = 0; j < get_column_count(); j++) \
        for (std::int64_t i = 0; i < get_row_count(); i++)

#define ENUMERATE_IMPL(ptr, ROW_MAJOR_LOOP, COLUMN_MAJOR_LOOP) \
    const std::int64_t stride = get_stride();                  \
    if (get_layout() == layout::row_major) {                   \
        ROW_MAJOR_LOOP() op(i, j, ptr[i * stride + j]);        \
    }                                                          \
    else {                                                     \
        COLUMN_MAJOR_LOOP() op(i, j, ptr[j * stride + i]);     \
    }

    template <typename Op>
    auto& enumerate(Op&& op) const {
        const Float* data = get_data();
        ENUMERATE_IMPL(data, ENUMERATE_LOOP_IJ, ENUMERATE_LOOP_JI)
        return *this;
    }

    template <typename Op>
    auto& enumerate_row_first(Op&& op) const {
        const Float* data = get_data();
        ENUMERATE_IMPL(data, ENUMERATE_LOOP_IJ, ENUMERATE_LOOP_IJ)
        return *this;
    }

    template <typename Op>
    auto& enumerate_column_first(Op&& op) const {
        const Float* data = get_data();
        ENUMERATE_IMPL(data, ENUMERATE_LOOP_JI, ENUMERATE_LOOP_JI)
        return *this;
    }

    template <typename Op>
    auto& mutable_enumerate(Op&& op) {
        Float* mutable_data = get_mutable_data();
        ENUMERATE_IMPL(mutable_data, ENUMERATE_LOOP_IJ, ENUMERATE_LOOP_JI)
        return *this;
    }

    template <typename Op>
    auto& mutable_enumerate_row_first(Op&& op) {
        Float* mutable_data = get_mutable_data();
        ENUMERATE_IMPL(mutable_data, ENUMERATE_LOOP_IJ, ENUMERATE_LOOP_IJ)
        return *this;
    }

    template <typename Op>
    auto& mutable_enumerate_column_first(Op&& op) {
        Float* mutable_data = get_mutable_data();
        ENUMERATE_IMPL(mutable_data, ENUMERATE_LOOP_JI, ENUMERATE_LOOP_JI)
        return *this;
    }

#undef ENUMERATE_LOOP_IJ
#undef ENUMERATE_LOOP_JI
#undef ENUMERATE_IMPL

    template <typename Op>
    auto& for_each(Op&& op) const {
        const Float* data = get_data();
        for (std::int64_t i = 0; i < get_count(); i++) {
            op(data[i]);
        }
        return *this;
    }

    template <typename Op>
    auto& mutable_for_each(Op&& op) {
        Float* mutable_data = get_mutable_data();
        for (std::int64_t i = 0; i < get_count(); i++) {
            op(mutable_data[i]);
        }
        return *this;
    }

    template <typename Op>
    auto& fill(Op&& op) {
        return mutable_enumerate([&](std::int64_t i, std::int64_t j, Float& x) {
            x = op(i, j);
        });
    }

    auto& fill(Float x) {
        return mutable_for_each([&](Float& y) {
            y = x;
        });
    }

    template <typename Op>
    Float reduce(Op&& op) const {
        Float reduced = get(0, 0);
        for_each([&](Float x) mutable {
            reduced = op(reduced, x);
        });
        return reduced;
    }

    Float max() const {
        return reduce([](Float x, Float y) {
            return std::max(x, y);
        });
    }

    template <typename Op>
    auto map(Op&& op) const {
        using T = decltype(op(std::declval<Float>()));
        auto mapped = matrix<T>::empty(get_shape());

        Float* mapped_ptr = mapped.get_mutable_data();
        const std::int64_t mapped_stride = mapped.get_stride();
        ONEDAL_ASSERT(mapped.get_layout() == layout::row_major);

        enumerate_row_first([&](std::int64_t i, std::int64_t j, Float x) mutable {
            mapped_ptr[i * mapped_stride + j] = op(x);
        });

        return mapped;
    }

    matrix abs() const {
        return map([](Float x) {
            return std::abs(x);
        });
    }

private:
    explicit matrix(const array<Float>& x, const shape& s, layout l, std::int64_t stride)
            : matrix_base(s, l, stride),
              x_(x) {
        ONEDAL_ASSERT(s.count() <= x.get_count(),
                      "Element count in matrix does not match "
                      "element count in the provided array");
    }

    std::int64_t get_linear_index(std::int64_t i, std::int64_t j) const {
        ONEDAL_ASSERT(i >= 0 && i < get_row_count(), "Row index is out of range");
        ONEDAL_ASSERT(j >= 0 && j < get_column_count(), "Column index is out of range");
        if (get_layout() == layout::row_major) {
            return i * get_stride() + j;
        }
        else {
            return j * get_stride() + i;
        }
    }

    static std::int64_t get_default_stride(const shape& s, layout l) {
        return (l == layout::row_major) ? s.columns() : s.rows();
    }

    static layout transpose_layout(layout l) {
        return (l == layout::row_major) ? layout::column_major : layout::row_major;
    }

    array<Float> x_;
};

template <typename Float>
std::ostream& operator<<(std::ostream& stream, const matrix<Float>& m) {
    m.enumerate_row_first([&](std::int64_t i, std::int64_t j, Float x) {
        stream << std::setw(10);
        stream << std::setiosflags(std::ios::fixed);
        stream << std::setprecision(3);
        stream << x;
        if (j + 1 == m.get_column_count()) {
            stream << std::endl;
        }
    });
    return stream;
}

} // namespace oneapi::dal::backend::linalg
