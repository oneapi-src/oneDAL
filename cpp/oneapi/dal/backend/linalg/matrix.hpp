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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::linalg {

enum class layout {
    row_major,
    column_major,
};

inline layout transpose_layout(layout l) {
    return (l == layout::row_major) ? layout::column_major : layout::row_major;
}

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
        ONEDAL_ASSERT(get_count() / get_column_count() == get_row_count(),
                      "Shape count overflow");
    }

    std::int64_t operator[](std::int64_t i) const {
        ONEDAL_ASSERT(i == 0 || i == 1, "Index can be only 0 or 1");
        return shape_[i];
    }

    std::int64_t get_row_count() const {
        return shape_[0];
    }

    std::int64_t get_column_count() const {
        return shape_[1];
    }

    std::int64_t get_count() const {
        return get_row_count() * get_column_count();
    }

    shape t() const {
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
        return s_.get_row_count();
    }

    std::int64_t get_column_count() const {
        return s_.get_column_count();
    }

    std::int64_t get_count() const {
        return s_.get_count();
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

protected:
    explicit matrix_base(const shape& s, layout l, std::int64_t stride)
            : s_(s),
              l_(l),
              stride_(stride) {
        if (l == layout::row_major) {
            ONEDAL_ASSERT(stride >= s.get_column_count(),
                          "Stride must be greater than "
                          "column count in row-major layout");
        }
        else if (l == layout::column_major) {
            ONEDAL_ASSERT(stride >= s.get_row_count(),
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
        return wrap(array<Float>::empty(s.get_count()), s, l);
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

    matrix t() const {
        const shape t_s = get_shape().t();
        const layout t_l = transpose_layout(get_layout());
        return matrix<Float>{ x_, t_s, t_l, get_stride() };
    }

    Float get(std::int64_t i, std::int64_t j) const {
        return get_data()[get_linear_index(i, j)];
    }

    Float& set(std::int64_t i, std::int64_t j) {
        return get_mutable_data()[get_linear_index(i, j)];
    }

    auto& fill(Float filler) {
        float* ptr = get_mutable_data();
        for (std::int64_t i = 0; i < get_count(); i++) {
            ptr[i] = filler;
        }
        return *this;
    }

private:
    explicit matrix(const array<Float>& x, const shape& s, layout l, std::int64_t stride)
            : matrix_base(s, l, stride),
              x_(x) {
        ONEDAL_ASSERT(s.get_count() <= x.get_count(),
                      "Element count in matrix does not match "
                      "element count in the provided array");
    }

    static std::int64_t get_default_stride(const shape& s, layout l) {
        return (l == layout::row_major) ? s.get_column_count() : s.get_row_count();
    }

    array<Float> x_;
};

} // namespace oneapi::dal::backend::linalg
