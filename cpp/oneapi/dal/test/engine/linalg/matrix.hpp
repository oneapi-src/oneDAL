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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::test::engine::linalg {

enum class layout {
    row_major,
    column_major,
};

inline constexpr layout transpose_layout(layout l) {
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
        ONEDAL_ASSERT_MUL_OVERFLOW(std::int64_t, row_count, column_count);
        shape_[0] = row_count;
        shape_[1] = column_count;
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

template <layout lyt>
class matrix_base {
public:
    constexpr layout get_layout() const {
        return lyt;
    }

    std::int64_t get_row_count() const {
        return shape_.get_row_count();
    }

    std::int64_t get_column_count() const {
        return shape_.get_column_count();
    }

    std::int64_t get_count() const {
        return shape_.get_count();
    }

    const shape& get_shape() const {
        return shape_;
    }

    std::int64_t get_stride() const {
        return stride_;
    }

    std::int64_t get_linear_index(std::int64_t i, std::int64_t j) const {
        ONEDAL_ASSERT(i >= 0 && i < get_row_count(), "Row index is out of range");
        ONEDAL_ASSERT(j >= 0 && j < get_column_count(), "Column index is out of range");
        if constexpr (lyt == layout::row_major) {
            return i * get_stride() + j;
        }
        else {
            return j * get_stride() + i;
        }
    }

protected:
    explicit matrix_base(const shape& s, std::int64_t stride) : shape_(s), stride_(stride) {
        if constexpr (lyt == layout::row_major) {
            ONEDAL_ASSERT(stride >= s.get_column_count(),
                          "Stride must be greater than "
                          "column count in row-major layout");
        }
        else if constexpr (lyt == layout::column_major) {
            ONEDAL_ASSERT(stride >= s.get_row_count(),
                          "Stride must be greater than "
                          "row count in column-major layout");
        }
    }

    explicit matrix_base(const shape& s) : matrix_base(s, get_default_stride(s)) {}

protected:
    static std::int64_t get_default_stride(const shape& s) {
        if constexpr (lyt == layout::row_major) {
            return s.get_column_count();
        }
        else {
            return s.get_row_count();
        }
        return 0;
    }

private:
    shape shape_;
    std::int64_t stride_;
};

template <typename Float, layout lyt = layout::row_major>
class matrix : public matrix_base<lyt> {
public:
    template <typename Float_, layout lyt_>
    friend class matrix;

    using base = matrix_base<lyt>;
    using base::get_row_count;
    using base::get_column_count;
    using base::get_count;
    using base::get_shape;
    using base::get_stride;
    using base::get_linear_index;

    static matrix wrap(const Float* data, const shape& s) {
        return matrix{ array<Float>::wrap(data, s.get_count()), s };
    }

#ifdef ONEDAL_DATA_PARALLEL
    static matrix wrap(const sycl::queue& q, const Float* data, const shape& s) {
        auto ary = array<Float>::wrap(q, data, s.get_count());
        return matrix{ q, std::move(ary), s };
    }
#endif

    static matrix wrap(const array<Float>& x) {
        return matrix{ x, { 1, x.get_count() } };
    }

    static matrix wrap(const array<Float>& x, const shape& s) {
        return matrix{ x, s };
    }

    static matrix wrap(const table& t) {
        if constexpr (lyt != layout::row_major) {
            // TODO: Figure out how to use column-major layout
            throw unimplemented{ dal::detail::error_messages::unsupported_data_layout() };
        }
        const auto t_flat = row_accessor<const Float>{ t }.pull();
        return wrap(t_flat, { t.get_row_count(), t.get_column_count() });
    }

    static matrix wrap(const matrix<Float>& x) {
        return matrix{ x.get_array(), { x.get_row_count(), x.get_column_count() } };
    }

    template <typename NdArrayLike>
    static matrix wrap_nd(const NdArrayLike& x) {
        static_assert(NdArrayLike::axis_count_v == 1 || NdArrayLike::axis_count_v == 2);

        if constexpr (NdArrayLike::axis_count_v == 1) {
            return wrap(x.flatten(), { 1, x.get_dimension(0) });
        }

        return wrap(x.flatten(), { x.get_dimension(0), x.get_dimension(1) });
    }

    static matrix empty(const shape& s) {
        return wrap(array<Float>::empty(s.get_count()), s);
    }

#ifdef ONEDAL_DATA_PARALLEL
    static matrix empty(const sycl::queue& q,
                        const shape& s,
                        sycl::usm::alloc alloc = sycl::usm::alloc::device) {
        return wrap(array<Float>::empty(q, s.get_count(), alloc), s);
    }
#endif

    template <typename Filler>
    static matrix full(const shape& s, Filler&& filler) {
        return empty(s).fill(std::forward<Filler>(filler));
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Filler>
    static matrix full(sycl::queue& q,
                       const shape& s,
                       Filler&& filler,
                       sycl::usm::alloc alloc = sycl::usm::alloc::device) {
        return empty(q, s, alloc).fill(std::forward<Filler>(filler));
    }
#endif

    static matrix ones(const shape& s) {
        return full(s, Float(1));
    }

#ifdef ONEDAL_DATA_PARALLEL
    static matrix ones(sycl::queue& q,
                       const shape& s,
                       sycl::usm::alloc alloc = sycl::usm::alloc::device) {
        return full(q, s, Float(1), alloc);
    }
#endif

    static matrix zeros(const shape& s) {
        return full(s, Float(0));
    }

#ifdef ONEDAL_DATA_PARALLEL
    static matrix zeros(sycl::queue& q,
                        const shape& s,
                        sycl::usm::alloc alloc = sycl::usm::alloc::device) {
        return full(q, s, Float(0), alloc);
    }
#endif

    static matrix diag(std::int64_t dim, Float value) {
        auto m = zeros({ dim, dim });
        Float* data = m.get_mutable_data();
        for (std::int64_t i = 0; i < dim; i++) {
            data[i * dim + i] = value;
        }
        return m;
    }

    static matrix eye(std::int64_t dim) {
        return diag(dim, Float(1));
    }

    matrix() : base({ 0, 0 }, 0) {}

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

    bool has_data() const {
        return get_count() > 0;
    }

    bool has_mutable_data() const {
        return has_data() && x_.has_mutable_data();
    }

#ifdef ONEDAL_DATA_PARALLEL
    std::optional<sycl::queue> get_queue() const {
        return x_.get_queue();
    }

    bool matches_usm_alloc(sycl::usm::alloc alloc) const {
        if (!get_queue().has_value()) {
            return false;
        }
        const auto pointer_type =
            sycl::get_pointer_type(x_.get_data(), get_queue().value().get_context());
        return pointer_type == alloc;
    }

    bool is_pure_host_alloc() const {
        return !get_queue().has_value();
    }

    bool is_device_usm_alloc() const {
        return matches_usm_alloc(sycl::usm::alloc::device);
    }

    bool is_host_usm_alloc() const {
        return matches_usm_alloc(sycl::usm::alloc::host);
    }

    bool is_shared_usm_alloc() const {
        return matches_usm_alloc(sycl::usm::alloc::shared);
    }

    bool is_unknown_usm_alloc() const {
        return matches_usm_alloc(sycl::usm::alloc::unknown);
    }

    bool is_device_friendly_usm() const {
        return is_device_usm_alloc() || is_shared_usm_alloc();
    }

    bool is_host_accessible() const {
        return is_pure_host_alloc() || is_host_usm_alloc() || is_shared_usm_alloc();
    }

    bool is_migratable_to(const sycl::queue& q) const {
        if (is_pure_host_alloc()) {
            return true;
        }
        ONEDAL_ASSERT(get_queue().has_value());
        return get_queue().value().get_context() == q.get_context();
    }

    bool is_accessible_on(const sycl::queue& q) const {
        if (is_pure_host_alloc()) {
            return false;
        }
        ONEDAL_ASSERT(get_queue().has_value());
        return get_queue().value().get_context() == q.get_context();
    }

    matrix to_host() const {
        if (is_pure_host_alloc()) {
            return *this;
        }

        ONEDAL_ASSERT(get_queue().has_value());
        auto q = get_queue().value();

        const auto host_copy = matrix<Float>::empty(this->get_shape());
        q.memcpy(host_copy.get_mutable_data(), x_.get_data(), x_.get_size()).wait_and_throw();
        return host_copy;
    }

    matrix to_device(sycl::queue& q) const {
        check_if_migratable_to(q);
        if (is_device_usm_alloc()) {
            return *this;
        }

        const auto device_copy =
            matrix<Float>::empty(q, this->get_shape(), sycl::usm::alloc::device);
        q.memcpy(device_copy.get_mutable_data(), x_.get_data(), x_.get_size()).wait_and_throw();
        return device_copy;
    }

    matrix to_shared(sycl::queue& q) const {
        check_if_migratable_to(q);
        if (is_shared_usm_alloc()) {
            return *this;
        }

        const auto shared_copy =
            matrix<Float>::empty(q, this->get_shape(), sycl::usm::alloc::shared);
        q.memcpy(shared_copy.get_mutable_data(), x_.get_data(), x_.get_size()).wait_and_throw();
        return shared_copy;
    }
#else
    bool is_pure_host_alloc() const {
        return true;
    }

    bool is_host_accessible() const {
        return true;
    }

    matrix to_host() const {
        return *this;
    }
#endif

    auto t() const {
        return matrix<Float, transpose_layout(lyt)>{ x_, get_shape().t(), get_stride() };
    }

    Float get(std::int64_t linear_i) const {
        return get_data()[linear_i];
    }

    Float get(std::int64_t i, std::int64_t j) const {
        return get_data()[get_linear_index(i, j)];
    }

    Float& set(std::int64_t linear_i) const {
        return get_mutable_data()[linear_i];
    }

    Float& set(std::int64_t i, std::int64_t j) {
        return get_mutable_data()[get_linear_index(i, j)];
    }

    matrix get_row(std::int64_t row_index) const {
        if constexpr (lyt == layout::row_major) {
            const Float* ptr = get_data() + get_stride() * row_index;
            const auto x_with_offset = array<Float>{ x_, ptr, get_column_count() };
            return wrap(x_with_offset, { 1, get_column_count() });
        }
        else {
            check_if_host_accessible();
            return full({ 1, get_column_count() }, [&](std::int64_t i) {
                return get(row_index, i);
            });
        }
    }

    matrix get_column(std::int64_t column_index) const {
        if constexpr (lyt == layout::column_major) {
            const Float* ptr = get_data() + get_stride() * column_index;
            const auto x_with_offset = array<Float>{ x_, ptr, get_row_count() };
            return wrap(x_with_offset, { get_row_count(), 1 });
        }
        else {
            check_if_host_accessible();
            return full({ get_row_count(), 1 }, [&](std::int64_t i) {
                return get(i, column_index);
            });
        }
    }

    auto& fill(Float filler) {
        __ONEDAL_IF_NO_QUEUE__(get_queue(), {
            Float* data_ptr = get_mutable_data();
            for (std::int64_t i = 0; i < get_count(); i++) {
                data_ptr[i] = filler;
            }
        });

        __ONEDAL_IF_QUEUE__(get_queue(), {
            auto q = get_queue().value();
            q.fill(get_mutable_data(), filler, get_count()) //
                .wait_and_throw();
        });

        return *this;
    }

    template <typename Filler, typename = std::enable_if_t<!std::is_arithmetic_v<Filler>>>
    auto& fill(Filler&& filler) {
        Float* data_ptr = get_mutable_data();

        __ONEDAL_IF_NO_QUEUE__(get_queue(), {
            for (std::int64_t i = 0; i < get_count(); i++) {
                data_ptr[i] = filler(i);
            }
        });

        __ONEDAL_IF_QUEUE__(get_queue(), {
            auto q = get_queue().value();
            const auto r = sycl::range<1>{ std::size_t(this->get_count()) };
            auto event = q.parallel_for(r, [=](sycl::id<1> id) {
                data_ptr[id] = filler(id);
            });
            event.wait_and_throw();
        });

        return *this;
    }

    template <typename T = Float, typename = std::enable_if_t<std::is_same_v<T, bool>>>
    bool all() const {
        const auto this_host = to_host();

        bool result = true;
        for (std::int64_t i = 0; i < get_count(); i++) {
            result = result && this_host.get(i);
        }
        return result;
    }

    template <typename T = Float, typename = std::enable_if_t<std::is_same_v<T, bool>>>
    bool any() const {
        const auto this_host = to_host();

        bool result = false;
        for (std::int64_t i = 0; i < get_count(); i++) {
            result = result || this_host.get(i);
        }
        return result;
    }

    matrix copy() const {
        __ONEDAL_IF_NO_QUEUE__(get_queue(), {
            const auto m = empty(get_shape());
            detail::memcpy(detail::default_host_policy{},
                           m.get_mutable_data(),
                           get_data(),
                           x_.get_size());
            return m;
        });

        __ONEDAL_IF_QUEUE__(get_queue(), {
            const auto q = get_queue().value();
            const auto alloc = sycl::get_pointer_type(get_data(), q.get_context());
            const auto m = empty(q, get_shape(), alloc);
            detail::memcpy(detail::data_parallel_policy{ q },
                           m.get_mutable_data(),
                           get_data(),
                           x_.get_size());
            return m;
        });

        ONEDAL_ASSERT(!"Never happen");
        return matrix{};
    }

    template <typename T>
    matrix<T, lyt> astype() const {
        if constexpr (std::is_same_v<Float, T>) {
            return *this;
        }

        const Float* data_ptr = this->get_data();
        const auto filler = [=](std::int64_t i) {
            return T(data_ptr[i]);
        };

        __ONEDAL_IF_NO_QUEUE__(get_queue(), { //
            return matrix<T, lyt>::full(get_shape(), filler);
        });

        __ONEDAL_IF_QUEUE__(get_queue(), {
            auto q = get_queue().value();
            return matrix<T, lyt>::full(q, get_shape(), filler);
        });

        ONEDAL_ASSERT(!"Never happen");
        return matrix<T, lyt>{};
    }

#ifdef ONEDAL_DATA_PARALLEL
    void check_if_migratable_to(const sycl::queue& q) const {
        if (!is_migratable_to(q)) {
            throw std::invalid_argument{ "Cannot migrate data to the device "
                                         "represented by the given queue" };
        }
    }

    void check_if_accessible_on(const sycl::queue& q) const {
        if (!is_accessible_on(q)) {
            throw std::invalid_argument{ "Cannot access data on the device "
                                         "represented by the given queue" };
        }
    }

    void check_if_host_accessible() const {
        if (!is_host_accessible()) {
            throw std::invalid_argument{ "Cannot access data on the host" };
        }
    }
#else
    void check_if_host_accessible() const {}
#endif

private:
    explicit matrix(const array<Float>& x, const shape& s, std::int64_t stride)
            : base(s, stride),
              x_(x) {
        ONEDAL_ASSERT(s.get_count() <= x.get_count(),
                      "Element count in matrix does not match "
                      "element count in the provided array");
    }

    explicit matrix(const array<Float>& x, const shape& s)
            : matrix(x, s, base::get_default_stride(s)) {}

    array<Float> x_;
};

template <typename T, typename Float, layout lyt>
inline matrix<T, lyt> astype(const matrix<Float, lyt>& m) {
    return m.template astype<T>();
}

template <typename Float, layout lyt>
inline matrix<Float, lyt> transpose(const matrix<Float, lyt>& m) {
    // TODO: Implement device version
    m.check_if_host_accessible();

    auto m_trans = matrix<Float, lyt>::empty(m.get_shape().t());
    for (std::int64_t i = 0; i < m.get_row_count(); i++) {
        for (std::int64_t j = 0; j < m.get_column_count(); j++) {
            m_trans.set(j, i) = m.get(i, j);
        }
    }

    return m_trans;
}

} // namespace oneapi::dal::test::engine::linalg
