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
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/primitives/ndshape.hpp"

namespace oneapi::dal::backend::primitives {

enum class ndorder {
    c, /* C-style ordering, row-major in 2D case */
    f /* Fortran-style ordering, column-major in 2D case */
};

template <ndorder order>
struct transposed_ndorder;

template <>
struct transposed_ndorder<ndorder::c> {
    static constexpr ndorder value = ndorder::f;
};

template <>
struct transposed_ndorder<ndorder::f> {
    static constexpr ndorder value = ndorder::c;
};

template <ndorder order>
constexpr ndorder transposed_ndorder_v = transposed_ndorder<order>::value;

template <std::int64_t axis_count, ndorder order = ndorder::c>
class ndarray_base : public base {
    static_assert(axis_count > 0, "Axis count must be non-zero");
    static_assert(order == ndorder::c || order == ndorder::f, "Only C or F orders are supported");

public:
    static constexpr std::int64_t axis_count_v = axis_count;

    ndarray_base() = default;

    explicit ndarray_base(const ndshape<axis_count>& shape, const ndshape<axis_count>& strides)
            : shape_(shape),
              strides_(strides) {
        if constexpr (order == ndorder::c) {
            ONEDAL_ASSERT(strides[axis_count - 1] == 1, "Last C-order stride must be 1");
        }
        else if constexpr (order == ndorder::f) {
            ONEDAL_ASSERT(strides[0] == 1, "First F-order stride must be 1");
        }

#ifdef ONEDAL_ENABLE_ASSERT
        const auto default_strides = get_default_strides(shape);
        for (std::int64_t i = 0; i < axis_count; i++) {
            ONEDAL_ASSERT(strides[i] >= default_strides[i],
                          "Custom stride must be greater than default");
        }
#endif
    }

    explicit ndarray_base(const ndshape<axis_count>& shape)
            : ndarray_base(shape, get_default_strides(shape)) {}

    constexpr ndorder get_order() const {
        return order;
    }

    const ndshape<axis_count>& get_shape() const {
        return shape_;
    }

    const ndshape<axis_count>& get_strides() const {
        return strides_;
    }

    std::int64_t get_dimension(std::int64_t axis) const {
        return shape_[axis];
    }

    std::int64_t get_stride(std::int64_t axis) const {
        return strides_[axis];
    }

    std::int64_t get_leading_stride() const {
        if constexpr (order == ndorder::c) {
            return strides_[0];
        }
        else if constexpr (order == ndorder::f) {
            return strides_[axis_count - 1];
        }
        return 0;
    }

    std::int64_t get_count() const {
        return get_shape().get_count();
    }

    template <typename... Indices>
    std::int64_t get_flat_index(Indices&&... indices) const {
        static_assert(std::tuple_size_v<std::tuple<Indices...>> == axis_count,
                      "Incorrect number of indices");

        std::int64_t flat_index = 0;
        ndindex<axis_count> idx{ std::int64_t(indices)... };

        for (std::int64_t i = 0; i < axis_count; i++) {
            ONEDAL_ASSERT(idx[i] < shape_[i], "Index is out of range");
            flat_index += strides_[i] * idx[i];
        }

        return flat_index;
    }

protected:
    static ndshape<axis_count> get_default_strides(const ndshape<axis_count>& shape) {
        // There is no need for multiplication overflow checks because
        // `shape` guaranties that shape[0] * ... * shape[axis_count - 1]
        // has no overflows

        ndindex<axis_count> strides;

        std::int64_t stride = 1;
        if constexpr (order == ndorder::c) {
            for (std::int64_t i = axis_count - 1; i >= 0; i--) {
                strides[i] = stride;
                stride *= shape[i];
            }
        }
        else if constexpr (order == ndorder::f) {
            for (std::int64_t i = 0; i < axis_count; i++) {
                strides[i] = stride;
                stride *= shape[i];
            }
        }

        return strides;
    }

    void check_if_strides_are_default() const {
#ifdef ONEDAL_ENABLE_ASSERT
        const auto default_strides = get_default_strides(this->get_shape());
        for (std::int64_t i = 0; i < axis_count; i++) {
            ONEDAL_ASSERT(this->get_strides()[i] == default_strides[i],
                          "Operation can be applied only to the array with default strides");
        }
#endif
    }

private:
    ndshape<axis_count> shape_;
    ndshape<axis_count> strides_;
};

template <typename T, std::int64_t axis_count, ndorder order = ndorder::c>
class ndarray;

template <typename T, std::int64_t axis_count, ndorder order = ndorder::c>
class ndview : public ndarray_base<axis_count, order> {
    static_assert(!std::is_const_v<T>, "T must be non-const");

    template <typename, std::int64_t, ndorder>
    friend class ndview;

    template <typename U>
    using enable_if_cv_match_t = std::enable_if_t<std::is_same_v<T, std::remove_cv_t<U>>>;

public:
    using base = ndarray_base<axis_count, order>;
    using shape_t = ndshape<axis_count>;

    ndview() : data_(nullptr) {}

    static ndview wrap(T* data, const shape_t& shape) {
        return ndview{ data, shape };
    }

    static ndview wrap(const T* data, const shape_t& shape) {
        return ndview{ data, shape };
    }

    static ndview wrap(T* data, const shape_t& shape, const shape_t& strides) {
        return ndview{ data, shape, strides };
    }

    static ndview wrap(const T* data, const shape_t& shape, const shape_t& strides) {
        return ndview{ data, shape, strides };
    }

    static ndview wrap_mutable(const array<T>& data, const shape_t& shape) {
        ONEDAL_ASSERT(data.has_mutable_data());
        ONEDAL_ASSERT(data.get_count() >= shape.get_count());
        return wrap(data.get_mutable_data(), shape);
    }

    static ndview wrap_mutable(const array<T>& data, const shape_t& shape, const shape_t& strides) {
        ONEDAL_ASSERT(data.has_mutable_data());
        ONEDAL_ASSERT(data.get_count() >= shape.get_count());
        return wrap(data.get_mutable_data(), shape, strides);
    }

    template <std::int64_t d = axis_count, typename = std::enable_if_t<d == 1>>
    static ndview wrap(const array<T>& data) {
        return wrap(data.get_data(), { data.get_count() });
    }

    template <std::int64_t d = axis_count, typename = std::enable_if_t<d == 1>>
    static ndview wrap_mutable(const array<T>& data) {
        ONEDAL_ASSERT(data.has_mutable_data());
        return wrap(data.get_mutable_data(), { data.get_count() });
    }

    const T* get_data() const {
        return data_;
    }

    T* get_mutable_data() const {
        ONEDAL_ASSERT(data_is_mutable_);
        return const_cast<T*>(data_);
    }

    bool has_data() const {
        return this->get_count() > 0;
    }

    bool has_mutable_data() const {
        return this->has_data() && this->data_is_mutable_;
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    const T& at(std::int64_t id) const {
        ONEDAL_ASSERT(has_data());
        ONEDAL_ASSERT((this->get_dimension(0) >= id) && (id >= 0));
        return *(get_data() + id);
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    const T& at(std::int64_t id0, std::int64_t id1) const {
        ONEDAL_ASSERT(has_data());
        ONEDAL_ASSERT((this->get_dimension(0) >= id0) && (id0 >= 0));
        ONEDAL_ASSERT((this->get_dimension(1) >= id1) && (id1 >= 0));
        if constexpr (order == ndorder::c) {
            const auto* row = get_data() + id0 * this->get_stride(0);
            return *(row + id1);
        }
        else {
            const auto* const col = get_data() + id1 * this->get_stride(1);
            return *(col + id0);
        }
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    T& at(std::int64_t id) {
        ONEDAL_ASSERT(has_mutable_data());
        ONEDAL_ASSERT((this->get_dimension(0) >= id) && (id >= 0));
        return *(get_mutable_data() + id);
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    T& at(std::int64_t id0, std::int64_t id1) {
        ONEDAL_ASSERT(has_mutable_data());
        ONEDAL_ASSERT((this->get_dimension(0) >= id0) && (id0 >= 0));
        ONEDAL_ASSERT((this->get_dimension(1) >= id1) && (id1 >= 0));
        if constexpr (order == ndorder::c) {
            auto* const row = get_mutable_data() + id0 * this->get_stride(0);
            return *(row + id1);
        }
        else {
            auto* const col = get_mutable_data() + id1 * this->get_stride(1);
            return *(col + id0);
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    T at_device(sycl::queue& queue, std::int64_t id, const event_vector& deps = {}) const {
        return this->get_slice(id, id + 1).to_host(queue, deps).at(0);
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    T at_device(sycl::queue& queue,
                std::int64_t id0,
                std::int64_t id1,
                const event_vector& deps = {}) const {
        return this->get_row_slice(id0, id0 + 1)
            .get_col_slice(id1, id1 + 1)
            .to_host(queue, deps)
            .at(0, 0);
    }
#endif // ONEDAL_DATA_PARALLEL

    auto t() const {
        using tranposed_ndview_t = ndview<T, axis_count, transposed_ndorder_v<order>>;
        const auto& shape = this->get_shape();
        const auto& strides = this->get_strides();
        return tranposed_ndview_t{ data_, shape.t(), strides.t(), data_is_mutable_ };
    }

    template <std::int64_t new_axis_count, ndorder new_order = order>
    auto reshape(const ndshape<new_axis_count>& new_shape) const {
        check_reshape_conditions(new_shape);
        using reshaped_ndview_t = ndview<T, new_axis_count, new_order>;
        return reshaped_ndview_t{ data_, new_shape, data_is_mutable_ };
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    ndview get_slice(std::int64_t from, std::int64_t to) const {
        ONEDAL_ASSERT((this->get_dimension(0) >= from) && (from >= 0));
        ONEDAL_ASSERT((this->get_dimension(0) >= to) && (to >= from));
        ONEDAL_ASSERT(this->has_data());
        const ndshape<1> new_shape{ to - from };
        const T* new_start_point = this->get_data() + from;
        return ndview(new_start_point, new_shape, this->get_strides(), this->data_is_mutable_);
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    ndview get_row_slice(std::int64_t from_row, std::int64_t to_row) const {
        ONEDAL_ASSERT((this->get_dimension(0) >= from_row) && (from_row >= 0));
        ONEDAL_ASSERT((this->get_dimension(0) >= to_row) && (to_row >= from_row));
        ONEDAL_ASSERT(this->has_data());
        const ndshape<2> new_shape{ (to_row - from_row), this->get_dimension(1) };
        if constexpr (order == ndorder::c) {
            const T* new_start_point = this->get_data() + from_row * this->get_leading_stride();
            return ndview(new_start_point, new_shape, this->get_strides(), this->data_is_mutable_);
        }
        const T* new_start_point = this->get_data() + from_row;
        return ndview(new_start_point, new_shape, this->get_strides(), this->data_is_mutable_);
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 2>>
    ndview get_col_slice(std::int64_t from_col, std::int64_t to_col) const {
        return this->t().get_row_slice(from_col, to_col).t();
    }

#ifdef ONEDAL_DATA_PARALLEL
    ndarray<T, axis_count, order> to_host(sycl::queue& q, const event_vector& deps = {}) const;
    ndarray<T, axis_count, order> to_device(sycl::queue& q, const event_vector& deps = {}) const;
#endif

#ifdef ONEDAL_DATA_PARALLEL
    sycl::event prefetch(sycl::queue& queue) const {
        return queue.prefetch(data_, this->get_count());
    }
#endif

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    T* begin() {
        ONEDAL_ASSERT(data_is_mutable_);
        return get_mutable_data();
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    T* end() {
        ONEDAL_ASSERT(data_is_mutable_);
        return get_mutable_data() + this->get_count();
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    const T* cbegin() const {
        return get_data();
    }

    template <std::int64_t n = axis_count, typename = std::enable_if_t<n == 1>>
    const T* cend() const {
        return get_data() + this->get_count();
    }

protected:
    explicit ndview(const T* data,
                    const shape_t& shape,
                    const shape_t& strides,
                    bool data_is_mutable)
            : base(shape, strides),
              data_(data),
              data_is_mutable_(data_is_mutable) {}

    template <typename U, typename = enable_if_cv_match_t<U>>
    explicit ndview(U* data, const shape_t& shape, const shape_t& strides)
            : base(shape, strides),
              data_(data),
              data_is_mutable_(std::is_same_v<U, T>) {}

    explicit ndview(const T* data, const shape_t& shape, bool data_is_mutable)
            : base(shape),
              data_(data),
              data_is_mutable_(data_is_mutable) {}

    template <typename U, typename = enable_if_cv_match_t<U>>
    explicit ndview(U* data, const shape_t& shape)
            : base(shape),
              data_(data),
              data_is_mutable_(std::is_same_v<U, T>) {}

    template <std::int64_t new_axis_count>
    void check_reshape_conditions(const ndshape<new_axis_count>& new_shape) const {
        ONEDAL_ASSERT(new_shape.get_count() == this->get_count(),
                      "Total element count must remain unchanged");
    }

    ndview& set_mutability(bool data_is_mutable) {
        data_is_mutable_ = data_is_mutable;
        return *this;
    }

private:
    const T* data_;
    bool data_is_mutable_;
};

template <typename T1, ndorder ord1, typename T2, ndorder ord2>
inline void copy(ndview<T1, 2, ord1>& dst, const ndview<T2, 2, ord2>& src) {
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    const ndshape<2> dst_shape = dst.get_shape();
    ONEDAL_ASSERT(dst_shape == src.get_shape());
    if constexpr (ord1 == ndorder::c) {
        T1* const dst_ptr = dst.get_mutable_data();
        const T2* const src_ptr = src.get_data();
        const auto dst_stride = dst.get_leading_stride();
        const auto src_stride = src.get_leading_stride();

        for (std::int64_t r = 0; r < dst_shape[0]; ++r) {
            for (std::int64_t c = 0; c < dst_shape[1]; ++c) {
                T1& dst_ref = *(dst_ptr + r * dst_stride + c);
                if constexpr (ord2 == ndorder::c) {
                    dst_ref = static_cast<T1>(*(src_ptr + r * src_stride + c));
                }
                else {
                    dst_ref = static_cast<T2>(*(src_ptr + c * src_stride + r));
                }
            }
        }
    }
    else {
        auto new_dst = dst.t();
        const auto new_src = src.t();
        copy(new_dst, new_src);
    }
}

#ifdef ONEDAL_DATA_PARALLEL
template <typename T1, ndorder ord1, typename T2, ndorder ord2>
inline sycl::event copy(sycl::queue& q,
                        ndview<T1, 2, ord1>& dst,
                        const ndview<T2, 2, ord2>& src,
                        const event_vector& deps = {}) {
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    const ndshape<2> dst_shape = dst.get_shape();
    ONEDAL_ASSERT(dst_shape == src.get_shape());
    sycl::event res_event;
    if constexpr (ord1 == ndorder::c) {
        T1* const dst_ptr = dst.get_mutable_data();
        const T2* const src_ptr = src.get_data();
        const auto dst_stride = dst.get_leading_stride();
        const auto src_stride = src.get_leading_stride();
        const auto cp_range = make_range_2d(dst_shape[0], dst_shape[1]);
        res_event = q.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(cp_range, [=](sycl::id<2> idx) {
                T1& dst_ref = *(dst_ptr + idx[0] * dst_stride + idx[1]);
                if constexpr (ord2 == ndorder::c) {
                    dst_ref = static_cast<T1>(*(src_ptr + idx[0] * src_stride + idx[1]));
                }
                else {
                    dst_ref = static_cast<T2>(*(src_ptr + idx[1] * src_stride + idx[0]));
                }
            });
        });
    }
    else {
        auto new_dst = dst.t();
        const auto new_src = src.t();
        res_event = copy(q, new_dst, new_src, deps);
    }
    return res_event;
}

template <typename T1, ndorder ord1, typename T2, ndorder ord2>
inline sycl::event copy(sycl::queue& q,
                        ndview<T1, 1, ord1>& dst,
                        const ndview<T2, 1, ord2>& src,
                        const event_vector& deps = {}) {
    auto dst_2d = dst.template reshape<2>({ 1l, dst.get_count() });
    auto src_2d = src.template reshape<2>({ 1l, src.get_count() });

    return copy(q, dst_2d, src_2d, deps);
}

template <typename T>
inline sycl::event fill(sycl::queue& q,
                        ndview<T, 1>& dst,
                        const T& value = T{},
                        const event_vector& deps = {}) {
    ONEDAL_ASSERT(dst.has_mutable_data());
    return q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.fill(dst.get_mutable_data(), value, dst.get_count());
    });
}

template <typename T, ndorder ord1>
inline sycl::event fill(sycl::queue& q,
                        ndview<T, 2, ord1>& dst,
                        const T& value = T{},
                        const event_vector& deps = {}) {
    ONEDAL_ASSERT(dst.has_mutable_data());
    sycl::event res_event;
    if constexpr (ord1 == ndorder::c) {
        T* const dst_ptr = dst.get_mutable_data();
        const ndshape<2> dst_shape = dst.get_shape();
        const auto dst_stride = dst.get_leading_stride();
        const auto fl_range = make_range_2d(dst_shape[0], dst_shape[1]);
        res_event = q.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(fl_range, [=](sycl::id<2> idx) {
                *(dst_ptr + idx[0] * dst_stride + idx[1]) = value;
            });
        });
    }
    else {
        auto new_dst = dst.t();
        res_event = fill(q, new_dst, value, deps);
    }
    return res_event;
}
#endif

template <typename T, std::int64_t axis_count, ndorder order>
class ndarray : public ndview<T, axis_count, order> {
    template <typename, std::int64_t, ndorder>
    friend class ndarray;

    using base = ndview<T, axis_count, order>;
    using shape_t = ndshape<axis_count>;
    using shared_t = dal::detail::shared<T>;
    using array_t = dal::array<std::remove_const_t<T>>;

    struct array_deleter {
        explicit array_deleter(const array_t& ary) : ary_(ary) {}

        explicit array_deleter(array_t&& ary) : ary_(std::move(ary)) {}

        void operator()(T* ptr) const {}

        array_t ary_;
    };

public:
    ndarray() = default;

    template <typename Deleter = dal::detail::empty_delete<T>>
    static ndarray wrap(T* data, const shape_t& shape, Deleter&& deleter = Deleter{}) {
        auto shared = shared_t{ data, std::forward<Deleter>(deleter) };
        return wrap(std::move(shared), shape);
    }

    template <typename Deleter = dal::detail::empty_delete<T>>
    static ndarray wrap(const T* data, const shape_t& shape, Deleter&& deleter = Deleter{}) {
        auto shared = shared_t{ const_cast<T*>(data), std::forward<Deleter>(deleter) };
        return wrap(std::move(shared), shape).set_mutability(false);
    }

    static ndarray wrap(const shared_t& data, const shape_t& shape) {
        return ndarray{ data, shape };
    }

    static ndarray wrap(shared_t&& data, const shape_t& shape) {
        return ndarray{ std::move(data), shape };
    }

    static ndarray wrap(const array_t& ary, const shape_t& shape) {
        ONEDAL_ASSERT(ary.get_count() == shape.get_count());
        return wrap(ary.get_data(), shape, array_deleter{ ary });
    }

    static ndarray wrap(array_t&& ary, const shape_t& shape) {
        ONEDAL_ASSERT(ary.get_count() == shape.get_count());
        const T* data_ptr = ary.get_data();
        return wrap(data_ptr, shape, array_deleter{ std::move(ary) });
    }

    static ndarray wrap(const array_t& ary) {
        static_assert(axis_count == 1);
        return wrap(ary, shape_t{ ary.get_count() });
    }

    static ndarray wrap(array_t&& ary) {
        static_assert(axis_count == 1);
        std::int64_t ary_count = ary.get_count();
        return wrap(std::move(ary), shape_t{ ary_count });
    }

    static ndarray wrap_mutable(const array_t& ary, const shape_t& shape) {
        ONEDAL_ASSERT(ary.get_count() == shape.get_count());
        auto shared = shared_t{ ary.get_mutable_data(), array_deleter{ ary } };
        return wrap(std::move(shared), shape);
    }

    static ndarray wrap_mutable(array_t&& ary, const shape_t& shape) {
        ONEDAL_ASSERT(ary.get_count() == shape.get_count());
        T* data_ptr = ary.get_mutable_data();
        auto shared = shared_t{ data_ptr, array_deleter{ std::move(ary) } };
        return wrap(std::move(shared), shape);
    }

    static ndarray wrap_mutable(const array_t& ary) {
        static_assert(axis_count == 1);
        return wrap_mutable(ary, shape_t{ ary.get_count() });
    }

    static ndarray wrap_mutable(array_t&& ary) {
        static_assert(axis_count == 1);
        std::int64_t ary_count = ary.get_count();
        return wrap_mutable(std::move(ary), shape_t{ ary_count });
    }

    static ndarray empty(const shape_t& shape) {
        T* ptr = dal::detail::malloc<T>(dal::detail::default_host_policy{}, shape.get_count());
        return wrap(ptr,
                    shape,
                    dal::detail::make_default_delete<T>(dal::detail::default_host_policy{}));
    }

    static ndarray copy(const T* data, const shape_t& shape) {
        auto ary = empty(shape);
        ary.assign(data, shape.get_count());
        return ary;
    }

    static ndarray zeros(const shape_t& shape) {
        auto ary = empty(shape);
        ary.fill(T(0));
        return ary;
    }

#ifdef ONEDAL_DATA_PARALLEL
    static ndarray empty(const sycl::queue& q,
                         const shape_t& shape,
                         const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        T* ptr = malloc<T>(q, shape.get_count(), alloc_kind);
        return wrap(ptr, shape, usm_deleter<T>{ q });
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    static std::tuple<ndarray, sycl::event> copy(
        sycl::queue& q,
        const T* data,
        const shape_t& shape,
        const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        auto ary = empty(q, shape, alloc_kind);
        auto event = ary.assign(q, data, shape.get_count());
        return { ary, event };
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    static std::tuple<ndarray, sycl::event> full(
        sycl::queue& q,
        const shape_t& shape,
        const T& value,
        const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared,
        const event_vector& deps = {}) {
        auto ary = empty(q, shape, alloc_kind);
        auto event = ary.fill(q, value, deps);
        return { ary, event };
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    static std::tuple<ndarray, sycl::event> zeros(
        sycl::queue& q,
        const shape_t& shape,
        const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        return full(q, shape, T(0), alloc_kind);
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    static std::tuple<ndarray, sycl::event> ones(
        sycl::queue& q,
        const shape_t& shape,
        const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        return full(q, shape, T(1), alloc_kind);
    }
#endif

    const base& get_view() const {
        return *this;
    }

    array_t flatten() const {
        return array_t{ data_, this->get_count() };
    }

#ifdef ONEDAL_DATA_PARALLEL
    array_t flatten(sycl::queue& q, const event_vector& deps = {}) const {
        ONEDAL_ASSERT(is_known_usm(q, data_.get()));
        return array_t{ q, data_, this->get_count(), deps };
    }
#endif

    auto t() const {
        using tranposed_ndarray_t = ndarray<T, axis_count, transposed_ndorder_v<order>>;
        const auto& shape = this->get_shape();
        const auto& strides = this->get_strides();
        return tranposed_ndarray_t{ data_, shape.t(), strides.t() }.set_mutability(
            this->has_mutable_data());
    }

    template <std::int64_t new_axis_count>
    auto reshape(const ndshape<new_axis_count>& new_shape) const {
        using reshaped_ndarray_t = ndarray<T, new_axis_count, order>;
        base::check_reshape_conditions(new_shape);
        return reshaped_ndarray_t{ data_, new_shape }.set_mutability(this->has_mutable_data());
    }

    void fill(T value) {
        T* data_ptr = this->get_mutable_data();
        for (std::int64_t i = 0; i < this->get_count(); i++) {
            data_ptr[i] = value;
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    sycl::event fill(sycl::queue& q, T value, const event_vector& deps = {}) {
        auto data_ptr = this->get_mutable_data();
        ONEDAL_ASSERT(is_known_usm(q, data_ptr));
        return q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.fill(data_ptr, value, this->get_count());
        });
    }
#endif

    void arange() {
        T* data_ptr = this->get_mutable_data();
        for (std::int64_t i = 0; i < this->get_count(); i++) {
            data_ptr[i] = i;
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    sycl::event arange(sycl::queue& q, const event_vector& deps = {}) {
        auto data_ptr = this->get_mutable_data();
        ONEDAL_ASSERT(is_known_usm(q, data_ptr));
        return q.submit([&](sycl::handler& cgh) {
            const auto range = dal::backend::make_range_1d(this->get_count());
            cgh.depends_on(deps);
            cgh.parallel_for(range, [=](sycl::id<1> idx) {
                data_ptr[idx] = idx;
            });
        });
    }
#endif

    void assign(const T* source_ptr, std::int64_t source_count) {
        ONEDAL_ASSERT(source_ptr != nullptr);
        ONEDAL_ASSERT(source_count > 0);
        ONEDAL_ASSERT(source_count <= this->get_count());
        return dal::backend::copy(this->get_mutable_data(), source_ptr, source_count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    sycl::event assign(sycl::queue& q,
                       const T* source_ptr,
                       std::int64_t source_count,
                       const event_vector& deps = {}) {
        ONEDAL_ASSERT(source_ptr != nullptr);
        ONEDAL_ASSERT(source_count > 0);
        ONEDAL_ASSERT(source_count <= this->get_count());
        return dal::backend::copy(q, this->get_mutable_data(), source_ptr, source_count, deps);
    }

    sycl::event assign_from_host(sycl::queue& q,
                                 const T* source_ptr,
                                 std::int64_t source_count,
                                 const event_vector& deps = {}) {
        ONEDAL_ASSERT(source_ptr != nullptr);
        ONEDAL_ASSERT(source_count > 0);
        ONEDAL_ASSERT(source_count <= this->get_count());
        return dal::backend::copy_host2usm(q,
                                           this->get_mutable_data(),
                                           source_ptr,
                                           source_count,
                                           deps);
    }

    sycl::event assign(sycl::queue& q, const ndarray& src, const event_vector& deps = {}) {
        ONEDAL_ASSERT(src.get_count() > 0);
        ONEDAL_ASSERT(src.get_count() <= this->get_count());
        return this->assign(q, src.get_data(), src.get_count(), deps);
    }

    sycl::event assign_from_host(sycl::queue& q,
                                 const ndarray& src,
                                 const event_vector& deps = {}) {
        ONEDAL_ASSERT(src.get_count() > 0);
        ONEDAL_ASSERT(src.get_count() <= this->get_count());
        return this->assign_from_host(q, src.get_data(), src.get_count(), deps);
    }
#endif

    ndarray slice(std::int64_t offset, std::int64_t count, std::int64_t axis = 0) const {
        ONEDAL_ASSERT(order == ndorder::c, "Only C-order is supported");
        ONEDAL_ASSERT(axis == 0, "Non-zero axis is not supported");
        ONEDAL_ASSERT(offset >= 0);
        ONEDAL_ASSERT(count > 0);
        ONEDAL_ASSERT(offset + count <= this->get_dimension(axis));

        const auto shape = this->get_shape();
        const std::int64_t rest_shape_count = shape.get_count() / shape[axis];
        ONEDAL_ASSERT(rest_shape_count > 0);

        T* data_ptr = data_.get() + offset * rest_shape_count;
        const auto aliased_data = shared_t{ data_, data_ptr };

        backend::ndindex<axis_count> shape_index = shape.get_index();
        shape_index[0] = count;

        return wrap(aliased_data, ndshape<axis_count>{ shape_index });
    }

#ifdef ONEDAL_DATA_PARALLEL
    std::vector<ndarray> split(std::int64_t fold_count, std::int64_t axis = 0) const {
        ONEDAL_ASSERT(order == ndorder::c, "Only C-order is supported");
        ONEDAL_ASSERT(axis == 0, "Non-zero axis is not supported");
        ONEDAL_ASSERT(fold_count >= 0);

        if (fold_count <= 0) {
            return {};
        }

        const std::int64_t regular_block = this->get_dimension(axis) / fold_count;
        ONEDAL_ASSERT(regular_block > 0);

        std::vector<ndarray> slices;
        slices.reserve(fold_count);

        for (std::int64_t i = 0; i < fold_count - 1; i++) {
            slices.push_back(this->slice(i * regular_block, regular_block, axis));
        }

        {
            const std::int64_t i = fold_count - 1;
            const std::int64_t tail_block = this->get_dimension(axis) - regular_block * i;
            slices.push_back(this->slice(i * regular_block, tail_block, axis));
        }

        return slices;
    }
#endif

private:
    explicit ndarray(const shared_t& data, const shape_t& shape, const shape_t& strides)
            : base(data.get(), shape, strides),
              data_(data) {}

    explicit ndarray(shared_t&& data, const shape_t& shape, const shape_t& strides)
            : base(data.get(), shape, strides),
              data_(std::move(data)) {}

    explicit ndarray(const shared_t& data, const shape_t& shape)
            : base(data.get(), shape),
              data_(data) {}

    explicit ndarray(shared_t&& data, const shape_t& shape)
            : base(data.get(), shape),
              data_(std::move(data)) {}

    ndarray& set_mutability(bool data_is_mutable) {
        base::set_mutability(data_is_mutable);
        return *this;
    }

    shared_t data_;
};

#ifdef ONEDAL_DATA_PARALLEL
template <typename T, std::int64_t axis_count, ndorder order>
ndarray<T, axis_count, order> ndview<T, axis_count, order>::to_host(
    sycl::queue& q,
    const event_vector& deps) const {
    T* host_ptr = dal::detail::host_allocator<T>().allocate(this->get_count());
    dal::backend::copy_usm2host(q, host_ptr, this->get_data(), this->get_count(), deps)
        .wait_and_throw();
    return ndarray<T, axis_count, order>::wrap(
        host_ptr,
        this->get_shape(),
        dal::detail::make_default_delete<T>(dal::detail::default_host_policy{}));
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <typename T, std::int64_t axis_count, ndorder order>
ndarray<T, axis_count, order> ndview<T, axis_count, order>::to_device(
    sycl::queue& q,
    const event_vector& deps) const {
    auto dev = ndarray<T, axis_count, order>::empty(q, this->get_shape(), sycl::usm::alloc::device);
    dal::backend::copy_host2usm(q,
                                dev.get_mutable_data(),
                                this->get_data(),
                                this->get_count(),
                                deps)
        .wait_and_throw();
    return dev;
}
#endif

#ifdef ONEDAL_DATA_PARALLEL

template <ndorder yorder,
          typename Type,
          ndorder xorder,
          sycl::usm::alloc alloc = sycl::usm::alloc::device>
inline auto copy(sycl::queue& q,
                 const ndview<Type, 2, xorder>& src,
                 const event_vector& deps = {}) {
    ONEDAL_ASSERT(src.has_data());
    const auto shape = src.get_shape();
    auto res_array = ndarray<Type, 2, yorder>::empty(q, shape, alloc);
    auto res_event = copy(q, res_array, src, deps);
    return std::make_pair(res_array, res_event);
}

template <typename Type, ndorder xorder, sycl::usm::alloc alloc = sycl::usm::alloc::device>
inline auto copy(sycl::queue& q,
                 const ndview<Type, 2, xorder>& src,
                 const event_vector& deps = {}) {
    return copy<xorder, Type, xorder, alloc>(q, src, deps);
}

#endif

} // namespace oneapi::dal::backend::primitives

namespace oneapi::dal::backend {

template <typename Type>
inline Type* begin(const primitives::ndview<Type, 1>& arr) {
    ONEDAL_ASSERT(arr.has_mutable_data());
    return arr.get_mutable_data();
}

template <typename Type>
inline Type* end(const primitives::ndview<Type, 1>& arr) {
    return begin(arr) + arr.get_count();
}

template <typename Type>
inline const Type* cbegin(const primitives::ndview<Type, 1>& arr) {
    ONEDAL_ASSERT(arr.has_data());
    return arr.get_data();
}

template <typename Type>
inline const Type* cend(const primitives::ndview<Type, 1>& arr) {
    return cbegin(arr) + arr.get_count();
}

} // namespace oneapi::dal::backend
