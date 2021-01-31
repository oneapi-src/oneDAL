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
    f  /* Fortran-style ordering, column-major in 2D case */
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
class ndarray_base {
    static_assert(axis_count > 0, "Axis count must be non-zero");

public:
    ndarray_base() = default;

    explicit ndarray_base(const ndshape<axis_count>& shape,
                          const ndshape<axis_count>& strides)
            : shape_(shape),
              strides_(strides) {
        if constexpr (order == ndorder::c) {
            ONEDAL_ASSERT(strides[axis_count - 1] == 1, "Last C-order stride must be 1");
        }
        else if constexpr (order == ndorder::f) {
            ONEDAL_ASSERT(strides[0] == 1, "First F-order stride must be 1");
        }
        else {
            ONEDAL_ASSERT(!"Unsupported order");
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


    ndorder get_order() const {
        return order;
    }

    const ndshape<axis_count>& get_shape() const {
        return shape_;
    }

    const ndshape<axis_count>& get_strides() const {
        return strides_;
    }

    std::int64_t get_count() const {
        return get_shape().get_count();
    }

    template <typename... Indices>
    std::int64_t get_flat_index(Indices&&... indices) const {
        static_assert(std::tuple_size_v<std::tuple<Indices...>> == axis_count,
                     "Incorrect number of indices");

        std::int64_t flat_index = 0;
        ndindex<axis_count> idx { std::int64_t(indices)... };

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
        else {
            ONEDAL_ASSERT(!"Unsupported order");
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
class ndview : public ndarray_base<axis_count, order> {

    template <typename, std::int64_t, ndorder>
    friend class ndview;

public:
    using base = ndarray_base<axis_count, order>;
    using shape_t = ndshape<axis_count>;

    ndview() :
        data_(nullptr) {}

    static ndview wrap(T* data, const shape_t& shape) {
        return ndview { data, shape };
    }

    static ndview wrap(T* data, const shape_t& shape, const shape_t& strides) {
        return ndview { data, shape, strides };
    }

    T* get_data() const {
        return data_;
    }

    bool has_data() const {
        return this->get_count() > 0;
    }

    auto t() const {
        using tranposed_ndview_t = ndview<T, axis_count, transposed_ndorder_v<order>>;
        const auto& shape = this->get_shape();
        const auto& strides = this->get_strides();
        return tranposed_ndview_t { data_, shape.t(), strides.t() };
    }

    template <std::int64_t new_axis_count>
    auto reshape(const ndshape<new_axis_count>& new_shape) const {
        using reshaped_ndview_t = ndview<T, new_axis_count, order>;
        check_reshape_conditions(new_shape);
        return reshaped_ndview_t { data_, new_shape };
    }

protected:
    explicit ndview(T* data, const shape_t& shape, const shape_t& strides)
            : base(shape, strides),
              data_(data) {}

    explicit ndview(T* data, const shape_t& shape)
            : base(shape),
              data_(data) {}

    template <std::int64_t new_axis_count>
    void check_reshape_conditions(const ndshape<new_axis_count>& new_shape) const {
        ONEDAL_ASSERT(new_shape.get_count() == this->get_count(),
                      "Total element count must remain unchanged");
        base::check_if_strides_are_default();
    }

private:
    T* data_;
};

template <typename T, std::int64_t axis_count, ndorder order = ndorder::c>
class ndarray : public ndview<T, axis_count, order> {

    template <typename, std::int64_t, ndorder>
    friend class ndarray;

private:
    using base = ndview<T, axis_count, order>;
    using shape_t = ndshape<axis_count>;
    using shared_t = dal::detail::shared<T>;
    using array_t = dal::array<std::remove_const_t<T>>;

    struct array_deleter {
        explicit array_deleter(const array_t& ary)
            : ary_(ary) {}

        explicit array_deleter(array_t&& ary)
            : ary_(std::move(ary)) {}

        void operator()(T *ptr) const { }

        array_t ary_;
    };

    template <typename U>
    using enable_if_const_t = std::enable_if_t<std::is_const_v<U>>;

    template <typename U>
    using enable_if_non_const_t = std::enable_if_t<!std::is_const_v<U>>;

public:
    ndarray() = default;

    template <typename Deleter = dal::detail::empty_delete<T>>
    static ndarray wrap(T* data, const shape_t& shape, Deleter&& deleter = Deleter{}) {
        auto shared = shared_t { data, std::forward<Deleter>(deleter) };
        return wrap(std::move(shared), shape);
    }

    static ndarray wrap(const shared_t& data, const shape_t& shape) {
        return ndarray { data, shape };
    }

    static ndarray wrap(shared_t&& data, const shape_t& shape) {
        return ndarray { std::move(data), shape };
    }

    template <typename U = T, typename = enable_if_const_t<U>>
    static ndarray wrap(const array_t& ary, const shape_t& shape) {
        return wrap(ary.get_data(), shape, array_deleter { ary });
    }

    template <typename U = T, typename = enable_if_const_t<U>>
    static ndarray wrap(array_t&& ary, const shape_t& shape) {
        const T* data_ptr = ary.get_data();
        return wrap(data_ptr, shape, array_deleter { std::move(ary) });
    }

    template <typename U = T, typename = enable_if_const_t<U>>
    static ndarray wrap(const array_t& ary) {
        static_assert(axis_count == 1);
        return wrap(ary, shape_t { ary.get_count() });
    }

    template <typename U = T, typename = enable_if_const_t<U>>
    static ndarray wrap(array_t&& ary) {
        static_assert(axis_count == 1);
        std::int64_t ary_count = ary.get_count();
        return wrap(std::move(ary), shape_t { ary_count });
    }

    template <typename U = T, typename = enable_if_non_const_t<U>>
    static ndarray wrap_mutable(const array_t& ary, const shape_t& shape) {
        auto shared = shared_t { ary.get_mutable_data(), array_deleter { ary } };
        return wrap(std::move(shared), shape);
    }

    template <typename U = T, typename = enable_if_non_const_t<U>>
    static ndarray wrap_mutable(array_t&& ary, const shape_t& shape) {
        T* data_ptr = ary.get_mutable_data();
        auto shared = shared_t { data_ptr, array_deleter { std::move(ary) } };
        return wrap(std::move(shared), shape);
    }

    template <typename U = T, typename = enable_if_non_const_t<U>>
    static ndarray wrap_mutable(const array_t& ary) {
        static_assert(axis_count == 1);
        return wrap_mutable(ary, shape_t { ary.get_count() });
    }

    template <typename U = T, typename = enable_if_non_const_t<U>>
    static ndarray wrap_mutable(array_t&& ary) {
        static_assert(axis_count == 1);
        std::int64_t ary_count = ary.get_count();
        return wrap_mutable(std::move(ary), shape_t { ary_count });
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename U = T, typename = enable_if_non_const_t<U>>
    static ndarray empty(const sycl::queue& q,
                         const shape_t& shape,
                         const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        T* ptr = malloc<T>(q, shape.get_count(), alloc_kind);
        return wrap(ptr, shape, usm_deleter<T> { q });
    }

    template <typename U = T, typename = enable_if_non_const_t<U>>
    static std::tuple<ndarray, sycl::event> full_async(sycl::queue& q,
                                                 const shape_t& shape,
                                                 const T& value,
                                                 const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        auto ary = empty(q, shape, alloc_kind);
        auto event = q.fill(ary.get_data(), value, ary.get_count());
        return { ary, event };
    }

    template <typename U = T, typename = enable_if_non_const_t<U>>
    static ndarray full(sycl::queue& q,
                        const shape_t& shape,
                        const T& value,
                        const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        auto [ ary, event ] = full_async(q, shape, value, alloc_kind);
        event.wait_and_throw();
        return ary;
    }

    template <typename U = T, typename = enable_if_non_const_t<U>>
    static std::tuple<ndarray, sycl::event> zeros_async(sycl::queue& q,
                                                  const shape_t& shape,
                                                  const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        return full_async(q, shape, T(0), alloc_kind);
    }

    template <typename U = T, typename = enable_if_non_const_t<U>>
    static ndarray zeros(sycl::queue& q,
                         const shape_t& shape,
                         const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        return full(q, shape, T(0), alloc_kind);
    }

    template <typename U = T, typename = enable_if_non_const_t<U>>
    static std::tuple<ndarray, sycl::event> ones_async(sycl::queue& q,
                                                       const shape_t& shape,
                                                       const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        return full_async(q, shape, T(1), alloc_kind);
    }

    template <typename U = T, typename = enable_if_non_const_t<U>>
    static ndarray ones(sycl::queue& q,
                        const shape_t& shape,
                        const sycl::usm::alloc& alloc_kind = sycl::usm::alloc::shared) {
        return full(q, shape, T(1), alloc_kind);
    }
#endif

    array_t flatten() const {
        return array_t { data_, this->get_count() };
    }

    auto t() const {
        using tranposed_ndarray_t = ndarray<T, axis_count, transposed_ndorder_v<order>>;
        const auto& shape = this->get_shape();
        const auto& strides = this->get_strides();
        return tranposed_ndarray_t { data_, shape.t(), strides.t() };
    }

    template <std::int64_t new_axis_count>
    auto reshape(const ndshape<new_axis_count>& new_shape) const {
        using reshaped_ndarray_t = ndarray<T, new_axis_count, order>;
        base::check_reshape_conditions(new_shape);
        return reshaped_ndarray_t { data_, new_shape };
    }

private:
    explicit ndarray(const shared_t& data,
                     const shape_t& shape,
                     const shape_t& strides)
            : base(data.get(), shape, strides),
              data_(data) {}

    explicit ndarray(shared_t&& data,
                     const shape_t& shape,
                     const shape_t& strides)
            : base(data.get(), shape, strides),
              data_(std::move(data)) {}

    explicit ndarray(const shared_t& data, const shape_t& shape)
            : base(data.get(), shape),
              data_(data) {}

    explicit ndarray(shared_t&& data, const shape_t& shape)
            : base(data.get(), shape),
              data_(std::move(data)) {}

    shared_t data_;
};

} // namespace oneapi::dal::backend::primitives
