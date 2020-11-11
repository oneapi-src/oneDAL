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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/detail/table_builder_impl.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T>
struct is_table_builder_impl {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(table, build, ())

    static constexpr bool value = has_method_build_v<T>;
};

template <typename T>
inline constexpr bool is_table_builder_impl_v = is_table_builder_impl<T>::value;

template <typename T>
struct is_homogen_table_builder_impl {
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(homogen_table, build, ())
    ONEDAL_HAS_METHOD_TRAIT(void, reset, (homogen_table && t), reset_from_table)
    ONEDAL_HAS_METHOD_TRAIT(void,
                            reset,
                            (const array<byte_t>& data,
                             std::int64_t row_count,
                             std::int64_t column_count),
                            reset_from_array)
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void, set_data_type, (data_type dt))
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void, set_feature_type, (feature_type ft))
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void,
                                   allocate,
                                   (std::int64_t row_count, std::int64_t column_count))
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void, set_layout, (data_layout layout))
    ONEDAL_SIMPLE_HAS_METHOD_TRAIT(void,
                                   copy_data,
                                   (const void* data,
                                    std::int64_t row_count,
                                    std::int64_t column_count))

    static constexpr bool value_host =
        has_method_build_v<T> && has_method_reset_from_table_v<T> &&
        has_method_reset_from_array_v<T> && has_method_set_data_type_v<T> &&
        has_method_set_feature_type_v<T> && has_method_allocate_v<T> &&
        has_method_set_layout_v<T> && has_method_copy_data_v<T>;

#ifdef ONEDAL_DATA_PARALLEL
    ONEDAL_HAS_METHOD_TRAIT(void,
                            allocate,
                            (const sycl::queue& queue,
                             std::int64_t row_count,
                             std::int64_t column_count,
                             sycl::usm::alloc kind),
                            allocate_dpc)
    ONEDAL_HAS_METHOD_TRAIT(
        void,
        copy_data,
        (sycl::queue & queue, const void* data, std::int64_t row_count, std::int64_t column_count),
        copy_data_dpc)

    static constexpr bool value_dpc = has_method_allocate_dpc_v<T> && has_method_copy_data_dpc_v<T>;
    static constexpr bool value = value_host && value_dpc;
#else
    static constexpr bool value = value_host;
#endif
};

template <typename T>
inline constexpr bool is_homogen_table_builder_impl_v = is_homogen_table_builder_impl<T>::value;

class ONEDAL_EXPORT table_builder {
    friend detail::pimpl_accessor;
    using pimpl_t = detail::pimpl<detail::table_builder_impl_iface>;

public:
    template <typename Impl,
              typename ImplType = std::decay_t<Impl>,
              typename = std::enable_if_t<is_table_builder_impl_v<ImplType> &&
                                          !std::is_base_of_v<table_builder, ImplType>>>
    table_builder(Impl&& impl)
            : table_builder(new detail::table_builder_impl_wrapper(std::forward<Impl>(impl))) {}

    table build() const {
        return impl_->build();
    }

protected:
    table_builder(detail::table_builder_impl_iface* obj) : impl_(obj) {}

private:
    pimpl_t impl_;
};

class ONEDAL_EXPORT homogen_table_builder : public table_builder {
public:
    homogen_table_builder();

    template <typename Impl,
              typename ImplType = std::decay_t<Impl>,
              typename = std::enable_if_t<is_homogen_table_builder_impl_v<ImplType> &&
                                          !std::is_base_of_v<table_builder, ImplType>>>
    homogen_table_builder(Impl&& impl)
            : table_builder(
                  new detail::homogen_table_builder_impl_wrapper(std::forward<Impl>(impl))) {}

    homogen_table build() {
        auto& impl = get_impl();
        return impl.build_homogen();
    }

    auto& reset(homogen_table&& t) {
        auto& impl = get_impl();
        impl.reset(std::move(t));
        return *this;
    }
    template <typename Data>
    auto& reset(const array<Data>& data, std::int64_t row_count, std::int64_t column_count) {
        array<byte_t> byte_data;
        if (data.has_mutable_data()) {
            byte_data.reset(data,
                            reinterpret_cast<byte_t*>(data.get_mutable_data()),
                            data.get_size());
        }
        else {
            byte_data.reset(data,
                            reinterpret_cast<const byte_t*>(data.get_data()),
                            data.get_size());
        }

        auto& impl = get_impl();
        impl.set_data_type(detail::make_data_type<Data>());
        impl.reset(byte_data, row_count, column_count);
        return *this;
    }
    auto& set_data_type(data_type dt) {
        auto& impl = get_impl();
        impl.set_data_type(dt);
        return *this;
    }
    auto& set_feature_type(feature_type ft) {
        auto& impl = get_impl();
        impl.set_feature_type(ft);
        return *this;
    }
    auto& allocate(std::int64_t row_count, std::int64_t column_count) {
        auto& impl = get_impl();
        impl.allocate(row_count, column_count);
        return *this;
    }
    auto& set_layout(data_layout layout) {
        auto& impl = get_impl();
        impl.set_layout(layout);
        return *this;
    }
    auto& copy_data(const void* data, std::int64_t row_count, std::int64_t column_count) {
        auto& impl = get_impl();
        impl.copy_data(data, row_count, column_count);
        return *this;
    }

#ifdef ONEDAL_DATA_PARALLEL
    auto& allocate(const sycl::queue& queue,
                   std::int64_t row_count,
                   std::int64_t column_count,
                   const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        auto& impl = get_impl();
        impl.allocate(queue, row_count, column_count, alloc);
        return *this;
    }

    auto& copy_data(sycl::queue& queue,
                    const void* data,
                    std::int64_t row_count,
                    std::int64_t column_count,
                    const sycl::vector_class<sycl::event>& dependencies = {}) {
        auto& impl = get_impl();
        detail::wait_and_throw(dependencies);
        impl.copy_data(queue, data, row_count, column_count);
        return *this;
    }
#endif

private:
    detail::homogen_table_builder_iface& get_impl() {
        return detail::cast_impl<detail::homogen_table_builder_iface>(*this);
    }
};

} // namespace v1

using v1::table_builder;
using v1::homogen_table_builder;

} // namespace oneapi::dal::detail
