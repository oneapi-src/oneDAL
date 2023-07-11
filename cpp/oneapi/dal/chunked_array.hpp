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
#include "oneapi/dal/detail/chunked_array_impl.hpp"

namespace oneapi::dal {
namespace v2 {

template <typename T>
class ONEDAL_EXPORT chunked_array : public detail::chunked_array_base {
    static_assert(!std::is_const_v<T>);

    using base_t = detail::chunked_array_base;

    template <typename U>
    friend class array;

    template <typename U>
    friend class detail::array_impl;

    friend detail::pimpl_accessor;

    friend detail::serialization_accessor;

public:
    using data_t = T;

    chunked_array(chunked_array&& array)
        : chunked_array_base{ std::forward<base_t>(array) } {}

    chunked_array(const chunked_array& array)
        : chunked_array_base{ array } {}

    chunked_array(std::int64_t chunk_count)
        : chunked_array_base{ chunk_count } {}

    chunked_array(dal::array<T>&& array)
        : chunked_array_base{ std::forward<dal::array<T>>(array) } {}

    template <typename... Arrays>
    chunked_array(Arrays&&... arrays)
        : chunked_array_base{ std::forward<Arrays>(arrays)... } {}

    template <typename Policy, typename Alloc>
    array<T> flatten(const Policy& policy, const Alloc& alloc) const {
        ONEDAL_ASSERT(base_t::validate());

        const auto raw = base_t::flatten_impl(policy, alloc);
        const auto impl = detail::array_impl<T>::reinterpret(raw);
        return array<T>{new detail::array_impl<T>{ std::move(impl) }};
    }

    template <typename Policy>
    array<T> flatten(const Policy& policy) const {
        using alloc_t = detail::policy_allocator_t<Policy>;
        return this->flatten(policy, alloc_t{ policy });
    }

    array<T> flatten() const {
        const detail::default_host_policy policy{};
        return this->flatten(policy);
    }

#ifdef ONEDAL_DATA_PARALLEL
    array<T> flatten(const detail::data_parallel_policy& policy,
                     const std::vector<sycl::event>& deps) const {
        sycl::event::wait_and_throw(deps);
        return this->flatten(policy);
    }

    array<T> flatten(sycl::queue& queue,
                     sycl::usm::alloc alloc = sycl::usm::alloc::shared,
                     const std::vector<sycl::event>& deps = {}) const {
        sycl::event::wait_and_throw(deps);

        using alloc_t = detail::data_parallel_allocator<byte_t>;
        const auto policy = detail::data_parallel_policy(queue);
        return this->flatten(policy, alloc_t{ policy, alloc });
    }
#endif // ONEDAL_DATA_PARALLEL

    template <typename Policy, typename Alloc>
    array<const T*> get_data(const Policy& policy, const Alloc& alloc) const {
        ONEDAL_ASSERT(base_t::validate());
        //ONEDAL_ASSERT(base_t::has_mutable_data());

        const auto raw = base_t::get_data_impl(policy, alloc);
        const auto impl = detail::array_impl<const T*>::reinterpret(raw);
        return array<const T*>{new detail::array_impl<const T*>{ std::move(impl) }};
    }

    template <typename Policy>
    array<const T*> get_data(const Policy& policy) const {
        using alloc_t = detail::policy_allocator_t<Policy, const byte_t*>;
        return this->get_data(policy, alloc_t{ policy });
    }

    array<const T*> get_data() const {
        const detail::default_host_policy policy{};
        return this->get_data(policy);
    }

#ifdef ONEDAL_DATA_PARALLEL
    array<const T*> get_data(const detail::data_parallel_policy& policy,
                        const std::vector<sycl::event>& deps) const {
        sycl::event::wait_and_throw(deps);
        return this->get_data(policy);
    }

    array<const T*> get_data(sycl::queue& queue,
                      sycl::usm::alloc alloc = sycl::usm::alloc::shared,
                      const std::vector<sycl::event>& deps = {}) const {
        sycl::event::wait_and_throw(deps);

        using alloc_t = detail::data_parallel_allocator<const byte_t*>;
        const auto policy = detail::data_parallel_policy(queue);
        return this->get_data(policy, alloc_t{ policy, alloc });
    }
#endif // ONEDAL_DATA_PARALLEL

    array<T> get_chunk(std::int64_t chunk) const {
        const auto& raw = base_t::get_chunk_impl(chunk);
        const auto impl = detail::array_impl<T>::reinterpret(raw);
        return array<T>{ new detail::array_impl<T>{ std::move(impl) } };
    }

    chunked_array<T>& set_chunk(std::int64_t i, const array<T>& arr) {
        base_t::set_chunk_impl(i, arr);
        return *this;
    }

    std::int64_t get_count() const {
        constexpr auto size = static_cast<std::int64_t>(sizeof(T));
        const auto size_in_bytes = base_t::get_size_in_bytes();
        const std::int64_t count = size_in_bytes / size;
        ONEDAL_ASSERT(size * count == size_in_bytes);
        return count;
    }

    chunked_array<T> get_slice(std::int64_t first, std::int64_t last) const {
        constexpr std::int64_t size = sizeof(T);
        const auto first_in_bytes = first * size;
        const auto last_in_bytes = last * size;

        const auto res = base_t::get_slice_impl( //
                            first_in_bytes, last_in_bytes);
        return chunked_array<T>{ std::move(res) };
    }

private:
    chunked_array(const chunked_array_base& other)
        : chunked_array_base{ other } {}

    chunked_array(chunked_array_base&& other)
        : chunked_array_base{ std::forward<chunked_array>(other) } {}

    void serialize(detail::output_archive& ar) const {
        ar(get_dtype());

        base_t::serialize_impl(ar);
    }

    void deserialize(detail::input_archive& ar) {
        constexpr auto dtype = get_dtype();
        const auto ar_dtype = ar.pop<data_type>();
        ONEDAL_ASSERT(dtype == ar_dtype);

        base_t::deserialize_impl(dtype, ar);
    }

    static void swap(chunked_array<T>& lhs, chunked_array<T>& rhs) {
        std::swap(lhs.impl_, rhs.impl_);
    }

    constexpr static data_type get_dtype() {
        return detail::make_data_type<T>();
    }
}; // class chunked_array

} // namespace v2

using v2::chunked_array;

} // namespace oneapi::dal
