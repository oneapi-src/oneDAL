/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::detail {

struct serialize_accessor {
    template <typename Object, typename... Args>
    static auto call(Object&& object, Args&&... args) {
        return std::forward<Object>(object).serialize(std::forward<Args>(args)...);
    }
};

class archive_iface {
public:
    virtual ~archive_iface() = default;
    virtual void process(void* data, data_type dtype) = 0;
    virtual void process(void*& data, std::int64_t count, data_type dtype) = 0;
};

class archive {
public:
    template <typename... Args>
    void operator()(Args&&... args) {
        (process(std::forward<Args>(args)), ...);
    }

    bool is_load() const {
        return is_load_;
    }

    bool is_save() const {
        return !is_load_;
    }

protected:
    explicit archive(archive_iface* impl, bool is_load) : impl_(impl), is_load_(is_load) {}

    template <typename Derived>
    Derived& get_impl() {
        return static_cast<Derived&>(*impl_);
    }

    template <typename Derived>
    const Derived& get_impl() const {
        return static_cast<const Derived&>(*impl_);
    }

private:
    template <typename T>
    using enable_if_primitive_t = std::enable_if_t<std::is_arithmetic_v<T>>;

    template <typename T>
    using enable_if_not_primitive_t = std::enable_if_t<!std::is_arithmetic_v<T>>;

    template <typename T, enable_if_primitive_t<T>* = nullptr>
    void process(const T& value) {
        T& mutable_value = const_cast<T&>(value);
        impl_->process(&mutable_value, make_data_type<T>());
    }

    template <typename T, enable_if_not_primitive_t<T>* = nullptr>
    void process(const T& value) {
        T& mutable_value = const_cast<T&>(value);
        serialize_accessor::call(mutable_value, *this);
    }

    pimpl<archive_iface> impl_;
    bool is_load_;
};

class binary_output_archive_impl : public archive_iface {};

class binary_output_archive : public archive {};

template <typename T>
void load(T& value, archive& ar) {
    ONEDAL_ASSERT(ar.is_load());
    ar(value);
}

template <typename T>
void save(T& value, archive& ar) {
    ONEDAL_ASSERT(ar.is_save());
    ar(value);
}

} // namespace oneapi::dal::detail
