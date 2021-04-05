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

/// Archive interface for deserialization
class input_archive_iface {
public:
    virtual ~input_archive_iface() = default;
    virtual void process_scalar(void* data, data_type dtype) = 0;
    virtual void process_vector(void* data, std::int64_t count, data_type dtype) = 0;
};

/// Archive interface for serialization
class output_archive_iface {
public:
    virtual ~output_archive_iface() = default;
    virtual void process_scalar(const void* data, data_type dtype) = 0;
    virtual void process_vector(const void* data, std::int64_t count, data_type dtype) = 0;
};

struct serialization_accessor {
    template <typename Object, typename... Args>
    static auto serialize(const Object& object, Args&&... args) {
        return object.serialize(std::forward<Args>(args)...);
    }

    template <typename Object, typename... Args>
    static auto deserialize(Object& object, Args&&... args) {
        return object.deserialize(std::forward<Args>(args)...);
    }
};

template <typename T>
using enable_if_trivially_serializable_t = std::enable_if_t<std::is_arithmetic_v<T>>;

template <typename T>
using enable_if_user_serializable_t = std::enable_if_t<!std::is_arithmetic_v<T>>;

template <typename Iface>
class archive_base {
protected:
    explicit archive_base(Iface* impl) : impl_(impl) {}

    template <typename DerivedIface = Iface>
    DerivedIface& get_impl() {
        return static_cast<DerivedIface&>(*impl_);
    }

    template <typename DerivedIface = Iface>
    const DerivedIface& get_impl() const {
        return static_cast<const DerivedIface&>(*impl_);
    }

private:
    pimpl<Iface> impl_;
};

class output_archive : public archive_base<output_archive_iface> {
public:
    template <typename... Args>
    void operator()(Args&&... args) {
        (process(std::forward<Args>(args)), ...);
    }

    template <typename T>
    void range(const T* begin, const T* end) {
        ONEDAL_ASSERT(begin);
        ONEDAL_ASSERT(end);
        ONEDAL_ASSERT(begin <= end);
        process(begin, end);
    }

protected:
    explicit output_archive(output_archive_iface* impl)
            : archive_base<output_archive_iface>(impl) {}

private:
    template <typename T, enable_if_trivially_serializable_t<T>* = nullptr>
    void process(const T& value) {
        get_impl().process_scalar(&value, make_data_type<T>());
    }

    template <typename T, enable_if_user_serializable_t<T>* = nullptr>
    void process(const T& value) {
        serialization_accessor::serialize(value, *this);
    }

    template <typename T>
    void process(const T* begin, const T* end) {
        const std::int64_t count = end - begin;
        get_impl().process_vector(begin, count, make_data_type<T>());
    }
};

class input_archive : public archive_base<input_archive_iface> {
public:
    template <typename... Args>
    void operator()(Args&&... args) {
        (process(std::forward<Args>(args)), ...);
    }

    template <typename T>
    void range(T* begin, T* end) {
        ONEDAL_ASSERT(begin);
        ONEDAL_ASSERT(end);
        ONEDAL_ASSERT(begin <= end);
        process(begin, end);
    }

    template <typename T>
    T pop() {
        T value;
        (*this)(value);
        return value;
    }

protected:
    explicit input_archive(input_archive_iface* impl) : archive_base<input_archive_iface>(impl) {}

private:
    template <typename T, enable_if_trivially_serializable_t<T>* = nullptr>
    void process(T& value) {
        get_impl().process_scalar(&value, make_data_type<T>());
    }

    template <typename T, enable_if_user_serializable_t<T>* = nullptr>
    void process(T& value) {
        serialization_accessor::deserialize(value, *this);
    }

    template <typename T>
    void process(T* begin, T* end) {
        std::int64_t count = end - begin;
        get_impl().process_vector(begin, count, make_data_type<T>());
    }
};

template <typename T>
inline void serialize(const T& value, output_archive& ar) {
    ar(value);
}

template <typename T>
inline void deserialize(T& value, input_archive& ar) {
    ar(value);
}

} // namespace oneapi::dal::detail
