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

#include <unordered_map>
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::detail {

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

/// Archive interface for deserialization
class input_archive_iface {
public:
    virtual ~input_archive_iface() = default;
    virtual void prologue() = 0;
    virtual void epilogue() = 0;
    virtual void deserialize(void* data, data_type dtype) = 0;
    virtual void deserialize(void* data, data_type dtype, std::int64_t count) = 0;
};

/// Archive interface for serialization
class output_archive_iface {
public:
    virtual ~output_archive_iface() = default;
    virtual void prologue() = 0;
    virtual void epilogue() = 0;
    virtual void serialize(const void* data, data_type dtype) = 0;
    virtual void serialize(const void* data, data_type dtype, std::int64_t count) = 0;
};

template <typename T>
using enable_if_trivially_serializable_t = std::enable_if_t<std::is_arithmetic_v<T>>;

template <typename T>
using enable_if_user_serializable_t = std::enable_if_t<!std::is_arithmetic_v<T>>;

template <typename Iface>
class archive_base {
protected:
    explicit archive_base(Iface* impl) noexcept : impl_(impl) {
        ONEDAL_ASSERT(impl);
    }

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

template <typename Archive>
class output_archive_impl : public output_archive_iface {
public:
    explicit output_archive_impl(Archive& archive) : archive_(archive) {}

    void prologue() override {
        archive_.prologue();
    }

    void epilogue() override {
        archive_.epilogue();
    }

    void serialize(const void* data, data_type dtype) override {
        archive_(data, dtype);
    }

    void serialize(const void* data, data_type dtype, std::int64_t count) override {
        archive_(data, dtype, count);
    }

private:
    std::remove_reference_t<Archive>& archive_;
};

class output_archive : public archive_base<output_archive_iface> {
    using base_t = archive_base<output_archive_iface>;

public:
    template <typename Archive>
    explicit output_archive(Archive& archive)
            : base_t(new output_archive_impl<Archive>{ archive }) {}

    void prologue() {
        get_impl().prologue();
    }

    void epilogue() {
        get_impl().epilogue();
    }

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

private:
    template <typename T, enable_if_trivially_serializable_t<T>* = nullptr>
    void process(const T& value) {
        get_impl().serialize(&value, make_data_type<T>());
    }

    template <typename T, enable_if_user_serializable_t<T>* = nullptr>
    void process(const T& value) {
        serialization_accessor::serialize(value, *this);
    }

    template <typename T>
    void process(const T* begin, const T* end) {
        const std::int64_t count = end - begin;
        get_impl().serialize(begin, make_data_type<T>(), count);
    }
};

template <typename Archive>
class input_archive_impl : public input_archive_iface {
public:
    explicit input_archive_impl(Archive& archive) : archive_(archive) {}

    void prologue() override {
        archive_.prologue();
    }

    void epilogue() override {
        archive_.epilogue();
    }

    void deserialize(void* data, data_type dtype) override {
        archive_(data, dtype);
    }

    void deserialize(void* data, data_type dtype, std::int64_t count) override {
        archive_(data, dtype, count);
    }

private:
    std::remove_reference_t<Archive>& archive_;
};

class input_archive : public archive_base<input_archive_iface> {
    using base_t = archive_base<input_archive_iface>;

public:
    template <typename Archive>
    explicit input_archive(Archive& archive) : base_t(new input_archive_impl<Archive>{ archive }) {}

    void prologue() {
        get_impl().prologue();
    }

    void epilogue() {
        get_impl().epilogue();
    }

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

private:
    template <typename T, enable_if_trivially_serializable_t<T>* = nullptr>
    void process(T& value) {
        get_impl().deserialize(&value, make_data_type<T>());
    }

    template <typename T, enable_if_user_serializable_t<T>* = nullptr>
    void process(T& value) {
        serialization_accessor::deserialize(value, *this);
    }

    template <typename T>
    void process(T* begin, T* end) {
        std::int64_t count = end - begin;
        get_impl().deserialize(begin, make_data_type<T>(), count);
    }
};

/// Interface that each serializable class must implement to support polymorphic serialization
class serializable_iface {
public:
    virtual ~serializable_iface() = default;
    virtual std::uint64_t get_serialization_id() const = 0;
    virtual void serialize(detail::output_archive& ar) const = 0;
    virtual void deserialize(detail::input_archive& ar) = 0;
};

class serializable_factory_iface {
public:
    virtual ~serializable_factory_iface() = default;
    virtual serializable_iface* make() const = 0;
};

class serializable_registry {
public:
    static serializable_registry& instance() {
        static serializable_registry factory;
        return factory;
    }

    template <typename T>
    T* make(std::uint64_t serialization_id) {
        ONEDAL_ASSERT(factories_.find(serialization_id) != factories_.end(),
                      "Factory with requested serialization_id was not registered");

        auto serializable = factories_[serialization_id]->make();
        ONEDAL_ASSERT(serializable, "Factory produced null object");

        if (serializable->get_serialization_id() != serialization_id) {
            throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
        }

        auto object = dynamic_cast<T*>(serializable);
        if (!object) {
            throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
        }

        return object;
    }

    void register_factory(std::uint64_t serialization_id,
                          const serializable_factory_iface* factory) {
        ONEDAL_ASSERT(factory);
        ONEDAL_ASSERT(factories_.find(serialization_id) == factories_.end(),
                      "Factory with the provided serialization_id is already registered");

        factories_[serialization_id] = factory;
    }

private:
    serializable_registry() = default;

    // TODO: Use own implementation of hash map
    std::unordered_map<std::uint64_t, const serializable_factory_iface*> factories_;
};

template <typename T>
class default_serializable_factory : public serializable_factory_iface {
public:
    static const default_serializable_factory& get_default() {
        static default_serializable_factory factory;
        return factory;
    }

    T* make() const override {
        return new T{};
    }
};

template <std::uint64_t SerializationId>
class serializable : public base, public serializable_iface {
public:
    static std::uint64_t serialization_id() {
        return SerializationId;
    }

    std::uint64_t get_serialization_id() const override {
        return SerializationId;
    }
};

#define __ONEDAL_SERIALIZATION_CONCAT4__(_1, _2, _3, _4) _1##_2##_3##_4
#define __ONEDAL_SERIALIZATION_DUMMY_NAME_IMPL__(_1, _2) \
    __ONEDAL_SERIALIZATION_CONCAT4__(__, _1, _2, __)
#define __ONEDAL_SERIALIZATION_DUMMY_NAME__(name) \
    __ONEDAL_SERIALIZATION_DUMMY_NAME_IMPL__(name, __LINE__)

#define __ONEDAL_REGISTER_SERIALIZABLE__(T)                                         \
    static char __ONEDAL_SERIALIZATION_DUMMY_NAME__(register_serializable_func)() { \
        oneapi::dal::detail::serializable_registry::instance().register_factory(    \
            T::serialization_id(),                                                  \
            &oneapi::dal::detail::default_serializable_factory<T>::get_default());  \
        return 0;                                                                   \
    }                                                                               \
    [[maybe_unused]] volatile static char __ONEDAL_SERIALIZATION_DUMMY_NAME__(      \
        register_serializable) =                                                    \
        __ONEDAL_SERIALIZATION_DUMMY_NAME__(register_serializable_func)();

template <typename T>
inline serializable_iface& get_serializable(T* object) {
    auto serializable = dynamic_cast<serializable_iface*>(object);
    if (!serializable) {
        throw invalid_argument{ error_messages::object_is_not_serializable() };
    }
    return *serializable;
}

template <typename T>
inline serializable_iface& get_serializable(const std::shared_ptr<T>& object) {
    return get_serializable(object.get());
}

template <typename T>
inline void serialize_polymorphic(const std::shared_ptr<T>& serializable_object,
                                  output_archive& archive) {
    auto& serializable = get_serializable(serializable_object);
    archive(serializable.get_serialization_id());
    serializable.serialize(archive);
}

template <typename T>
inline void deserialize_polymorphic(std::shared_ptr<T>& serializable_object,
                                    input_archive& archive,
                                    bool strict_type_match = true) {
    const auto serialization_id = archive.pop<std::uint64_t>();
    if (serializable_object && strict_type_match) {
        if (get_serializable(serializable_object).get_serialization_id() != serialization_id) {
            throw invalid_argument{ error_messages::archive_content_does_not_match_type() };
        }
    }

    T* object_ptr = serializable_registry::instance().make<T>(serialization_id);
    get_serializable(object_ptr).deserialize(archive);
    serializable_object.reset(object_ptr);
}

template <typename T, typename OutputArchive>
inline void serialize(const T& value, OutputArchive& archive) {
    output_archive internal_archive{ archive };
    internal_archive.prologue();
    internal_archive(value);
    internal_archive.epilogue();
}

template <typename T, typename InputArchive>
inline void deserialize(T& value, InputArchive& archive) {
    input_archive internal_archive{ archive };
    internal_archive.prologue();
    internal_archive(value);
    internal_archive.epilogue();
}

class binary_output_archive {
public:
    void prologue() {}
    void epilogue() {}
    void operator()(const void* data, data_type dtype, std::int64_t count = 1) {}
};

class binary_input_archive {
public:
    void prologue() {}
    void epilogue() {}
    void operator()(void* data, data_type dtype, std::int64_t count = 1) {}
};

} // namespace oneapi::dal::detail
