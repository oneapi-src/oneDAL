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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;

struct pod_type {
    std::int8_t x1;
    std::int16_t x2;
    std::int32_t x3;
    std::int64_t x4;
    float x5;
    double x6;

    void serialize(detail::output_archive& ar) const {
        ar(x1, x2, x3, x4, x5, x6);
    }

    void deserialize(detail::input_archive& ar) {
        ar(x1, x2, x3, x4, x5, x6);
    }
};

TEST("mock primitive type") {
    const float original = 3.14;

    te::mock_archive_state state;

    INFO("serialize") {
        te::mock_output_archive ar(state);
        detail::serialize(original, ar);

        REQUIRE(ar.prologue_call_count == 1);
        REQUIRE(ar.epilogue_call_count == 1);
        REQUIRE(state.get<float>(0) == original);
    }

    INFO("deserialize") {
        float deserialized;
        te::mock_input_archive ar(state);
        detail::deserialize(deserialized, ar);

        REQUIRE(ar.prologue_call_count == 1);
        REQUIRE(ar.epilogue_call_count == 1);
        REQUIRE(deserialized == original);
    }
}

enum class my_enum : std::uint64_t { one, two, three };

TEST("mock enum") {
    my_enum original = my_enum::two;

    te::mock_archive_state state;

    INFO("serialize") {
        te::mock_output_archive ar(state);
        detail::serialize(original, ar);

        REQUIRE(ar.prologue_call_count == 1);
        REQUIRE(ar.epilogue_call_count == 1);
        REQUIRE(state.get<my_enum>(0) == original);
    }

    INFO("deserialize") {
        my_enum deserialized;
        te::mock_input_archive ar(state);
        detail::deserialize(deserialized, ar);

        REQUIRE(ar.prologue_call_count == 1);
        REQUIRE(ar.epilogue_call_count == 1);
        REQUIRE(deserialized == original);
    }
}

TEST("mock POD type") {
    pod_type original;
    original.x1 = 2;
    original.x2 = -4;
    original.x3 = 8;
    original.x4 = -16;
    original.x5 = 32.5;
    original.x6 = 64.6;

    te::mock_archive_state state;

    INFO("serialize") {
        te::mock_output_archive ar(state);
        detail::serialize(original, ar);

        REQUIRE(state.get<std::int8_t>(0) == original.x1);
        REQUIRE(state.get<std::int16_t>(1) == original.x2);
        REQUIRE(state.get<std::int32_t>(2) == original.x3);
        REQUIRE(state.get<std::int64_t>(3) == original.x4);
        REQUIRE(state.get<float>(4) == original.x5);
        REQUIRE(state.get<double>(5) == original.x6);
    }

    INFO("deserialize") {
        pod_type deserialized;
        te::mock_input_archive ar(state);
        detail::deserialize(deserialized, ar);

        REQUIRE(deserialized.x1 == original.x1);
        REQUIRE(deserialized.x2 == original.x2);
        REQUIRE(deserialized.x3 == original.x3);
        REQUIRE(deserialized.x4 == original.x4);
        REQUIRE(deserialized.x5 == original.x5);
        REQUIRE(deserialized.x6 == original.x6);
    }
}

struct vector_type {
    std::vector<float> vec;

    static vector_type even_odd(std::int64_t count, float even, float odd) {
        vector_type result;
        result.vec.clear();
        result.vec.reserve(count);

        for (std::int64_t i = 0; i < count; i++) {
            if (i % 2 > 0) {
                result.vec.push_back(odd);
            }
            else {
                result.vec.push_back(even);
            }
        }

        return result;
    }

    void serialize(detail::output_archive& ar) const {
        ar(static_cast<std::int64_t>(vec.size()));
        ar.range(vec.data(), vec.data() + vec.size());
    }

    void deserialize(detail::input_archive& ar) {
        const std::int64_t count = ar.pop<std::int64_t>();
        ONEDAL_ASSERT(count >= 0);

        vec.resize(count);
        ar.range(vec.data(), vec.data() + vec.size());
    }
};

TEST("mock non-trivially copyable type") {
    const std::int64_t element_count = 10;
    const float even_filler = 3.14f;
    const float odd_filler = 2.71f;
    const vector_type original = //
        vector_type::even_odd(element_count, even_filler, odd_filler);

    te::mock_archive_state state;

    INFO("serialize") {
        te::mock_output_archive ar(state);
        detail::serialize(original, ar);

        REQUIRE(state.get<std::int64_t>(0) == element_count);
        for (std::int64_t i = 1; i < element_count + 1; i++) {
            CAPTURE(i);
            REQUIRE(state.get<float>(i) == original.vec[i - 1]);
        }
    }

    INFO("deserialize") {
        vector_type deserialized;
        te::mock_input_archive ar(state);
        detail::deserialize(deserialized, ar);

        REQUIRE(deserialized.vec.size() == element_count);
        for (std::int64_t i = 0; i < element_count; i++) {
            CAPTURE(i);
            REQUIRE(deserialized.vec[i] == original.vec[i]);
        }
    }
}

class polymorphic_iface {
public:
    virtual ~polymorphic_iface() = default;
    virtual std::string get_name() = 0;
};

class polymorphic {
    friend struct detail::serialization_accessor;

public:
    polymorphic() = default;

    std::string get_name() const {
        return impl_->get_name();
    }

protected:
    explicit polymorphic(polymorphic_iface* impl) : impl_(impl) {}

    template <typename Impl>
    const Impl& get_impl() const {
        return static_cast<const Impl&>(*impl_);
    }

    template <typename Impl>
    Impl& get_impl() {
        return static_cast<Impl&>(*impl_);
    }

    void init_impl(polymorphic_iface* impl) {
        impl_.reset(impl);
    }

private:
    void serialize(detail::output_archive& ar) const {
        detail::serialize_polymorphic_shared(impl_, ar);
    }

    void deserialize(detail::input_archive& ar) {
        detail::deserialize_polymorphic_shared(impl_, ar);
    }

    detail::pimpl<polymorphic_iface> impl_;
};

inline constexpr std::uint64_t a_serialization_id = 77777;
inline constexpr std::uint64_t b_serialization_id = 88888;

class polymorphic_impl_a : public polymorphic_iface,
                           public detail::serializable<a_serialization_id> {
public:
    std::string get_name() override {
        return "A";
    }

    void serialize(detail::output_archive& ar) const override {
        ar(x1, x2);
    }

    void deserialize(detail::input_archive& ar) override {
        ar(x1, x2);
    }

    float x1 = 0.0f;
    std::int64_t x2 = 0;
};
__ONEDAL_REGISTER_SERIALIZABLE__(polymorphic_impl_a)

class polymorphic_impl_b : public polymorphic_iface,
                           public detail::serializable<b_serialization_id> {
public:
    std::string get_name() override {
        return "B";
    }

    void serialize(detail::output_archive& ar) const override {
        ar(x1, x2);
    }

    void deserialize(detail::input_archive& ar) override {
        ar(x1, x2);
    }

    double x1 = 0.0;
    std::int32_t x2 = 0;
};
__ONEDAL_REGISTER_SERIALIZABLE__(polymorphic_impl_b)

class derived_a : public polymorphic {
public:
    derived_a() : polymorphic(new polymorphic_impl_a{}) {}

    float get_x1() const {
        return get_impl<polymorphic_impl_a>().x1;
    }

    derived_a& set_x1(float value) {
        get_impl<polymorphic_impl_a>().x1 = value;
        return *this;
    }

    std::int64_t get_x2() const {
        return get_impl<polymorphic_impl_a>().x2;
    }

    derived_a& set_x2(std::int64_t value) {
        get_impl<polymorphic_impl_a>().x2 = value;
        return *this;
    }

    void deserialize(detail::input_archive& ar) {
        init_impl(detail::deserialize_polymorphic<polymorphic_iface>(ar, { a_serialization_id }));
    }
};

class derived_b : public polymorphic {
public:
    derived_b() : polymorphic(new polymorphic_impl_b{}) {}

    double get_x1() const {
        return get_impl<polymorphic_impl_a>().x1;
    }

    derived_b& set_x1(double value) {
        get_impl<polymorphic_impl_b>().x1 = value;
        return *this;
    }

    std::int32_t get_x2() const {
        return get_impl<polymorphic_impl_a>().x2;
    }

    derived_b& set_x2(std::int32_t value) {
        get_impl<polymorphic_impl_b>().x2 = value;
        return *this;
    }

    void deserialize(detail::input_archive& ar) {
        init_impl(detail::deserialize_polymorphic<polymorphic_iface>(ar, { b_serialization_id }));
    }
};

TEST("mock polymorphic type") {
    constexpr float a_x1 = 3.14f;
    constexpr std::int64_t a_x2 = -100;
    const polymorphic original = derived_a{}.set_x1(a_x1).set_x2(a_x2);
    ONEDAL_ASSERT(original.get_name() == "A");

    te::mock_archive_state state;

    INFO("serialize") {
        te::mock_output_archive ar(state);
        detail::serialize(original, ar);

        REQUIRE(state.get<std::uint8_t>(0) == 1);
        REQUIRE(state.get<std::uint64_t>(1) == polymorphic_impl_a::serialization_id());
        REQUIRE(state.get<float>(2) == a_x1);
        REQUIRE(state.get<std::int64_t>(3) == a_x2);
    }

    SECTION("deserialize to base type") {
        polymorphic deserialized;
        te::mock_input_archive ar(state);
        detail::deserialize(deserialized, ar);

        REQUIRE(deserialized.get_name() == "A");
        REQUIRE(static_cast<derived_a&>(deserialized).get_x1() == a_x1);
        REQUIRE(static_cast<derived_a&>(deserialized).get_x2() == a_x2);
    }

    SECTION("deserialize to exact type") {
        derived_a deserialized;
        te::mock_input_archive ar(state);
        detail::deserialize(deserialized, ar);

        REQUIRE(deserialized.get_name() == "A");
        REQUIRE(deserialized.get_x1() == a_x1);
        REQUIRE(deserialized.get_x2() == a_x2);
    }

    SECTION("deserialize to wrong type") {
        derived_b deserialized;
        te::mock_input_archive ar(state);

        REQUIRE_THROWS_AS(detail::deserialize(deserialized, ar), invalid_argument);
    }
}

} // namespace oneapi::dal::test
