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

#include <vector>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/detail/serialization.hpp"

namespace oneapi::dal::test {

class mock_archive_entry {
public:
    explicit mock_archive_entry(const std::uint8_t* data, data_type dtype) : dtype_(dtype) {
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);
        data_.assign(data, data + dtype_size);
    }

    template <typename T>
    T get() const {
        if (dtype_ != detail::make_data_type<T>()) {
            throw std::invalid_argument{ "Data types do not match" };
        }
        const T* data = reinterpret_cast<const T*>(data_.data());
        return *data;
    }

    void write_to(std::uint8_t* dst, data_type dtype) const {
        if (dtype != dtype_) {
            throw std::invalid_argument{ "Data types do not match" };
        }
        std::copy(data_.begin(), data_.end(), dst);
    }

private:
    std::vector<std::uint8_t> data_;
    data_type dtype_;
};

class mock_archive_state {
public:
    using entries_t = std::vector<mock_archive_entry>;

    mock_archive_state() : entries_(new entries_t{}) {}

    const entries_t& get() const {
        return *entries_;
    }

    entries_t& get() {
        return *entries_;
    }

    template <typename T>
    T get(std::int64_t index) const {
        ONEDAL_ASSERT(index < std::int64_t(get().size()));
        return get()[index].template get<T>();
    }

private:
    std::shared_ptr<entries_t> entries_;
};

class mock_input_archive {
public:
    std::int64_t prologue_call_count = 0;
    std::int64_t epilogue_call_count = 0;

    explicit mock_input_archive(const mock_archive_state& state) : state_(state) {}

    void operator()(void* data, data_type dtype, std::int64_t count = 1) {
        check_possition();

        for (std::int64_t i = 0; i < count; i++) {
            load(i, data, dtype);
        }
    }

    void prologue() {
        prologue_call_count++;
    }

    void epilogue() {
        epilogue_call_count++;
    }

private:
    void check_possition() const {
        if (position_ >= state_.get().size()) {
            throw std::runtime_error{ "We reached the end of input stream" };
        }
    }

    void load(std::int64_t index, void* data, data_type dtype) {
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);
        std::uint8_t* data_bytes = static_cast<std::uint8_t*>(data);
        state_.get()[position_++].write_to(data_bytes + index * dtype_size, dtype);
    }

    mock_archive_state state_;
    std::size_t position_ = 0;
};

class mock_output_archive {
public:
    std::int64_t prologue_call_count = 0;
    std::int64_t epilogue_call_count = 0;

    explicit mock_output_archive(const mock_archive_state& state) : state_(state) {}

    void operator()(const void* data, data_type dtype, std::int64_t count = 1) {
        state_.get().reserve(state_.get().size() + count);

        for (std::int64_t i = 0; i < count; i++) {
            save(i, data, dtype);
        }
    }

    void prologue() {
        prologue_call_count++;
    }

    void epilogue() {
        epilogue_call_count++;
    }

private:
    void save(std::int64_t index, const void* data, data_type dtype) {
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);
        const std::uint8_t* data_bytes = static_cast<const std::uint8_t*>(data);
        state_.get().emplace_back(data_bytes + index * dtype_size, dtype);
    }

    mock_archive_state state_;
};

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

TEST("mock primitive type") {
    const float original = 3.14;

    mock_archive_state state;

    INFO("serialize") {
        mock_output_archive ar(state);
        detail::serialize(original, ar);

        REQUIRE(ar.prologue_call_count == 1);
        REQUIRE(ar.epilogue_call_count == 1);
        REQUIRE(state.get<float>(0) == original);
    }

    INFO("deserialize") {
        float deserialized;
        mock_input_archive ar(state);
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

    mock_archive_state state;

    INFO("serialize") {
        mock_output_archive ar(state);
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
        mock_input_archive ar(state);
        detail::deserialize(deserialized, ar);

        REQUIRE(deserialized.x1 == original.x1);
        REQUIRE(deserialized.x2 == original.x2);
        REQUIRE(deserialized.x3 == original.x3);
        REQUIRE(deserialized.x4 == original.x4);
        REQUIRE(deserialized.x5 == original.x5);
        REQUIRE(deserialized.x6 == original.x6);
    }
}

TEST("mock non-trivially copyable type") {
    const std::int64_t element_count = 10;
    const float even_filler = 3.14f;
    const float odd_filler = 2.71f;
    const vector_type original = //
        vector_type::even_odd(element_count, even_filler, odd_filler);

    mock_archive_state state;

    INFO("serialize") {
        mock_output_archive ar(state);
        detail::serialize(original, ar);

        REQUIRE(state.get<std::int64_t>(0) == element_count);
        for (std::int64_t i = 1; i < element_count + 1; i++) {
            CAPTURE(i);
            REQUIRE(state.get<float>(i) == original.vec[i - 1]);
        }
    }

    INFO("deserialize") {
        vector_type deserialized;
        mock_input_archive ar(state);
        detail::deserialize(deserialized, ar);

        REQUIRE(deserialized.vec.size() == element_count);
        for (std::int64_t i = 0; i < element_count; i++) {
            CAPTURE(i);
            REQUIRE(deserialized.vec[i] == original.vec[i]);
        }
    }
}

} // namespace oneapi::dal::test
