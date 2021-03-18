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

class output_data_entry {
public:
    explicit output_data_entry(const void* data, data_type dtype) : dtype_(dtype) {
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);
        const std::uint8_t* data_bytes = static_cast<const std::uint8_t*>(data);
        data_.assign(data_bytes, data_bytes + dtype_size);
    }

    template <typename T>
    T get() const {
        if (dtype_ != detail::make_data_type<T>()) {
            throw std::invalid_argument{ "Data types do not match" };
        }
        const T* data = reinterpret_cast<const T*>(data_.data());
        return *data;
    }

private:
    std::vector<std::uint8_t> data_;
    data_type dtype_;
};

class mock_output_archive_impl : public detail::archive_iface {
public:
    void process(void* data, data_type dtype) override {
        entries_.emplace_back(data, dtype);
    }

    void process(void*& data, std::int64_t count, data_type dtype) override {}

    template <typename T>
    T get(std::int64_t index) const {
        ONEDAL_ASSERT(index < std::int64_t(entries_.size()));
        return entries_[index].template get<T>();
    }

private:
    std::vector<output_data_entry> entries_;
};

class mock_output_archive : public detail::archive {
public:
    mock_output_archive() : archive(new mock_output_archive_impl{}, false) {}

    template <typename T>
    T get(std::int64_t index) const {
        return get_impl<mock_output_archive_impl>().template get<T>(index);
    }
};

struct pod_type {
    std::int8_t x1;
    std::int16_t x2;
    std::int32_t x3;
    std::int64_t x4;
    float x5;
    double x6;

    void serialize(detail::archive& ar) {
        ar(x1, x2, x3, x4, x5, x6);
    }
};

TEST("save POD type") {
    mock_output_archive ar;

    pod_type pod;
    pod.x1 = 2;
    pod.x2 = -4;
    pod.x3 = 8;
    pod.x4 = -16;
    pod.x5 = 32.5;
    pod.x6 = 64.6;

    detail::save(pod, ar);

    REQUIRE(ar.get<std::int8_t>(0) == pod.x1);
    REQUIRE(ar.get<std::int16_t>(1) == pod.x2);
    REQUIRE(ar.get<std::int32_t>(2) == pod.x3);
    REQUIRE(ar.get<std::int64_t>(3) == pod.x4);
    REQUIRE(ar.get<float>(4) == pod.x5);
    REQUIRE(ar.get<double>(5) == pod.x6);
}

} // namespace oneapi::dal::test
