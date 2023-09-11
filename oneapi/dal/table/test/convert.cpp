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

// TODO: These tests need to be refactored and moved to corresponding submodules
//       Test for conversion should be moved to dal/table/backend

#include <array>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/linalg.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/table/backend/convert.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend::test {

namespace te = oneapi::dal::test::engine;
namespace la = te::linalg;

class convert_test : public te::policy_fixture {
public:
    const std::int32_t* get_data(std::int32_t) const {
        return int_data_.data();
    }

    const std::int64_t* get_data(std::int64_t) const {
        return int64_data_.data();
    }

    const float* get_data(float) const {
        return float_data_.data();
    }

    const double* get_data(double) const {
        return double_data_.data();
    }

    std::int64_t get_element_count() const {
        return element_count_;
    }

    template <typename Src, typename Dst>
    void test_host2host_conversion(std::int64_t stride = 1) {
        const Src* source_data = get_data(Src{});
        auto dst_host = la::matrix<Dst>::zeros({ 1, element_count_ });

        convert_vector(dal::detail::default_host_policy{},
                       source_data,
                       dst_host.get_mutable_data(),
                       dal::detail::make_data_type<Src>(),
                       dal::detail::make_data_type<Dst>(),
                       stride,
                       stride,
                       get_element_count() / stride);

        check_conversion<Src, Dst>(source_data, dst_host.get_data(), stride);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Src, typename Dst>
    void test_device2device_conversion(std::int64_t stride = 1) {
        auto& q = get_queue();

        const Src* source_data = get_data(Src{});
        auto src_device = la::matrix<Src>::wrap(source_data, { 1, element_count_ }).to_device(q);
        auto dst_device = la::matrix<Dst>::zeros(q, { 1, element_count_ });

        convert_vector(q,
                       src_device.get_data(),
                       dst_device.get_mutable_data(),
                       dal::detail::make_data_type<Src>(),
                       dal::detail::make_data_type<Dst>(),
                       stride,
                       stride,
                       get_element_count() / stride);

        auto dst_host = dst_device.to_host();
        check_conversion<Src, Dst>(source_data, dst_host.get_data(), stride);
    }

    template <typename Src, typename Dst>
    void test_host2device_conversion(std::int64_t stride = 1) {
        auto& q = get_queue();

        const Src* source_data = get_data(Src{});
        auto dst_device = la::matrix<Dst>::zeros(q, { 1, element_count_ });

        convert_vector(q,
                       source_data,
                       dst_device.get_mutable_data(),
                       dal::detail::make_data_type<Src>(),
                       dal::detail::make_data_type<Dst>(),
                       stride,
                       stride,
                       get_element_count() / stride);

        auto dst_host = dst_device.to_host();
        check_conversion<Src, Dst>(source_data, dst_host.get_data(), stride);
    }

    template <typename Src, typename Dst>
    void test_device2host_conversion(std::int64_t stride = 1) {
        auto& q = get_queue();

        const Src* source_data = get_data(Src{});
        auto src_device = la::matrix<Src>::wrap(source_data, { 1, element_count_ }).to_device(q);
        auto dst_host = la::matrix<Dst>::zeros({ 1, element_count_ });

        convert_vector(q,
                       src_device.get_data(),
                       dst_host.get_mutable_data(),
                       dal::detail::make_data_type<Src>(),
                       dal::detail::make_data_type<Dst>(),
                       stride,
                       stride,
                       get_element_count() / stride);

        check_conversion<Src, Dst>(source_data, dst_host.get_data(), stride);
    }
#endif

private:
    template <typename Src, typename Dst>
    void check_conversion(const Src* original_src, const Dst* dst_to_check, std::int64_t stride) {
        for (std::int64_t i = 0; i < element_count_; i++) {
            if (i % stride == 0) {
                REQUIRE(dst_to_check[i] == Dst(original_src[i]));
            }
            else {
                REQUIRE(dst_to_check[i] == Dst(0));
            }
        }
    }

    static constexpr std::int64_t element_count_ = 6;
    std::array<std::int32_t, element_count_> int_data_ = { 1, -2, //
                                                           3, -4, //
                                                           5, -6 };
    std::array<std::int64_t, element_count_> int64_data_ = { 1, -2, //
                                                             3, -4, //
                                                             5, -6 };
    std::array<float, element_count_> float_data_ = { 1.6f, -2.5f, //
                                                      3.4f, -4.3f, //
                                                      5.2f, -6.1f };
    std::array<double, element_count_> double_data_ = { 1.6, -2.5, //
                                                        3.4, -4.3, //
                                                        5.2, -6.1 };
};

// host -> host tests
TEST_M(convert_test, "host2host convert identical types", "[host2host]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("int -> int") {
            test_host2host_conversion<std::int32_t, std::int32_t>(stride);
        }
        SECTION("float -> float") {
            test_host2host_conversion<float, float>(stride);
        }
        SECTION("double -> double") {
            test_host2host_conversion<double, double>(stride);
        }
    }
}

TEST_M(convert_test, "convert types of the same size host2host", "[host2host]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("float -> std::int32_t") {
            test_host2host_conversion<float, std::int32_t>(stride);
        }
        SECTION("std::int32_t -> float") {
            test_host2host_conversion<std::int32_t, float>(stride);
        }
    }
}

TEST_M(convert_test, "convert types of the different sizes host2host", "[host2host]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("float -> double") {
            test_host2host_conversion<float, double>(stride);
        }
        SECTION("double -> float") {
            test_host2host_conversion<double, float>(stride);
        }
        SECTION("std::int32_t -> double") {
            test_host2host_conversion<std::int32_t, double>(stride);
        }
        SECTION("double -> std::int32_t") {
            test_host2host_conversion<double, std::int32_t>(stride);
        }
    }
}

// device -> device tests
#ifdef ONEDAL_DATA_PARALLEL
TEST_M(convert_test, "device2device convert identical types", "[device2device]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("int -> int") {
            test_device2device_conversion<std::int32_t, std::int32_t>(stride);
        }
        SECTION("float -> float") {
            test_device2device_conversion<float, float>(stride);
        }
        SECTION("double -> double") {
            SKIP_IF(!this->get_policy().has_native_float64());
            test_device2device_conversion<double, double>(stride);
        }
    }
}

TEST_M(convert_test, "convert types of the same size device2device", "[device2device]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("float -> std::int32_t") {
            test_device2device_conversion<float, std::int32_t>(stride);
        }
        SECTION("std::int32_t -> float") {
            test_device2device_conversion<std::int32_t, float>(stride);
        }
    }
}

TEST_M(convert_test, "convert types of the different sizes device2device", "[device2device]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("float -> std::int64_t") {
            test_device2device_conversion<float, std::int64_t>(stride);
        }
        SECTION("std::int64_t -> float") {
            test_device2device_conversion<std::int64_t, float>(stride);
        }
        SECTION("std::int32_t -> std::int64_t") {
            test_device2device_conversion<std::int32_t, std::int64_t>(stride);
        }
        SECTION("std::int64_t -> std::int32_t") {
            test_device2device_conversion<std::int64_t, std::int32_t>(stride);
        }
    }
}
#endif

// host -> device tests
#ifdef ONEDAL_DATA_PARALLEL
TEST_M(convert_test, "host2device convert identical types without stride", "[host2device]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("int -> int") {
            test_host2device_conversion<std::int32_t, std::int32_t>(stride);
        }
        SECTION("float -> float") {
            test_host2device_conversion<float, float>(stride);
        }
        SECTION("double -> double") {
            SKIP_IF(!this->get_policy().has_native_float64());
            test_host2device_conversion<double, double>(stride);
        }
    }
}

TEST_M(convert_test, "host2device convert types of the same size", "[host2device]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("float -> std::int32_t") {
            test_host2device_conversion<float, std::int32_t>(stride);
        }
        SECTION("std::int32_t -> float") {
            test_host2device_conversion<std::int32_t, float>(stride);
        }
    }
}

TEST_M(convert_test, "host2device convert types of the different sizes", "[host2device]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("float -> std::int64_t") {
            test_host2device_conversion<float, std::int64_t>(stride);
        }
        SECTION("std::int64_t -> float") {
            test_host2device_conversion<std::int64_t, float>(stride);
        }
        SECTION("std::int32_t -> std::int64_t") {
            test_host2device_conversion<std::int32_t, std::int64_t>(stride);
        }
        SECTION("std::int64_t -> std::int32_t") {
            test_host2device_conversion<std::int64_t, std::int32_t>(stride);
        }
    }
}
#endif

// device -> host tests
#ifdef ONEDAL_DATA_PARALLEL
TEST_M(convert_test, "device2host convert identical types", "[device2host]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("int -> int") {
            test_device2host_conversion<std::int32_t, std::int32_t>(stride);
        }
        SECTION("float -> float") {
            test_device2host_conversion<float, float>(stride);
        }
        SECTION("double -> double") {
            SKIP_IF(!this->get_policy().has_native_float64());
            test_device2host_conversion<double, double>(stride);
        }
    }
}

TEST_M(convert_test, "device2host convert types of the same size", "[device2host]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("float -> std::int32_t") {
            test_device2host_conversion<float, std::int32_t>(stride);
        }
        SECTION("std::int32_t -> float") {
            test_device2host_conversion<std::int32_t, float>(stride);
        }
    }
}

TEST_M(convert_test, "device2host convert types of the different sizes", "[device2host]") {
    const std::int64_t stride = GENERATE_COPY(1, 2, 3);
    SECTION(fmt::format("stride = {}", stride)) {
        SECTION("float -> std::int64_t") {
            test_device2host_conversion<float, std::int64_t>(stride);
        }
        SECTION("std::int64_t -> float") {
            test_device2host_conversion<std::int64_t, float>(stride);
        }
        SECTION("std::int32_t -> std::int64_t") {
            test_device2host_conversion<std::int32_t, std::int64_t>(stride);
        }
        SECTION("std::int64_t -> std::int32_t") {
            test_device2host_conversion<std::int64_t, std::int32_t>(stride);
        }
    }
}
#endif

} // namespace oneapi::dal::backend::test
