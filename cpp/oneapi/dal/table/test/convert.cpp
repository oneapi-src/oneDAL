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

// TODO: These tests need to be refactored and moved to corresponding submodules
//       Test for conversion should be moved to dal/table/backend
//       Test for gather/scatter should be moved dal/backend

#include <array>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/linalg.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/table/backend/convert.hpp"
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend::test {

namespace te = oneapi::dal::test::engine;
namespace la = te::linalg;

// TEST("scatter from host to device", "[device-usm]") {
//     DECLARE_TEST_POLICY(policy);
//     auto& q = policy.get_queue();

//     constexpr std::int64_t element_count = 6;
//     float host_ptr[element_count] = { 1.6f, -2.5f, //
//                                       3.4f, -4.3f, //
//                                       5.2f, -6.1f };

//     scatter_host2device();
// }

class convert_test : public te::policy_fixture {
public:
    const std::int32_t* get_data(std::int32_t) const {
        return int_data_.data();
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
        auto dst_host = la::matrix<Dst>::empty({ 1, element_count_ });

        convert_vector(dal::detail::default_host_policy{},
                       source_data,
                       dst_host.get_mutable_data(),
                       dal::detail::make_data_type<Src>(),
                       dal::detail::make_data_type<Dst>(),
                       stride,
                       stride,
                       get_element_count());

        for (std::int64_t i = 0; i < element_count_ / stride; i++) {
            REQUIRE(dst_host.get(i * stride) == Dst(source_data[i * stride]));
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Src, typename Dst>
    void test_device2device_conversion(std::int64_t stride = 1) {
        auto& q = get_queue();

        const Src* source_data = get_data(Src{});
        auto src_device = la::matrix<Src>::wrap(source_data, { 1, element_count_ }).to_device(q);
        auto dst_device = la::matrix<Dst>::empty_device(q, { 1, element_count_ });

        convert_vector(q,
                       src_device.get_data(),
                       dst_device.get_mutable_data(),
                       dal::detail::make_data_type<Src>(),
                       dal::detail::make_data_type<Dst>(),
                       stride,
                       stride,
                       get_element_count());

        auto dst_host = dst_device.to_host(q);
        for (std::int64_t i = 0; i < element_count_ / stride; i++) {
            REQUIRE(dst_host.get(i * stride) == Dst(source_data[i * stride]));
        }
    }

    template <typename Src, typename Dst>
    void test_host2device_conversion() {
        auto& q = get_queue();

        const Src* source_data = get_data(Src{});
        auto dst_device = la::matrix<Dst>::empty_device(q, { 1, element_count_ });

        convert_vector(q,
                       source_data,
                       dst_device.get_mutable_data(),
                       dal::detail::make_data_type<Src>(),
                       dal::detail::make_data_type<Dst>(),
                       1,
                       1,
                       get_element_count());

        auto dst_host = dst_device.to_host(q);
        for (std::int64_t i = 0; i < element_count_; i++) {
            REQUIRE(dst_host.get(i) == Dst(source_data[i]));
        }
    }

    template <typename Src, typename Dst>
    void test_device2host_conversion() {
        auto& q = get_queue();

        const Src* source_data = get_data(Src{});
        auto src_device = la::matrix<Src>::wrap(source_data, { 1, element_count_ }).to_device(q);
        auto dst_host = la::matrix<Dst>::empty({ 1, element_count_ });

        convert_vector(q,
                       src_device.get_data(),
                       dst_host.get_mutable_data(),
                       dal::detail::make_data_type<Src>(),
                       dal::detail::make_data_type<Dst>(),
                       1,
                       1,
                       get_element_count());

        for (std::int64_t i = 0; i < element_count_; i++) {
            REQUIRE(dst_host.get(i) == Dst(source_data[i]));
        }
    }
#endif

private:
    static constexpr std::int64_t element_count_ = 6;
    std::array<std::int32_t, element_count_> int_data_ = { 1, -2, //
                                                           3, -4, //
                                                           5, -6 };
    std::array<float, element_count_> float_data_ = { 1.6f, -2.5f, //
                                                      3.4f, -4.3f, //
                                                      5.2f, -6.1f };
    std::array<double, element_count_> double_data_ = { 1.6, -2.5, //
                                                        3.4, -4.3, //
                                                        5.2, -6.1 };
};

TEST_M(convert_test,
       "host2host convert identical types without stride",
       "[host2host][device-usm][no-stride]") {
    SECTION("int -> int") {
        test_host2host_conversion<std::int32_t, std::int32_t>();
    }
    SECTION("float -> float") {
        test_host2host_conversion<float, float>();
    }
    SECTION("double -> double") {
        test_host2host_conversion<double, double>();
    }
}

#ifdef ONEDAL_DATA_PARALLEL
TEST_M(convert_test,
       "device2device convert identical types without stride",
       "[device2device][device-usm][no-stride]") {
    SECTION("int -> int") {
        test_device2device_conversion<std::int32_t, std::int32_t>();
    }
    SECTION("float -> float") {
        test_device2device_conversion<float, float>();
    }
    SECTION("double -> double") {
        test_device2device_conversion<double, double>();
    }
}

TEST_M(convert_test,
       "convert types of the same size device2device without stride",
       "[device2device][device-usm][no-stride]") {
    SECTION("float -> std::int32_t") {
        test_device2device_conversion<float, std::int32_t>();
    }
    SECTION("std::int32_t -> float") {
        test_device2device_conversion<float, std::int32_t>();
    }
}

TEST_M(convert_test,
       "convert types of the different sizes device2device without stride",
       "[device2device][device-usm][no-stride]") {
    SECTION("float -> double") {
        test_device2device_conversion<float, double>();
    }
    SECTION("double -> float") {
        test_device2device_conversion<double, float>();
    }
    SECTION("std::int32_t -> double") {
        test_device2device_conversion<std::int32_t, double>();
    }
    SECTION("double -> std::int32_t") {
        test_device2device_conversion<double, std::int32_t>();
    }
}

TEST_M(convert_test,
       "host2device convert identical types without stride",
       "[host2device][device-usm][no-stride]") {
    SECTION("int -> int") {
        test_host2device_conversion<std::int32_t, std::int32_t>();
    }
    SECTION("float -> float") {
        test_host2device_conversion<float, float>();
    }
    SECTION("double -> double") {
        test_host2device_conversion<double, double>();
    }
}

TEST_M(convert_test,
       "convert types of the same size host2device without stride",
       "[host2device][device-usm][no-stride]") {
    SECTION("float -> std::int32_t") {
        test_host2device_conversion<float, std::int32_t>();
    }
    SECTION("std::int32_t -> float") {
        test_host2device_conversion<float, std::int32_t>();
    }
}

TEST_M(convert_test,
       "convert types of the different sizes host2device without stride",
       "[host2device][device-usm][no-stride]") {
    SECTION("float -> double") {
        test_host2device_conversion<float, double>();
    }
    SECTION("double -> float") {
        test_host2device_conversion<double, float>();
    }
    SECTION("std::int32_t -> double") {
        test_host2device_conversion<std::int32_t, double>();
    }
    SECTION("double -> std::int32_t") {
        test_host2device_conversion<double, std::int32_t>();
    }
}

TEST_M(convert_test,
       "device2host convert identical types without stride",
       "[device2host][device-usm][no-stride]") {
    SECTION("int -> int") {
        test_device2host_conversion<std::int32_t, std::int32_t>();
    }
    SECTION("float -> float") {
        test_device2host_conversion<float, float>();
    }
    SECTION("double -> double") {
        test_device2host_conversion<double, double>();
    }
}

TEST_M(convert_test,
       "convert types of the same size device2host without stride",
       "[device2host][device-usm][no-stride]") {
    SECTION("float -> std::int32_t") {
        test_device2host_conversion<float, std::int32_t>();
    }
    SECTION("std::int32_t -> float") {
        test_device2host_conversion<float, std::int32_t>();
    }
}

TEST_M(convert_test,
       "convert types of the different sizes device2host without stride",
       "[device2host][device-usm][no-stride]") {
    SECTION("float -> double") {
        test_device2host_conversion<float, double>();
    }
    SECTION("double -> float") {
        test_device2host_conversion<double, float>();
    }
    SECTION("std::int32_t -> double") {
        test_device2host_conversion<std::int32_t, double>();
    }
    SECTION("double -> std::int32_t") {
        test_device2host_conversion<double, std::int32_t>();
    }
}
#endif

} // namespace oneapi::dal::backend::test
