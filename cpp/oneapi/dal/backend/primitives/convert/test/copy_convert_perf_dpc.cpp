/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include <chrono>
#include <array>
#include <cmath>
#include <type_traits>

#include "oneapi/dal/detail/array_utils.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/convert/copy_convert.hpp"

#ifdef _MSC_VER
    #define PRETTY_FUNCTION __FUNCSIG__
#endif // _MSC_VER

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using convert_types = std::tuple<std::tuple<float, std::tuple<float, float, float, float, float>>,
                                 std::tuple<float, std::tuple<float, std::uint32_t, std::int64_t>>,
                                 std::tuple<std::int32_t, std::tuple<std::int8_t, double, std::uint64_t>>,
                                 std::tuple<double, std::tuple<std::int16_t, float, double, std::uint8_t>>,
                                 std::tuple<std::int8_t, std::tuple<std::int8_t, float, std::int8_t, float>>>;

template <typename... Types>
constexpr auto make_types_array(const std::tuple<Types...>*) {
    constexpr std::size_t count = sizeof...(Types);
    return std::array<data_type, count>{  detail::make_data_type<Types>()... };
}

template <typename... Types>
constexpr auto get_col_size(const std::tuple<Types...>*) {
    return (sizeof(Types) + ...);
}

template <typename Param>
class copy_convert_dpc_test : public te::policy_fixture {
public:
    using result_t = std::tuple_element_t<0, Param>;
    using sources_t = std::tuple_element_t<1, Param>;

    constexpr static inline const sources_t* dummy_ptr = nullptr;
    constexpr static inline auto data_types = make_types_array(dummy_ptr);
    constexpr static inline std::int64_t col_size = get_col_size(dummy_ptr);
    constexpr static inline std::int64_t row_count = std::tuple_size_v<sources_t>;
    constexpr static inline auto result_type = detail::make_data_type<result_t>();

    bool is_initialized() const {
        return col_count > 0l;
    }

    void check_if_initialized() const {
        if (!is_initialized()) {
            throw std::runtime_error{ "convert test is not initialized" };
        }
    }

    inline auto get_host_policy() const {
        return detail::default_host_policy{};
    }

    inline auto get_device_policy() {
        auto& queue = this->get_queue();
        return detail::data_parallel_policy{ queue };
    }

    void generate_input(std::int64_t seed = 777) {
        std::minstd_rand generator(seed);
        std::uniform_int_distribution<int> dist(0, 127);

        const auto inp_size = col_size * col_count;
        const auto gtr_size = row_count * col_count;

        inp = dal::array<dal::byte_t>::empty(inp_size);
        gtr = dal::array<result_t>::empty(gtr_size);

        std::int64_t inp_offset = 0l, gtr_offset = 0l;
        for (std::int64_t row = 0l; row < row_count; ++row) {
            const data_type dtype = data_types.at(row);
            const auto type_size = detail::get_data_type_size(dtype);

            auto* inp_raw = inp.get_mutable_data() + inp_offset;
            auto* gtr_ptr = gtr.get_mutable_data() + gtr_offset;

            backend::dispatch_by_data_type(dtype,
            [&](auto type) -> void {
                using type_t = std::remove_cv_t<decltype(type)>;
                auto* inp_ptr = reinterpret_cast<type_t*>(inp_raw);
                for(std::int64_t col = 0l; col < col_count; ++col) {
                    const int value = dist(generator);
                    inp_ptr[col] = static_cast<type_t>(value);
                    gtr_ptr[col] = static_cast<result_t>(inp_ptr[col]);
                }
            });

            inp_offset += (type_size * col_count);
            gtr_offset += col_count;
        }

        REQUIRE(inp_offset == inp_size);
        REQUIRE(gtr_offset == gtr_size);

        auto dev_policy = get_device_policy();
        dev = detail::copy(dev_policy, inp);
    }

    void generate() {
        col_count = GENERATE(100'000'001);
        CAPTURE(col_count, row_count);
        generate_input();
    }

    dal::array<data_type> get_types_array() const {
        auto result = dal::array<data_type>::empty(col_count);
        data_type* const res_ptr = result.get_mutable_data();
        std::copy(data_types.cbegin(), data_types.cend(), res_ptr);
        return result;
    }

    void test_copy_convert_rm() {
        auto& queue = this->get_queue();
        auto policy = get_device_policy();

        const auto res_size = col_count * row_count * sizeof(result_t);
        auto result = dal::array<dal::byte_t>::empty(queue, res_size);
        dal::array<data_type> types = get_types_array();

        BENCHMARK(__PRETTY_FUNCTION__) {
            copy_convert(policy, types, dev, { row_count, col_count },
                result_type, result, { col_count, 1l}).wait_and_throw();
        };
    }

    void test_copy_convert_cm() {
        auto& queue = this->get_queue();
        auto policy = get_device_policy();

        const auto res_size = col_count * row_count * sizeof(result_t);
        auto result = dal::array<dal::byte_t>::empty(queue, res_size);
        dal::array<data_type> types = get_types_array();

        BENCHMARK(__PRETTY_FUNCTION__) {
            copy_convert(policy, types, dev, { row_count, col_count },
                result_type, result, { 1l, row_count }).wait_and_throw();
        };
    }

private:
    std::int64_t col_count = 0l;
    dal::array<dal::byte_t> dev;
    dal::array<dal::byte_t> inp;
    dal::array<result_t> gtr;
};

TEMPLATE_LIST_TEST_M(copy_convert_dpc_test,
                     "Determenistic random array",
                     "[convert][2d][small]",
                     convert_types) {
    this->generate();
    this->test_copy_convert_rm();
    this->test_copy_convert_cm();
}

} // namespace oneapi::dal::backend::primitives::test
