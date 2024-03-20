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

#include "oneapi/dal/table/backend/convert/copy_convert.hpp"
#include "oneapi/dal/table/backend/test/copy_convert_fixture.hpp"

#ifdef _MSC_VER
#define PRETTY_FUNCTION __FUNCSIG__
#endif // _MSC_VER

namespace oneapi::dal::backend::primitives::test {

template <typename Param>
class copy_convert_cpu_test : public copy_convert_fixture<Param> {
public:
    using result_t = std::tuple_element_t<0, Param>;
    using sources_t = std::tuple_element_t<1, Param>;

    constexpr static inline const sources_t* dummy_ptr = nullptr;
    constexpr static inline auto data_types = make_types_array(dummy_ptr);
    constexpr static inline std::int64_t col_size = get_col_size(dummy_ptr);
    constexpr static inline std::int64_t row_count = std::tuple_size_v<sources_t>;
    constexpr static inline auto result_type = detail::make_data_type<result_t>();

    void generate() {
        this->col_count = GENERATE(100'000'001);
        CAPTURE(this->col_count, row_count);
        this->generate_input();
    }

    void test_copy_convert_rm() {
        auto policy = this->get_host_policy();

        const auto res_count = this->col_count * row_count;
        const auto res_size = res_count * sizeof(result_t);
        auto result = dal::array<dal::byte_t>::empty(res_size);
        dal::array<data_type> types = this->get_types_array();

        BENCHMARK(__PRETTY_FUNCTION__) {
            copy_convert(policy,
                         types,
                         this->inp,
                         { row_count, this->col_count },
                         result_type,
                         result,
                         { this->col_count, 1l });
        };
    }

    void test_copy_convert_cm() {
        auto policy = this->get_host_policy();

        const auto res_count = this->col_count * row_count;
        const auto res_size = res_count * sizeof(result_t);
        auto result = dal::array<dal::byte_t>::empty(res_size);
        dal::array<data_type> types = this->get_types_array();

        BENCHMARK(__PRETTY_FUNCTION__) {
            copy_convert(policy,
                         types,
                         this->inp,
                         { row_count, this->col_count },
                         result_type,
                         result,
                         { 1l, row_count });
        };
    }
};

#ifdef ONEDAL_DATA_PARALLEL

template <typename Param>
class copy_convert_dpc_test : public copy_convert_fixture<Param> {
public:
    using result_t = std::tuple_element_t<0, Param>;
    using sources_t = std::tuple_element_t<1, Param>;

    constexpr static inline const sources_t* dummy_ptr = nullptr;
    constexpr static inline auto data_types = make_types_array(dummy_ptr);
    constexpr static inline std::int64_t col_size = get_col_size(dummy_ptr);
    constexpr static inline std::int64_t row_count = std::tuple_size_v<sources_t>;
    constexpr static inline auto result_type = detail::make_data_type<result_t>();

    void generate() {
        this->col_count = GENERATE(100'000'001);
        CAPTURE(this->col_count, row_count);
        this->generate_input();
    }

    void test_copy_convert_rm() {
        auto& queue = this->get_queue();
        auto policy = this->get_device_policy();

        const auto res_count = this->col_count * row_count;
        const auto res_size = res_count * sizeof(result_t);
        auto result = dal::array<dal::byte_t>::empty(queue, res_size);
        dal::array<data_type> types = this->get_types_array();

        BENCHMARK(__PRETTY_FUNCTION__) {
            copy_convert(policy,
                         types,
                         this->dev,
                         { row_count, this->col_count },
                         result_type,
                         result,
                         { this->col_count, 1l })
                .wait_and_throw();
        };
    }

    void test_copy_convert_cm() {
        auto& queue = this->get_queue();
        auto policy = this->get_device_policy();

        const auto res_count = this->col_count * row_count;
        const auto res_size = res_count * sizeof(result_t);
        auto result = dal::array<dal::byte_t>::empty(queue, res_size);
        dal::array<data_type> types = this->get_types_array();

        BENCHMARK(__PRETTY_FUNCTION__) {
            copy_convert(policy,
                         types,
                         this->dev,
                         { row_count, this->col_count },
                         result_type,
                         result,
                         { 1l, row_count })
                .wait_and_throw();
        };
    }
};
#endif

TEMPLATE_LIST_TEST_M(copy_convert_cpu_test,
                     "Deterministic random array",
                     "[convert][2d][small]",
                     convert_types) {
    this->generate();
    this->test_copy_convert_rm();
    this->test_copy_convert_cm();
}

#ifdef ONEDAL_DATA_PARALLEL
TEMPLATE_LIST_TEST_M(copy_convert_dpc_test,
                     "Deterministic random array",
                     "[convert][2d][small]",
                     convert_types) {
    this->generate();
    this->test_copy_convert_rm();
    this->test_copy_convert_cm();
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives::test
