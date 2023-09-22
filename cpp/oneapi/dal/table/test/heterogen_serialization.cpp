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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/heterogen.hpp"

#include "oneapi/dal/table/detail/metadata_utils.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

using heterogen_types =
    std::tuple<std::tuple<std::uint8_t, std::int16_t, std::uint32_t, std::uint64_t>,
               std::tuple<std::uint64_t, std::int32_t, std::uint16_t, std::uint8_t>,
               std::tuple<float, std::int32_t, float>,
               std::tuple<std::int64_t, std::int8_t>,
               std::tuple<float>>;

class base_heterogen_table_serialization_test : public te::policy_fixture {
public:
    virtual void compare_tables(const heterogen_table& original, const table& deserialized) = 0;

    virtual void compare_tables(const heterogen_table& original,
                                const heterogen_table& deserialized) = 0;

    void check_table_serialization(const heterogen_table& original) {
        SECTION("deserialize as exact type") {
            heterogen_table deserialized;

            SECTION("serialize as exact type") {
                te::serialize_deserialize(original, deserialized);
            }

            SECTION("serialize as base type") {
                table original_as_base = original;
                te::serialize_deserialize(original_as_base, deserialized);
            }

            compare_tables(original, deserialized);
        }

        SECTION("deserialize as base type") {
            heterogen_table deserialized;

            SECTION("serialize as exact type") {
                te::serialize_deserialize(original, deserialized);
            }

            SECTION("serialize as base type") {
                table original_as_base = original;
                te::serialize_deserialize(original_as_base, deserialized);
            }

            compare_tables(original, deserialized);
        }
    }
};

class empty_heterogen_table_serialization_test : public base_heterogen_table_serialization_test {
public:
    void compare_tables(const heterogen_table& original, const table& deserialized) override {
        check_empty_base_table(deserialized);
    }

    void compare_tables(const heterogen_table& original,
                        const heterogen_table& deserialized) override {
        check_empty_heterogen_table(deserialized);
    }

    void check_empty_base_table(const table& deserialized) {
        REQUIRE(deserialized.has_data() == false);
        REQUIRE(deserialized.get_row_count() == 0);
        REQUIRE(deserialized.get_column_count() == 0);
        REQUIRE(deserialized.get_metadata().get_feature_count() == 0);
        REQUIRE(deserialized.get_data_layout() == data_layout::column_major);
    }

    void check_empty_heterogen_table(const heterogen_table& deserialized) {
        check_empty_base_table(deserialized);
        REQUIRE(deserialized.kind() == heterogen_table::kind());
    }
};

template <typename TypesTuple>
class heterogen_table_serialization_test : public base_heterogen_table_serialization_test {
public:
    constexpr static auto column_count = std::tuple_size_v<TypesTuple>;
    constexpr static const TypesTuple* dummy = nullptr;

    template <typename... Types>
    static auto get_metadata(const std::tuple<Types...>* = dummy) {
        auto dtypes = detail::find_array_dtypes<Types...>();
        auto ftypes = detail::find_array_ftypes<Types...>();
        return table_metadata(dtypes, ftypes);
    }

    template <typename F, typename... Types>
    heterogen_table fill_table(F&& generate, const std::tuple<Types...>* = dummy) {
        auto meta = get_metadata(dummy);
        auto table = heterogen_table::empty(meta);

        std::int64_t col = 0l;
        detail::apply(
            [&](const auto& type) -> void {
                const auto column = generate(type, col);
                table.set_column(col++, column);
            },
            Types{}...);
        REQUIRE(col == column_count);

        return table;
    }

    template <typename T>
    chunked_array<T> generate_random_host(std::int64_t count, int seed) {
        auto random = la::generate_uniform_matrix<T>({ count, 1l }, 0, 10, seed);
        return chunked_array<T>(random.get_array());
    }

    heterogen_table get_host_backed_table(std::int64_t row_count, int seed = 7777) {
        auto generate = [row_count, seed, this](auto type, std::int64_t col) {
            const std::int64_t sd = seed + col;
            using type_t = std::decay_t<decltype(type)>;
            return this->generate_random_host<type_t>(row_count, sd);
        };
        return this->fill_table(generate, dummy);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T>
    chunked_array<T> generate_random_device(std::int64_t count, int seed) {
        auto random = la::generate_uniform_matrix<T>({ count, 1l }, 0, 10, seed);
        auto random_device = random.to_device(this->get_queue()).get_array();
        return chunked_array<T>(random_device);
    }

    heterogen_table get_device_backed_table(std::int64_t row_count, int seed = 7777) {
        auto generate = [row_count, seed, this](auto type, std::int64_t col) {
            const std::int64_t sd = seed + col;
            using type_t = std::decay_t<decltype(type)>;
            return this->generate_random_device<type_t>(row_count, sd);
        };
        return this->fill_table(generate, dummy);
    }
#endif

    void compare_tables(const heterogen_table& original, const table& deserialized) override {
        te::check_if_tables_equal<float>(deserialized, original);
    }

    void compare_tables(const heterogen_table& original,
                        const heterogen_table& deserialized) override {
        te::check_if_tables_equal<float>(deserialized, original);
    }
};

TEST_CASE_METHOD(empty_heterogen_table_serialization_test,
                 "Empty heterogen table",
                 "[empty][heterogen]") {
    const heterogen_table empty_table;

    this->check_table_serialization(empty_table);
}

TEMPLATE_LIST_TEST_M(heterogen_table_serialization_test,
                     "Random heterogen table - device",
                     "[empty][heterogen][host]",
                     heterogen_types) {
    const std::int64_t row_count = GENERATE(1, 10, 1000);
    const std::int64_t column_count = GENERATE(1, 10, 100);
    const heterogen_table original = this->get_host_backed_table(row_count, column_count);

    this->check_table_serialization(original);
}

#ifdef ONEDAL_DATA_PARALLEL
TEMPLATE_LIST_TEST_M(heterogen_table_serialization_test,
                     "Random heterogen table - host",
                     "[empty][heterogen][device]",
                     heterogen_types) {
    const std::int64_t row_count = GENERATE(1, 10, 1000);
    const std::int64_t column_count = GENERATE(1, 10, 100);
    const heterogen_table original = this->get_device_backed_table(row_count, column_count);

    this->check_table_serialization(original);
}
#endif

} // namespace oneapi::dal::test
