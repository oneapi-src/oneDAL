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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;

TEST("serialize/deserialize empty table metadata") {
    table_metadata empty_meta;

    const auto deserialized = te::serialize_deserialize(empty_meta);

    REQUIRE(deserialized.get_feature_count() == 0);
}

TEST("serialize/deserialize simple table metadata") {
    constexpr std::int64_t feature_count = 5;
    const data_type dtypes[feature_count] = { data_type::int32,
                                              data_type::int32,
                                              data_type::float32,
                                              data_type::float64,
                                              data_type::uint8 };
    const feature_type ftypes[feature_count] = { feature_type::ordinal,
                                                 feature_type::ordinal,
                                                 feature_type::interval,
                                                 feature_type::ratio,
                                                 feature_type::nominal };
    table_metadata simple_meta(array<data_type>::wrap(dtypes, feature_count),
                               array<feature_type>::wrap(ftypes, feature_count));

    const auto deserialized = te::serialize_deserialize(simple_meta);

    REQUIRE(deserialized.get_feature_count() == feature_count);
    for (std::int64_t i = 0; i < feature_count; i++) {
        REQUIRE(deserialized.get_data_type(i) == dtypes[i]);
        REQUIRE(deserialized.get_feature_type(i) == ftypes[i]);
    }
}

TEST("serialize/deserialize empty table") {
    table empty_table;

    const auto deserialized = te::serialize_deserialize(empty_table);

    REQUIRE(deserialized.get_row_count() == 0);
    REQUIRE(deserialized.get_column_count() == 0);
    REQUIRE(deserialized.has_data() == false);
    REQUIRE(deserialized.get_kind() == empty_table.get_kind());
    REQUIRE(deserialized.get_metadata().get_feature_count() == 0);
}

} // namespace oneapi::dal::test
