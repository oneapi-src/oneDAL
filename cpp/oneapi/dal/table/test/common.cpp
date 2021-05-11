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

#include "oneapi/dal/table/common.hpp"
#include "gtest/gtest.h"

using namespace oneapi::dal;
using namespace oneapi;

TEST(table_test, can_construct_empty_table) {
    table t;

    ASSERT_FALSE(t.has_data());
    ASSERT_EQ(t.get_row_count(), 0);
    ASSERT_EQ(t.get_column_count(), 0);
}

TEST(simple_metadata_bad_arg_tests, get_feature_type) {
    const std::int64_t n_features = 5;
    auto dtypes = array<data_type>::full(n_features, data_type::int32);
    auto ftypes = array<feature_type>::full(n_features, feature_type::nominal);

    table_metadata md(dtypes, ftypes);

    ASSERT_THROW(md.get_feature_type(-1), dal::out_of_range);
    ASSERT_THROW(md.get_feature_type(n_features), dal::out_of_range);
    ASSERT_NO_THROW(md.get_feature_type(0));
}

TEST(simple_metadata_bad_arg_tests, get_data_type) {
    const std::int64_t n_features = 5;
    auto dtypes = array<data_type>::full(n_features, data_type::int32);
    auto ftypes = array<feature_type>::full(n_features, feature_type::nominal);

    table_metadata md(dtypes, ftypes);

    ASSERT_THROW(md.get_data_type(-1), dal::out_of_range);
    ASSERT_THROW(md.get_data_type(n_features), dal::out_of_range);
    ASSERT_NO_THROW(md.get_data_type(0));
}
