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

#include "gtest/gtest.h"
#include "oneapi/dal/algo/kmeans_init/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi;
namespace kmn_init = oneapi::dal::kmeans_init;

using kmn_init_methods = testing::Types<kmn_init::method::dense,
                                        kmn_init::method::random_dense,
                                        kmn_init::method::plus_plus_dense,
                                        kmn_init::method::parallel_plus_dense>;

template <typename Tuple>
class kmeans_init_common_bad_arg_tests : public ::testing::Test {};

TYPED_TEST_SUITE_P(kmeans_init_common_bad_arg_tests);

TYPED_TEST_P(kmeans_init_common_bad_arg_tests, test_set_cluster_count) {
    ASSERT_THROW(
        (kmn_init::descriptor<float, TypeParam, kmn_init::task::init>().set_cluster_count(0)),
        dal::domain_error);
    ASSERT_THROW(
        (kmn_init::descriptor<float, TypeParam, kmn_init::task::init>().set_cluster_count(-1)),
        dal::domain_error);
    ASSERT_NO_THROW(
        (kmn_init::descriptor<float, TypeParam, kmn_init::task::init>().set_cluster_count(1)));
}

TYPED_TEST_P(kmeans_init_common_bad_arg_tests, throws_if_data_table_is_empty) {
    const auto kmeans_init_desc =
        kmn_init::descriptor<float, TypeParam, kmn_init::task::init>().set_cluster_count(2);

    ASSERT_THROW(compute(kmeans_init_desc, dal::homogen_table()), dal::domain_error);
}

REGISTER_TYPED_TEST_SUITE_P(kmeans_init_common_bad_arg_tests,
                            test_set_cluster_count,
                            throws_if_data_table_is_empty);

INSTANTIATE_TYPED_TEST_SUITE_P(run_kmeans_init_common_bad_arg_tests,
                               kmeans_init_common_bad_arg_tests,
                               kmn_init_methods);
