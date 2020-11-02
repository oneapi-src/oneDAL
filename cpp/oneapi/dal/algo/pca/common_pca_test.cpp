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
#include "oneapi/dal/algo/pca/infer.hpp"
#include "oneapi/dal/algo/pca/train.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi;
namespace pca = oneapi::dal::pca;

using pca_methods = testing::Types<pca::method::cov, pca::method::svd>;

template <typename Tuple>
class pca_common_bad_arg_tests : public ::testing::Test {};

TYPED_TEST_SUITE_P(pca_common_bad_arg_tests);

TYPED_TEST_P(pca_common_bad_arg_tests, test_set_component_count) {
    ASSERT_THROW(
        (pca::descriptor<float, TypeParam, pca::task::dim_reduction>().set_component_count(-1)),
        dal::domain_error);
    ASSERT_NO_THROW(
        (pca::descriptor<float, TypeParam, pca::task::dim_reduction>().set_component_count(0)));
}

TYPED_TEST_P(pca_common_bad_arg_tests, throws_if_train_data_table_is_empty) {
    const auto pca_desc =
        pca::descriptor<float, TypeParam, pca::task::dim_reduction>().set_component_count(2);

    ASSERT_THROW(train(pca_desc, dal::homogen_table()), dal::domain_error);
}

TYPED_TEST_P(pca_common_bad_arg_tests,
             throws_if_train_data_table_columns_less_than_component_count) {
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };

    const auto data_table = dal::homogen_table::wrap(data, row_count, column_count);

    const auto pca_desc =
        pca::descriptor<float, TypeParam, pca::task::dim_reduction>().set_component_count(4);

    ASSERT_THROW(train(pca_desc, data_table), dal::invalid_argument);
}

TYPED_TEST_P(pca_common_bad_arg_tests, throws_if_infer_data_table_is_empty) {
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t component_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };

    const auto data_table = dal::homogen_table::wrap(data, row_count, column_count);

    const auto pca_desc =
        pca::descriptor<float, TypeParam, pca::task::dim_reduction>().set_component_count(2);

    const auto result_train = train(pca_desc, data_table);
    ASSERT_THROW(infer(pca_desc, result_train.get_model(), dal::homogen_table()),
                 dal::domain_error);
}

TYPED_TEST_P(pca_common_bad_arg_tests, throws_if_component_count_ne_eigenvector_rows) {
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t component_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };

    const auto data_table = dal::homogen_table::wrap(data, row_count, column_count);

    const float data_infer[] = { 1.0, 1.0,  0.0, 1.0,  1.0,  0.0,  2.0, 2.0,  7.0,
                                 0.0, -1.0, 0.0, -5.0, -5.0, -5.0, 0.0, -2.0, 1.0 };
    const auto data_infer_table = dal::homogen_table::wrap(data_infer, row_count, column_count);

    auto pca_desc =
        pca::descriptor<float, TypeParam, pca::task::dim_reduction>().set_component_count(2);

    const auto result_train = train(pca_desc, data_table);

    pca_desc.set_component_count(4);

    ASSERT_THROW(infer(pca_desc, result_train.get_model(), data_infer_table),
                 dal::invalid_argument);
}

TYPED_TEST_P(pca_common_bad_arg_tests, throws_if_infer_data_column_count_ne_eigenvector_columns) {
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t component_count = 2;

    const float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };

    const auto data_table = dal::homogen_table::wrap(data, row_count, column_count);

    const float data_infer[] = { 1.0, 1.0,  0.0, 1.0,  1.0,  0.0,  2.0, 2.0,  7.0,
                                 0.0, -1.0, 0.0, -5.0, -5.0, -5.0, 0.0, -2.0, 1.0 };
    const auto data_infer_table = dal::homogen_table::wrap(data_infer, 4, 4);

    auto pca_desc =
        pca::descriptor<float, TypeParam, pca::task::dim_reduction>().set_component_count(2);

    const auto result_train = train(pca_desc, data_table);

    ASSERT_THROW(infer(pca_desc, result_train.get_model(), data_infer_table),
                 dal::invalid_argument);
}

REGISTER_TYPED_TEST_SUITE_P(pca_common_bad_arg_tests,
                            test_set_component_count,
                            throws_if_train_data_table_is_empty,
                            throws_if_train_data_table_columns_less_than_component_count,
                            throws_if_infer_data_table_is_empty,
                            throws_if_component_count_ne_eigenvector_rows,
                            throws_if_infer_data_column_count_ne_eigenvector_columns);

INSTANTIATE_TYPED_TEST_SUITE_P(run_pca_common_bad_arg_tests, pca_common_bad_arg_tests, pca_methods);
