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

#include <CL/sycl.hpp>

#include "gtest/gtest.h"
#include "oneapi/dal/algo/knn/infer.hpp"
#include "oneapi/dal/algo/knn/train.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi;
namespace nn = oneapi::dal::knn;

TEST(knn_brute_force_bad_arg_tests, throws_if_x_test_table_is_empty) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);
    dal::homogen_table x_test_table;

    const auto knn_desc =
        nn::descriptor<float, nn::method::brute_force, nn::task::classification>{ 2, 1 };
    const auto result_train = dal::train(queue, knn_desc, x_train_table, y_train_table);

    ASSERT_THROW(dal::infer(queue, knn_desc, x_test_table, result_train.get_model()),
                 dal::domain_error);
}

TEST(knn_brute_force_unit_tests, train_no_throw) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    const auto knn_desc =
        nn::descriptor<float, nn::method::brute_force, nn::task::classification>{ 2, 1 };
    ASSERT_NO_THROW(dal::train(queue, knn_desc, x_train_table, y_train_table));
}

TEST(knn_brute_force_unit_tests, train_and_infer_no_throw) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    const auto knn_desc =
        nn::descriptor<float, nn::method::brute_force, nn::task::classification>{ 2, 1 };
    const auto result_train = dal::train(queue, knn_desc, x_train_table, y_train_table);

    ASSERT_NO_THROW(dal::infer(queue, knn_desc, x_train_table, result_train.get_model()));
}
