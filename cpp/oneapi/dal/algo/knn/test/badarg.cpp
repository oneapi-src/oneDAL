/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <array>

#include "oneapi/dal/algo/knn/infer.hpp"
#include "oneapi/dal/algo/knn/train.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::knn::test {

namespace te = dal::test::engine;

template <typename TestType>
class knn_badarg_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    static constexpr std::int64_t class_count = 2;
    static constexpr std::int64_t neighbor_count = 3;

    static constexpr std::int64_t column_count = 2;
    static constexpr std::int64_t train_row_count = 8;
    static constexpr std::int64_t train_element_count = column_count * train_row_count;
    static constexpr std::int64_t infer_row_count = 5;
    static constexpr std::int64_t infer_element_count = column_count * infer_row_count;

    static constexpr bool is_kd_tree = std::is_same_v<Method, knn::method::kd_tree>;
    static constexpr bool is_brute_force = std::is_same_v<Method, knn::method::brute_force>;

    bool not_available_on_device() {
        return (get_policy().is_gpu() && is_kd_tree);
    }

    auto get_descriptor(std::int64_t override_class_count = class_count,
                        std::int64_t override_neighbor_count = neighbor_count) const {
        return knn::descriptor<Float, Method, knn::task::classification>(override_class_count,
                                                                         override_neighbor_count);
    }

    table get_train_data(std::int64_t override_row_count = train_row_count,
                         std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= train_element_count);
        return homogen_table::wrap(train_data_.data(), override_row_count, override_column_count);
    }

    table get_train_responses(std::int64_t override_row_count = train_row_count,
                              std::int64_t override_column_count = 1) const {
        ONEDAL_ASSERT(override_row_count <= train_row_count);
        return homogen_table::wrap(train_responses_.data(),
                                   override_row_count,
                                   override_column_count);
    }

    table get_infer_data(std::int64_t override_row_count = infer_row_count,
                         std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= infer_element_count);
        return homogen_table::wrap(infer_data_.data(), override_row_count, override_column_count);
    }

private:
    static constexpr std::array<Float, train_element_count> train_data_ = {
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0
    };

    static constexpr std::array<Float, train_row_count> train_responses_ = { 0.0, 0.0, 0.0, 0.0,
                                                                             1.0, 1.0, 1.0, 1.0 };

    static constexpr std::array<Float, infer_element_count> infer_data_ = { 1.0,  1.0,  2.0,  2.0,
                                                                            -1.0, -1.0, -1.0, -2.0,
                                                                            -2.0, -1.0 };
};

using knn_types = COMBINE_TYPES((float), (knn::method::brute_force, knn::method::kd_tree));

#define KNN_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(knn_badarg_test, name, "[knn][badarg]", knn_types)

KNN_BADARG_TEST("accepts positive class_count in constructor") {
    REQUIRE_NOTHROW(this->get_descriptor(this->class_count, this->neighbor_count));
}

KNN_BADARG_TEST("accepts class_count more than one in set_class_count") {
    REQUIRE_NOTHROW(this->get_descriptor().set_class_count(this->class_count));
}

KNN_BADARG_TEST("throws if class_count is one in constructor") {
    REQUIRE_THROWS_AS(this->get_descriptor(1, this->neighbor_count), domain_error);
}

KNN_BADARG_TEST("throws if class_count is zero in constructor") {
    REQUIRE_THROWS_AS(this->get_descriptor(0, this->neighbor_count), domain_error);
}

KNN_BADARG_TEST("throws if class_count is zero in set_class_count") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_class_count(0), domain_error);
}

KNN_BADARG_TEST("throws if class_count is negative in constructor") {
    REQUIRE_THROWS_AS(this->get_descriptor(-1, this->neighbor_count), domain_error);
}

KNN_BADARG_TEST("throws if class_count is negative in set_class_count") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_class_count(-1), domain_error);
}

KNN_BADARG_TEST("accepts positive neighbor_count in set_class_count") {
    REQUIRE_NOTHROW(this->get_descriptor().set_neighbor_count(this->neighbor_count));
}

KNN_BADARG_TEST("throws if neighbor_count is zero in constructor") {
    REQUIRE_THROWS_AS(this->get_descriptor(this->class_count, 0), domain_error);
}

KNN_BADARG_TEST("throws if neighbor_count is zero in set_neighbor_count") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_neighbor_count(0), domain_error);
}

KNN_BADARG_TEST("throws if neighbor_count is negative in constructor") {
    REQUIRE_THROWS_AS(this->get_descriptor(this->class_count, -1), domain_error);
}

KNN_BADARG_TEST("throws if neighbor_count is negative in set_neighbor_count") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_neighbor_count(-1), domain_error);
}

KNN_BADARG_TEST("throws if class_count = 1 and neighbor_count = 0 in constructor") {
    REQUIRE_THROWS_AS(this->get_descriptor(1, 0), domain_error);
}

KNN_BADARG_TEST("throws if both class_count and neighbor_count are zero in constructor") {
    REQUIRE_THROWS_AS(this->get_descriptor(0, 0), domain_error);
}

KNN_BADARG_TEST("throws if both class_count and neighbor_count are negative in constructor") {
    REQUIRE_THROWS_AS(this->get_descriptor(-1, -1), domain_error);
}

KNN_BADARG_TEST("accepts train data and responses") {
    SKIP_IF(this->not_available_on_device());
    const auto knn_desc = this->get_descriptor();
    REQUIRE_NOTHROW(this->train(knn_desc, this->get_train_data(), this->get_train_responses()));
}

KNN_BADARG_TEST("throws if both train data and responses are empty") {
    SKIP_IF(this->not_available_on_device());
    const auto knn_desc = this->get_descriptor();
    REQUIRE_THROWS_AS(this->train(knn_desc, homogen_table{}, homogen_table{}), domain_error);
}

KNN_BADARG_TEST("throws if train responses are empty") {
    SKIP_IF(this->not_available_on_device());
    const auto knn_desc = this->get_descriptor();
    REQUIRE_THROWS_AS(this->train(knn_desc, this->get_train_data(), homogen_table{}), domain_error);
}

KNN_BADARG_TEST("throws if train data is empty") {
    SKIP_IF(this->not_available_on_device());
    const auto knn_desc = this->get_descriptor();
    REQUIRE_THROWS_AS(this->train(knn_desc, homogen_table{}, this->get_train_responses()),
                      domain_error);
}

KNN_BADARG_TEST(
    "throws if the number of train samples is not equal to the number of train responses") {
    SKIP_IF(this->not_available_on_device());
    const auto knn_desc = this->get_descriptor();
    REQUIRE_THROWS_AS(this->train(knn_desc, this->get_train_data(3), this->get_train_responses()),
                      domain_error);
    REQUIRE_THROWS_AS(this->train(knn_desc, this->get_train_data(), this->get_train_responses(3)),
                      domain_error);
}

KNN_BADARG_TEST("throws if the number of columns in responses is greater then one") {
    SKIP_IF(this->not_available_on_device());
    const auto knn_desc = this->get_descriptor();
    REQUIRE_THROWS_AS(
        this->train(knn_desc, this->get_train_data(3), this->get_train_responses(3, 2)),
        domain_error);
}

KNN_BADARG_TEST("accept if infer data has suitable shape and not empty") {
    SKIP_IF(this->not_available_on_device());
    const auto knn_desc = this->get_descriptor();
    const auto train_result =
        this->train(knn_desc, this->get_train_data(), this->get_train_responses());
    const auto infer_data = this->get_infer_data();
    REQUIRE_NOTHROW(this->infer(knn_desc, infer_data, train_result.get_model()));
}

//TODO: Model class should be extended to provide information about task dimensionality
/*KNN_BADARG_TEST("throws if the number of infer data columns exceeds the number of columns in train") {
    SKIP_IF(this->not_available_on_device());
    const auto knn_desc = this->get_descriptor();
    const auto train_result = this->train(knn_desc, this->get_train_data(), this->get_train_responses());
    const auto infer_data = this->get_infer_data(2, 3);
    REQUIRE_THROWS_AS(this->infer(knn_desc, infer_data, train_result.get_model()), invalid_argument);
}

KNN_BADARG_TEST("throws if the number of infer data columns is less then the number of columns in train") {
    SKIP_IF(this->not_available_on_device());
    const auto knn_desc = this->get_descriptor();
    const auto train_result = this->train(knn_desc, this->get_train_data(), this->get_train_responses());
    const auto infer_data = this->get_infer_data(7, 1);
    REQUIRE_THROWS_AS(this->infer(knn_desc, infer_data, train_result.get_model()), invalid_argument);
}*/

KNN_BADARG_TEST("throws if infer data is empty") {
    SKIP_IF(this->not_available_on_device());
    const auto knn_desc = this->get_descriptor();
    const auto train_result =
        this->train(knn_desc, this->get_train_data(), this->get_train_responses());
    REQUIRE_THROWS_AS(this->infer(knn_desc, homogen_table{}, train_result.get_model()),
                      domain_error);
}

} // namespace oneapi::dal::knn::test
