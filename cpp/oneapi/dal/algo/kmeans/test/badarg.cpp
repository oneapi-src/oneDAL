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

#include "oneapi/dal/algo/kmeans/infer.hpp"
#include "oneapi/dal/algo/kmeans/train.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::kmeans::test {

namespace te = dal::test::engine;

template <typename Method>
class kmeans_badarg_test : public te::algo_fixture {
public:
    static constexpr std::int64_t row_count = 8;
    static constexpr std::int64_t column_count = 2;
    static constexpr std::int64_t element_count = row_count * column_count;
    static constexpr std::int64_t cluster_count = 2;
    static constexpr std::int64_t invalid_cluster_count = 5;
    static constexpr std::int64_t bad_cluster_count = 2;
    static constexpr std::int64_t too_big_cluster_count = 10;
    static constexpr std::int64_t bad_column_count = 3;
    static constexpr std::int64_t cluster_element_count = cluster_count * column_count;
    static constexpr std::int64_t too_big_cluster_element_count =
        too_big_cluster_count * column_count;
    static constexpr std::int64_t bad_cluster_element_count = cluster_count * bad_column_count;

    auto get_descriptor() const {
        return kmeans::descriptor<float, Method, kmeans::task::clustering>{};
    }

    table get_train_data(std::int64_t override_row_count = row_count,
                         std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count);
        return homogen_table::wrap(train_data_.data(), override_row_count, override_column_count);
    }

    table get_infer_data(std::int64_t override_row_count = row_count,
                         std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count);
        return homogen_table::wrap(infer_data_.data(), override_row_count, override_column_count);
    }

    table get_initial_centroids(std::int64_t override_row_count = cluster_count,
                                std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count);
        return homogen_table::wrap(initial_centroids.data(),
                                   override_row_count,
                                   override_column_count);
    }

    table get_too_big_initial_centroids(std::int64_t override_row_count = too_big_cluster_count,
                                        std::int64_t override_column_count = column_count) const {
        return homogen_table::wrap(too_big_initial_centroids.data(),
                                   override_row_count,
                                   override_column_count);
    }

    table get_bad_initial_centroids(std::int64_t override_row_count = cluster_count,
                                    std::int64_t override_column_count = bad_column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count);
        return homogen_table::wrap(bad_initial_centroids.data(),
                                   override_row_count,
                                   override_column_count);
    }

private:
    static constexpr std::array<float, element_count> train_data_ = {
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0
    };

    static constexpr std::array<float, element_count> infer_data_ = {
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0
    };

    static constexpr std::array<float, bad_cluster_element_count> bad_initial_centroids = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };

    static constexpr std::array<float, cluster_element_count> initial_centroids = { 0.0,
                                                                                    0.0,
                                                                                    0.0,
                                                                                    0.0 };

    static constexpr std::array<float, too_big_cluster_element_count> too_big_initial_centroids = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
};

#define KMEANS_BADARG_TEST(name) \
    TEMPLATE_TEST_M(kmeans_badarg_test, name, "[kmeans][badarg]", method::lloyd_dense)

KMEANS_BADARG_TEST("accepts positive cluster_count") {
    REQUIRE_NOTHROW(this->get_descriptor().set_cluster_count(1));
}

KMEANS_BADARG_TEST("throws if cluster_count is negative") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_cluster_count(-1), domain_error);
}

KMEANS_BADARG_TEST("throws if cluster_count is zero") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_cluster_count(0), domain_error);
}

KMEANS_BADARG_TEST("throws if cluster_count overflow") {
    constexpr std::int64_t huge_cluster_count =
        dal::detail::limits<std::int32_t>::max() + std::int64_t(1);
    REQUIRE_THROWS_AS(this->get_descriptor().set_cluster_count(huge_cluster_count), domain_error);
}

KMEANS_BADARG_TEST("accepts positive max iteration count") {
    REQUIRE_NOTHROW(this->get_descriptor().set_max_iteration_count(1));
}

KMEANS_BADARG_TEST("accepts zero max iteration count") {
    REQUIRE_NOTHROW(this->get_descriptor().set_max_iteration_count(0));
}

KMEANS_BADARG_TEST("throws if max iteration count is negative") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_max_iteration_count(-1), domain_error);
}

KMEANS_BADARG_TEST("accepts positive accuracy threshold") {
    REQUIRE_NOTHROW(this->get_descriptor().set_accuracy_threshold(0.01));
}

KMEANS_BADARG_TEST("accepts zero accuracy threshold") {
    REQUIRE_NOTHROW(this->get_descriptor().set_accuracy_threshold(0.0));
}

KMEANS_BADARG_TEST("throws if accuracy threshold is negative") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_accuracy_threshold(-0.1), domain_error);
}

KMEANS_BADARG_TEST("throws if train data is empty") {
    const auto kmeans_desc = this->get_descriptor().set_cluster_count(2);

    REQUIRE_THROWS_AS(this->train(kmeans_desc, homogen_table{}), domain_error);
}

KMEANS_BADARG_TEST("throws if initial centroids rows less than cluster count") {
    const auto kmeans_desc = this->get_descriptor().set_cluster_count(this->invalid_cluster_count);

    REQUIRE_THROWS_AS(train(kmeans_desc, this->get_train_data(), this->get_initial_centroids()),
                      invalid_argument);
}

KMEANS_BADARG_TEST("throws if train data columns neq initial centroid columns") {
    const auto kmeans_desc = this->get_descriptor().set_cluster_count(this->cluster_count);

    REQUIRE_THROWS_AS(train(kmeans_desc, this->get_train_data(), this->get_bad_initial_centroids()),
                      invalid_argument);
}

KMEANS_BADARG_TEST("throws if cluster count exceeds data row count") {
    const auto kmeans_desc = this->get_descriptor().set_cluster_count(this->too_big_cluster_count);

    REQUIRE_THROWS_AS(
        train(kmeans_desc, this->get_train_data(), this->get_too_big_initial_centroids()),
        invalid_argument);
}

KMEANS_BADARG_TEST("throws if infer data is empty") {
    const auto kmeans_desc = this->get_descriptor().set_cluster_count(this->cluster_count);
    const auto model =
        train(kmeans_desc, this->get_train_data(), this->get_initial_centroids()).get_model();

    REQUIRE_THROWS_AS(infer(kmeans_desc, model, homogen_table{}), domain_error);
}

KMEANS_BADARG_TEST("throws if objective function is not available") {
    const auto kmeans_desc = this->get_descriptor().set_cluster_count(this->cluster_count);

    const auto result =
        train(kmeans_desc, this->get_train_data(), this->get_initial_centroids()).get_model();

    const auto kmeans_desc_infer =
        this->get_descriptor()
            .set_cluster_count(this->cluster_count)
            .set_result_options(kmeans::result_options::compute_assignments);

    const auto model = infer(kmeans_desc_infer, result, this->get_train_data());
    REQUIRE_NOTHROW(model.get_responses());
    REQUIRE_THROWS_AS(model.get_objective_function_value(), domain_error);
}

KMEANS_BADARG_TEST("throws if all metrics are available") {
    const auto kmeans_desc = this->get_descriptor().set_cluster_count(this->cluster_count);

    const auto result =
        train(kmeans_desc, this->get_train_data(), this->get_initial_centroids()).get_model();

    const auto model = infer(kmeans_desc, result, this->get_train_data());

    REQUIRE_NOTHROW(model.get_objective_function_value());
    REQUIRE_NOTHROW(model.get_responses());
}

} // namespace oneapi::dal::kmeans::test
