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

#include "oneapi/dal/algo/knn/test/fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::knn::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace de = dal::detail;
namespace knn = oneapi::dal::knn;

template <typename TestType>
class knn_spmd_test : public knn_test<TestType, knn_spmd_test<TestType>> {
public:
    using base_t = knn_test<TestType, knn_spmd_test<TestType>>;
    using float_t = typename base_t::float_t;
    using method_t = typename base_t::method_t;
    using task_t = typename base_t::task_t;
    using descriptor_t = descriptor<float_t, method_t, task_t>;
    using train_result_t = typename base_t::train_result_t;
    using train_input_t = typename base_t::train_input_t;
    using infer_result_t = typename base_t::infer_result_t;
    using infer_input_t = typename base_t::infer_input_t;

    using default_distance_t = oneapi::dal::minkowski_distance::descriptor<>;

    using voting_t = oneapi::dal::knn::voting_mode;
    constexpr static inline voting_t default_voting = voting_t::uniform;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename Descriptor, typename... Args>
    infer_result_t train_and_infer_spmd(const Descriptor& knn_desc,
                                        const table& x_train_table,
                                        const table& y_train_table,
                                        const table& x_infer_table) {
        auto train_result = this->train_override(knn_desc, x_train_table, y_train_table);
        auto infer_result = this->infer(knn_desc, x_infer_table, train_result);
        return infer_result;
    }

    template <typename Descriptor, typename... Args>
    std::vector<train_result_t> train_override(const Descriptor& desc, Args&&... args) {
        return this->train_via_spmd_threads(rank_count_, desc, std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<train_input_t> split_train_input_override(std::int64_t split_count,
                                                          Args&&... args) {
        const train_input_t input{ std::forward<Args>(args)... };

        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);
        const auto split_responses = te::split_table_by_rows<float_t>(this->get_policy(),
                                                                      input.get_responses(),
                                                                      split_count);

        std::vector<train_input_t> split_input;
        split_input.reserve(split_count);

        for (std::int64_t i = 0; i < split_count; i++) {
            split_input.push_back( //
                train_input_t{ split_data[i], split_responses[i] });
        }

        return split_input;
    }

    template <typename... Args>
    infer_result_t infer_override(Args&&... args) {
        return this->infer_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<infer_input_t> split_infer_input_override(
        std::int64_t split_count,
        const table& x_infer_table,
        const std::vector<train_result_t>& results) {
        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), x_infer_table, split_count);

        std::vector<infer_input_t> split_input;
        split_input.reserve(split_count);

        for (std::int64_t i = 0; i < split_count; i++) {
            split_input.push_back( //
                infer_input_t{ split_data[i], results[i].get_model() });
        }

        return split_input;
    }

    infer_result_t merge_infer_result_override(const std::vector<infer_result_t>& results) {
        // Responses are distributed accross the ranks, we combine them into one table;
        std::vector<table> responses;
        for (const auto& r : results) {
            responses.push_back(r.get_responses());
        }
        const auto full_responses = te::stack_tables_by_rows<float_t>(responses);

        return infer_result_t{}
            .set_result_options(result_options::responses)
            .set_responses(full_responses);
    }

    template <typename Distance = default_distance_t>
    float_t distr_classification(const table& train_data,
                                 const table& train_responses,
                                 const table& infer_data,
                                 const table& infer_responses,
                                 const std::int64_t n_classes,
                                 const std::int64_t n_neighbors,
                                 const Distance distance = Distance{},
                                 const voting_t voting = default_voting,
                                 const float_t tolerance = float_t(1.e-5)) {
        INFO("check if data shape is expected")
        REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        REQUIRE(train_responses.get_column_count() == 1);
        REQUIRE(infer_responses.get_column_count() == 1);
        REQUIRE(infer_data.get_row_count() == infer_responses.get_row_count());
        REQUIRE(train_data.get_row_count() == train_responses.get_row_count());

        const auto knn_desc = this->get_descriptor(n_classes, n_neighbors, distance, voting);

        auto infer_result =
            this->train_and_infer_spmd(knn_desc, train_data, train_responses, infer_data);
        auto [prediction] = this->unpack_result(infer_result);

        const auto score_table =
            te::accuracy_score<float_t>(infer_responses, prediction, tolerance);
        const auto score = row_accessor<const float_t>(score_table).pull({ 0, -1 })[0];
        return score;
    }

    template <typename Distance = default_distance_t>
    float_t distr_regression(const table& train_data,
                             const table& train_responses,
                             const table& infer_data,
                             const table& infer_responses,
                             const std::int64_t n_neighbors,
                             const Distance distance = Distance{},
                             const voting_t voting = default_voting) {
        INFO("check if data shape is expected")
        REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        REQUIRE(train_responses.get_column_count() == 1);
        REQUIRE(infer_responses.get_column_count() == 1);
        REQUIRE(infer_data.get_row_count() == infer_responses.get_row_count());
        REQUIRE(train_data.get_row_count() == train_responses.get_row_count());

        constexpr knn::task::regression task{};

        const auto knn_desc = this->get_descriptor(42, n_neighbors, distance, voting, task);

        auto infer_result =
            this->train_and_infer_spmd(knn_desc, train_data, train_responses, infer_data);
        auto [prediction] = this->unpack_result(infer_result);

        const auto score_table = te::mse_score<float_t>(infer_responses, prediction);
        const auto score = row_accessor<const float_t>(score_table).pull({ 0, -1 })[0];
        return score;
    }

private:
    std::int64_t rank_count_ = 1;
};

#define KNN_SPMD_SMALL_TEST(name)                                         \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                   \
                         name,                                            \
                         "[small-dataset][knn][integration][spmd][test]", \
                         knn_cls_types)

#define KNN_SPMD_CLS_SYNTHETIC_TEST(name)                                     \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                       \
                         name,                                                \
                         "[synthetic-dataset][knn][integration][spmd][test]", \
                         knn_cls_types)

#define KNN_SPMD_CLS_EXTERNAL_TEST(name)                                     \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                      \
                         name,                                               \
                         "[external-dataset][knn][integration][spmd][test]", \
                         knn_cls_types)

#define KNN_SPMD_CLS_BF_EXTERNAL_TEST(name)                                  \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                      \
                         name,                                               \
                         "[external-dataset][knn][integration][spmd][test]", \
                         knn_cls_bf_types)

#define KNN_SPMD_REG_SYNTHETIC_TEST(name)                                     \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                       \
                         name,                                                \
                         "[synthetic-dataset][knn][integration][spmd][test]", \
                         knn_reg_types)

#define KNN_SPMD_REG_EXTERNAL_TEST(name)                                     \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                      \
                         name,                                               \
                         "[external-dataset][knn][integration][spmd][test]", \
                         knn_reg_types)

#define KNN_SPMD_REG_BF_EXTERNAL_TEST(name)                                  \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                      \
                         name,                                               \
                         "[external-dataset][knn][integration][spmd][test]", \
                         knn_reg_bf_types)

KNN_SPMD_SMALL_TEST("knn nearest points test predefined 9x9x1") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    this->set_rank_count(2);
    constexpr std::int64_t train_row_count = 9;
    constexpr std::int64_t infer_row_count = 9;
    constexpr std::int64_t column_count = 1;

    CAPTURE(train_row_count, infer_row_count, column_count);

    constexpr std::int64_t train_element_count = train_row_count * column_count;
    constexpr std::int64_t infer_element_count = infer_row_count * column_count;

    constexpr std::array<float, train_element_count> train = { -1.f, 0.f,  1.f,   3.f,   5.f,
                                                               10.f, 20.f, 100.f, 1000.f };

    constexpr std::array<float, infer_element_count> infer = { -10.f,  0.f,   0.6f,  0.6f, 40.f,
                                                               1000.f, 999.f, -0.4f, 0.6f };

    const auto x_train_table = homogen_table::wrap(train.data(), train_row_count, column_count);
    const auto x_infer_table = homogen_table::wrap(infer.data(), infer_row_count, column_count);
    const auto y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto infer_result =
        this->train_and_infer_spmd(knn_desc, x_train_table, y_train_table, x_infer_table);

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SPMD_CLS_SYNTHETIC_TEST("distributed knn nearest points test random uniform 513x301x17") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    this->set_rank_count(4);

    constexpr std::int64_t train_row_count = 513;
    constexpr std::int64_t infer_row_count = 301;
    constexpr std::int64_t column_count = 17;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.3, 1.));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto infer_result =
        this->train_and_infer_spmd(knn_desc, x_train_table, y_train_table, x_infer_table);

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SPMD_REG_SYNTHETIC_TEST(
    "distributed knn nearest points test random uniform using regression 513x301x17") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    this->set_rank_count(4);

    constexpr std::int64_t train_row_count = 513;
    constexpr std::int64_t infer_row_count = 301;
    constexpr std::int64_t column_count = 17;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.3, 1.));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    using distance_t = oneapi::dal::minkowski_distance::descriptor<>;
    constexpr double minkowski_degree = 2.0;
    const auto distance_desc = distance_t(minkowski_degree);
    constexpr auto voting = oneapi::dal::knn::voting_mode::uniform;
    constexpr knn::task::regression task{};

    const auto knn_desc = this->get_descriptor(train_row_count, 1, distance_desc, voting, task);

    auto infer_result =
        this->train_and_infer_spmd(knn_desc, x_train_table, y_train_table, x_infer_table);

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SPMD_CLS_SYNTHETIC_TEST("distributed knn nearest points test random uniform 17000x20x5") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    this->set_rank_count(2);

    constexpr std::int64_t train_row_count = 17000;
    constexpr std::int64_t infer_row_count = 20;
    constexpr std::int64_t column_count = 5;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.3, 1.));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto infer_result =
        this->train_and_infer_spmd(knn_desc, x_train_table, y_train_table, x_infer_table);

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SPMD_CLS_SYNTHETIC_TEST("distributed knn nearest points test random uniform 1000x34000x3") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    this->set_rank_count(2);

    constexpr std::int64_t train_row_count = 1000;
    constexpr std::int64_t infer_row_count = 34000;
    constexpr std::int64_t column_count = 3;

    CAPTURE(train_row_count, infer_row_count, column_count);

    const auto train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ train_row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_train_table = train_dataframe.get_table(this->get_homogen_table_id());
    const auto infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ infer_row_count, column_count }.fill_uniform(-0.3, 1.));
    const table x_infer_table = infer_dataframe.get_table(this->get_homogen_table_id());

    const table y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto infer_result =
        this->train_and_infer_spmd(knn_desc, x_train_table, y_train_table, x_infer_table);

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SPMD_CLS_EXTERNAL_TEST("distributed knn classification hepmass 50kx10k") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(4);

    constexpr double target_score = 0.8;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    const auto score = this->distr_classification(x_train_table,
                                                  y_train_table,
                                                  x_infer_table,
                                                  y_infer_table,
                                                  n_classes,
                                                  n_neighbors);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_SPMD_REG_EXTERNAL_TEST("distributed knn distance regression hepmass 50kx10k") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    this->set_rank_count(4);

    constexpr double target_score = 0.072;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::minkowski_distance::descriptor<>;
    constexpr double minkowski_degree = 2.0;
    const auto distance_desc = distance_t(minkowski_degree);
    constexpr auto voting = oneapi::dal::knn::voting_mode::distance;

    const auto score = this->distr_regression(x_train_table,
                                              y_train_table,
                                              x_infer_table,
                                              y_infer_table,
                                              n_neighbors,
                                              distance_desc,
                                              voting);
    CAPTURE(score, target_score);
    REQUIRE(score < target_score);
}

KNN_SPMD_REG_EXTERNAL_TEST("distributed knn uniform regression hepmass 50kx10k") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    this->set_rank_count(4);

    constexpr double target_score = 0.072;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    const auto score = this->distr_regression(x_train_table,
                                              y_train_table,
                                              x_infer_table,
                                              y_infer_table,
                                              n_neighbors);
    CAPTURE(score, target_score);
    REQUIRE(score < target_score);
}

KNN_SPMD_CLS_BF_EXTERNAL_TEST(
    "distributed knn classification hepmass 50kx10k with distance voting)") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(3);

    const double target_score = this->get_policy().is_gpu() ? 0.8 : 0.6;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::minkowski_distance::descriptor<>;
    constexpr double minkowski_degree = 2.0;
    const auto distance_desc = distance_t(minkowski_degree);
    constexpr auto voting = oneapi::dal::knn::voting_mode::distance;
    const auto score = this->distr_classification(x_train_table,
                                                  y_train_table,
                                                  x_infer_table,
                                                  y_infer_table,
                                                  n_classes,
                                                  n_neighbors,
                                                  distance_desc,
                                                  voting);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_SPMD_CLS_BF_EXTERNAL_TEST(
    "distributed knn classification hepmass 50kx10k with Minkowski distance (p = 2.5)") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(4);

    constexpr double target_score = 0.8;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::minkowski_distance::descriptor<>;
    const double minkowski_degree = 2.5;
    const auto distance_desc = distance_t(minkowski_degree);
    const auto score = this->distr_classification(x_train_table,
                                                  y_train_table,
                                                  x_infer_table,
                                                  y_infer_table,
                                                  n_classes,
                                                  n_neighbors,
                                                  distance_desc);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_SPMD_CLS_BF_EXTERNAL_TEST(
    "distributed knn classification hepmass 50kx10k with Cosine distance") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(3);

    constexpr double target_score = 0.78;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::cosine_distance::descriptor<>;
    const auto distance_desc = distance_t();
    const auto score = this->distr_classification(x_train_table,
                                                  y_train_table,
                                                  x_infer_table,
                                                  y_infer_table,
                                                  n_classes,
                                                  n_neighbors,
                                                  distance_desc);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_SPMD_CLS_BF_EXTERNAL_TEST(
    "distributed knn classification hepmass 50kx10k with Chebyshev distance") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->not_available_on_device());

    this->set_rank_count(4);

    constexpr double target_score = 0.8;

    constexpr std::int64_t feature_count = 28;
    constexpr std::int64_t n_classes = 2;
    constexpr std::int64_t n_neighbors = 3;

    const te::dataframe train_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_50t_test.csv" });

    const te::dataframe infer_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/hepmass/dataset/hepmass_10t_test.csv" });

    const table x_train_table =
        train_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));
    const table x_infer_table =
        infer_dataframe.get_table(this->get_homogen_table_id(), range(0, feature_count));

    const table y_train_table = train_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));
    const table y_infer_table = infer_dataframe.get_table(this->get_homogen_table_id(),
                                                          range(feature_count, feature_count + 1));

    using distance_t = oneapi::dal::chebyshev_distance::descriptor<>;
    const auto score = this->distr_classification(x_train_table,
                                                  y_train_table,
                                                  x_infer_table,
                                                  y_infer_table,
                                                  n_classes,
                                                  n_neighbors,
                                                  distance_t{});
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

} // namespace oneapi::dal::knn::test
