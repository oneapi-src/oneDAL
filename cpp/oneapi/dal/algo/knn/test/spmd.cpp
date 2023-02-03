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
    using result_t = typename base_t::result_t;
    using input_t = typename base_t::input_t;

    using default_distance_t = oneapi::dal::minkowski_distance::descriptor<>;

    using voting_t = oneapi::dal::knn::voting_mode;
    constexpr static inline voting_t default_voting = voting_t::uniform;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename... Args>
    result_t infer_override(Args&&... args) {
        return this->infer_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<input_t> split_infer_input_override(std::int64_t split_count, Args&&... args) {
        const input_t input{ std::forward<Args>(args)... };
        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);

        std::vector<input_t> split_input;
        split_input.reserve(split_count);

        for (std::int64_t i = 0; i < split_count; i++) {
            split_input.push_back( //
                input_t{ split_data[i], input.get_model() });
        }

        return split_input;
    }

    result_t merge_infer_result_override(const std::vector<result_t>& results) {
        // Responses are distributed accross the ranks, we combine them into one table;
        // Model, iteration_count, objective_function_value are the same for all ranks
        std::vector<table> responses;
        for (const auto& r : results) {
            responses.push_back(r.get_responses());
        }
        const auto full_responses = te::stack_tables_by_rows<float_t>(responses);

        return result_t{}
            .set_result_options(result_options::responses)
            .set_responses(full_responses);
    }



private:
    std::int64_t rank_count_ = 1;
};



#define KNN_SPMD_SMALL_TEST(name)                                               \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                   \
                         name,                                             \
                         "[small-dataset][knn][integration][spmd][test]", \
                         knn_cls_types)

#define KNN_SPMD_CLS_SYNTHETIC_TEST(name)                                               \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                       \
                         name,                                                 \
                         "[synthetic-dataset][knn][integration][spmd][test]", \
                         knn_cls_types)

#define KNN_SPMD_CLS_EXTERNAL_TEST(name)                                               \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                      \
                         name,                                                \
                         "[external-dataset][knn][integration][spmd][test]", \
                         knn_cls_types)

#define KNN_SPMD_CLS_BF_EXTERNAL_TEST(name)                                            \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                      \
                         name,                                                \
                         "[external-dataset][knn][integration][spmd][test]", \
                         knn_cls_bf_types)

#define KNN_SPMD_REG_SYNTHETIC_TEST(name)                                               \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                       \
                         name,                                                 \
                         "[synthetic-dataset][knn][integration][spmd][test]", \
                         knn_reg_types)

#define KNN_SPMD_REG_EXTERNAL_TEST(name)                                               \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                      \
                         name,                                                \
                         "[external-dataset][knn][integration][spmd][test]", \
                         knn_reg_types)

#define KNN_SPMD_REG_BF_EXTERNAL_TEST(name)                                            \
    TEMPLATE_LIST_TEST_M(knn_spmd_test,                                      \
                         name,                                                \
                         "[external-dataset][knn][integration][spmd][test]", \
                         knn_reg_bf_types)

KNN_SPMD_SMALL_TEST("knn nearest points test predefined 7x5x2") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    this->set_rank_count(1);
    constexpr std::int64_t train_row_count = 9;
    constexpr std::int64_t infer_row_count = 9;
    constexpr std::int64_t column_count = 1;

    CAPTURE(train_row_count, infer_row_count, column_count);

    constexpr std::int64_t train_element_count = train_row_count * column_count;
    constexpr std::int64_t infer_element_count = infer_row_count * column_count;

    constexpr std::array<float, train_element_count> train = { -1.f, 0.f, 1.f, 3.f, 5.f, 10.f, 20.f, 100.f, 1000.f };//{ 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 30.f, 100.f, 1000.f };

    constexpr std::array<float, infer_element_count> infer = {-10.f, 0.f, 0.6f, 0.6f, 40.f, 1000.f, 999.f, -0.4f, 0.6f };//{ -1.f, 2.49f, 3.1f, 5.1f, -100.f, 11.1f, 11.9f, 11.9f, 20.f, 100.f };

    const auto x_train_table = homogen_table::wrap(train.data(), train_row_count, column_count);
    const auto x_infer_table = homogen_table::wrap(infer.data(), infer_row_count, column_count);
    const auto y_train_table = this->arange(train_row_count);

    const auto knn_desc = this->get_descriptor(train_row_count, 1);

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SPMD_CLS_SYNTHETIC_TEST("distributed knn nearest points test random uniform 513x301x17") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    this->set_rank_count(10);

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

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SPMD_REG_SYNTHETIC_TEST("distributed knn nearest points test random uniform using regression 513x301x17") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    this->set_rank_count(10);

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

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SPMD_CLS_SYNTHETIC_TEST("distributed knn nearest points test random uniform 16390x20x5") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_kd_tree);

    this->set_rank_count(10);

    constexpr std::int64_t train_row_count = 16390;
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

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}

KNN_SPMD_CLS_EXTERNAL_TEST("distributed knn classification hepmass 50kx10k") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(10);

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

    const auto score = this->classification(x_train_table,
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

    this->set_rank_count(10);

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

    const auto score = this->regression(x_train_table,
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

    this->set_rank_count(10);

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

    const auto score =
        this->regression(x_train_table, y_train_table, x_infer_table, y_infer_table, n_neighbors);

    CAPTURE(score, target_score);
    REQUIRE(score < target_score);
}

KNN_SPMD_CLS_BF_EXTERNAL_TEST("distributed knn classification hepmass 50kx10k with distance voting)") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(10);

    // TODO: Investigate low accuracy on CPU
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
    const auto score = this->classification(x_train_table,
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

KNN_SPMD_CLS_BF_EXTERNAL_TEST("distributed knn classification hepmass 50kx10k with Minkowski distance (p = 2.5)") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(10);

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
    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors,
                                            distance_desc);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_SPMD_CLS_BF_EXTERNAL_TEST("distributed knn classification hepmass 50kx10k with Cosine distance") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(10);

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
    const auto score = this->classification(x_train_table,
                                            y_train_table,
                                            x_infer_table,
                                            y_infer_table,
                                            n_classes,
                                            n_neighbors,
                                            distance_desc);
    CAPTURE(score);
    REQUIRE(score >= target_score);
}

KNN_SPMD_CLS_BF_EXTERNAL_TEST("distributed knn classification hepmass 50kx10k with Chebyshev distance") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->is_kd_tree);
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->not_available_on_device());

    this->set_rank_count(10);

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
    const auto score = this->classification(x_train_table,
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

