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

#include "oneapi/dal/algo/kmeans/test/fixture.hpp"

namespace oneapi::dal::kmeans::test {

template <typename TestType>
class kmeans_batch_test : public kmeans_test<TestType, kmeans_batch_test<TestType>> {};

/*
TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans degenerated test",
                     "[kmeans][batch]",
                     kmeans_types) {
    // number of observations is equal to number of centroids (obvious clustering)
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;
    Float data[] = { 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 1.0 };
    const auto x = homogen_table::wrap(data, 3, 5);

    Float labels[] = { 0, 1, 2 };
    const auto y = homogen_table::wrap(labels, 3, 1);
    this->exact_checks(x, x, x, y, 3, 2, 0.0, 0.0, false);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test, "kmeans relocation test", "[kmeans][batch]", kmeans_types) {
    // relocation of empty cluster to the best candidate
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;
    Float data[] = { 0, 0, 0.5, 0, 0.5, 1, 1, 1 };
    const auto x = homogen_table::wrap(data, 4, 2);

    Float initial_centroids[] = { 0.5, 0.5, 3, 3 };
    const auto c_init = homogen_table::wrap(initial_centroids, 2, 2);

    Float final_centroids[] = { 0.25, 0, 0.75, 1 };
    const auto c_final = homogen_table::wrap(final_centroids, 2, 2);

    std::int64_t labels[] = { 0, 0, 1, 1 };
    const auto y = homogen_table::wrap(labels, 4, 1);

    Float expected_obj_function = 0.25;
    std::int64_t expected_n_iters = 4;
    this->exact_checks_with_reordering(x,
                                       c_init,
                                       c_final,
                                       y,
                                       2,
                                       expected_n_iters + 1,
                                       0.0,
                                       expected_obj_function,
                                       false);
}
*/

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans empty clusters test",
                     "[kmeans][batch]",
                     kmeans_types) {
    // proper relocation order for multiple empty clusters
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;
    Float data[] = { -10, -9.5, -9, -8.5, -8, -1, 1, 9, 9.5, 10 };
    const auto x = homogen_table::wrap(data, 10, 1);

    Float initial_centroids[] = { -10, -10, -10 };
    const auto c_init = homogen_table::wrap(initial_centroids, 3, 1);

    Float final_centroids[] = { -1.65, 10, 9.5 };
    const auto c_final = homogen_table::wrap(final_centroids, 3, 1);

    Float labels[] = { 0, 0, 0, 0, 0, 0, 0, 2, 2, 1 };
    const auto y = homogen_table::wrap(labels, 10, 1);

    this->exact_checks(x, c_init, c_final, y, 3, 1, 0.0);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans train/infer test",
                     "[kmeans][batch]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;
    const Float data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                           -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };
    const auto x = homogen_table::wrap(data, 8, 2);

    const Float final_centroids[] = { -1.5, -1.5, 1.5, 1.5 };
    const auto c_final = homogen_table::wrap(final_centroids, 2, 2);

    const int labels[] = { 1, 1, 1, 1, 0, 0, 0, 0 };
    const auto y = homogen_table::wrap(labels, 8, 1);

    const auto model = this->train_with_initialization_checks(x, c_final, y, 2, 4, 0.001);

    const Float data_infer[] = { 1.0, 1.0,  0.0, 1.0,  1.0,  0.0,  2.0, 2.0,  7.0,
                                 0.0, -1.0, 0.0, -5.0, -5.0, -5.0, 0.0, -2.0, 1.0 };
    const auto x2 = homogen_table::wrap(data_infer, 9, 2);
    Float expected_obj_function = 4;
    this->infer_checks(x, model, y, expected_obj_function);
}

// This stress test is commented due to CPU K-Means crash.
// Will be added when the issue is resolved.
TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans block test",
                     "[kmeans][batch][nightly]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr std::int64_t row_count = 1024 * 1024;
    constexpr std::int64_t column_count = 1024 * 2 / sizeof(Float);
    constexpr std::int64_t cluster_count = 1;
    constexpr std::int64_t max_iteration_count = 1;

    const auto x_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x_table = x_dataframe.get_table(this->get_homogen_table_id());

    const auto first_row = row_accessor<const Float>(x_table).pull({ 0, 1 });
    const auto c_init = homogen_table::wrap(first_row.get_data(), 1, column_count);

    auto labels = array<float>::zeros(row_count);
    const auto y = homogen_table::wrap(labels.get_data(), row_count, 1);

    auto stat = te::compute_basic_statistics<Float>(x_dataframe);
    const auto c_final = homogen_table::wrap(stat.get_means().get_data(), 1, column_count);
    auto variance = stat.get_variances().get_data();
    double obj_function = 0.0;
    for (std::int64_t i = 0; i < column_count; ++i) {
        obj_function += variance[i];
    }
    obj_function *= column_count - 1;

    this->exact_checks(x_table,
                       c_init,
                       c_final,
                       y,
                       cluster_count,
                       max_iteration_count,
                       obj_function);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans partial centroids stress test",
                     "[kmeans][batch][nightly][stress]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    constexpr std::int64_t row_count = 8 * 1024;
    constexpr std::int64_t column_count = 2 * 1024;
    constexpr std::int64_t cluster_count = 8 * 1024;

    const auto x_dataframe = GENERATE_DATAFRAME(
        te::dataframe_builder{ row_count, column_count }.fill_uniform(-0.2, 0.5));
    const table x = x_dataframe.get_table(this->get_homogen_table_id());

    const auto first_row = row_accessor<const Float>(x).pull({ 0, 1 });
    const auto c_init = homogen_table::wrap(first_row.get_data(), 1, column_count);

    auto labels = array<std::int32_t>::zeros(1 * cluster_count);
    auto label_ptr = labels.get_mutable_data();
    auto first_label = &label_ptr[0];
    std::iota(first_label, first_label + row_count, std::int32_t(0));
    const auto y = homogen_table::wrap(labels.get_data(), row_count, 1);

    this->exact_checks(x, x, x, y, cluster_count, 1, 0.0);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "higgs: samples=1M, clusters=10, iters=3",
                     "[kmeans][batch][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/higgs/dataset/higgs_1m_test.csv" });
    const table x = data.get_table(this->get_homogen_table_id());

    constexpr std::int64_t cluster_count = 10;
    constexpr std::int64_t max_iteration_count = 3;
    constexpr Float ref_dbi = 3.1997724684;
    constexpr Float ref_obj_func = 14717484.0;

    this->dbi_deterministic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "higgs: samples=1M, clusters=100, iters=3",
                     "[kmeans][nightly][batch][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/higgs/dataset/higgs_1m_test.csv" });
    const table x = data.get_table(this->get_homogen_table_id());

    constexpr std::int64_t cluster_count = 100;
    constexpr std::int64_t max_iteration_count = 3;
    constexpr Float ref_dbi = 2.7450205195;
    constexpr Float ref_obj_func = 10704352.0;

    this->dbi_deterministic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "higgs: samples=1M, clusters=250, iters=3",
                     "[kmeans][nightly][batch][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/higgs/dataset/higgs_1m_test.csv" });
    const table x = data.get_table(this->get_homogen_table_id());

    constexpr std::int64_t cluster_count = 250;
    constexpr std::int64_t max_iteration_count = 3;
    constexpr Float ref_dbi = 2.5923397174;
    constexpr Float ref_obj_func = 9335216.0;

    this->dbi_deterministic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "susy: samples=0.5M, clusters=10, iters=10",
                     "[kmeans][nightly][batch][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/susy/dataset/susy_test.csv" });
    const table x = data.get_table(this->get_homogen_table_id());

    constexpr std::int64_t cluster_count = 10;
    constexpr std::int64_t max_iteration_count = 10;
    constexpr Float ref_dbi = 1.7730860782;
    constexpr Float ref_obj_func = 3183696.0;

    this->dbi_deterministic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "susy: samples=0.5M, clusters=100, iters=10",
                     "[kmeans][nightly][batch][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/susy/dataset/susy_test.csv" });
    const table x = data.get_table(this->get_homogen_table_id());

    constexpr std::int64_t cluster_count = 100;
    constexpr std::int64_t max_iteration_count = 10;
    constexpr Float ref_dbi = 1.9384844916;
    constexpr Float ref_obj_func = 1757022.625;

    this->dbi_deterministic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "susy: samples=0.5M, clusters=250, iters=10",
                     "[kmeans][nightly][batch][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/susy/dataset/susy_test.csv" });
    const table x = data.get_table(this->get_homogen_table_id());

    constexpr std::int64_t cluster_count = 250;
    constexpr std::int64_t max_iteration_count = 10;
    constexpr Float ref_dbi = 1.8950113604;
    constexpr Float ref_obj_func = 1400958.5;

    this->dbi_deterministic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "epsilon: samples=80K, clusters=512, iters=2",
                     "[kmeans][nightly][batch][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const te::dataframe data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/epsilon/dataset/epsilon_80k_train.csv" });
    const table x = data.get_table(this->get_homogen_table_id());

    constexpr std::int64_t cluster_count = 512;
    constexpr std::int64_t max_iteration_count = 2;
    constexpr Float ref_dbi = 6.9367580565;
    constexpr Float ref_obj_func = 50128.640625;

    this->dbi_deterministic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func,
                                   1.0e-3);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "epsilon: samples=80K, clusters=1024, iters=2",
                     "[kmeans][nightly][batch][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const te::dataframe data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/epsilon/dataset/epsilon_80k_train.csv" });
    const table x = data.get_table(this->get_homogen_table_id());

    constexpr std::int64_t cluster_count = 1024;
    constexpr std::int64_t max_iteration_count = 2;
    constexpr Float ref_dbi = 5.59003873;
    constexpr Float ref_obj_func = 49518.75;

    this->dbi_deterministic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func,
                                   1.0e-3);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "epsilon: samples=80K, clusters=2048, iters=2",
                     "[kmeans][nightly][batch][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    using Float = std::tuple_element_t<0, TestType>;

    const te::dataframe data = GENERATE_DATAFRAME(
        te::dataframe_builder{ "workloads/epsilon/dataset/epsilon_80k_train.csv" });
    const table x = data.get_table(this->get_homogen_table_id());

    constexpr std::int64_t cluster_count = 2048;
    constexpr std::int64_t max_iteration_count = 2;
    constexpr Float ref_dbi = 4.3202752143;
    constexpr Float ref_obj_func = 48437.6015625;

    this->dbi_deterministic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func,
                                   1.0e-3);
}

} // namespace oneapi::dal::kmeans::test
