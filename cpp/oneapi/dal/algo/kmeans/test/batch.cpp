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

#include <limits>
#include <cmath>

#include "oneapi/dal/algo/kmeans/train.hpp"
#include "oneapi/dal/algo/kmeans/infer.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/metrics/clustering.hpp"

namespace oneapi::dal::kmeans::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class kmeans_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor(std::int64_t cluster_count,
                        std::int64_t max_iteration_count,
                        Float accuracy_threshold) const {
        return kmeans::descriptor<Float, Method>{}
            .set_cluster_count(cluster_count)
            .set_max_iteration_count(max_iteration_count)
            .set_accuracy_threshold(accuracy_threshold);
    }

    auto get_descriptor(std::int64_t cluster_count) const {
        return kmeans::descriptor<Float, Method>{ cluster_count };
    }

    void exact_checks(const table& data,
                      const table& initial_centroids,
                      const table& ref_centroids,
                      const table& ref_labels,
                      std::int64_t cluster_count,
                      std::int64_t max_iteration_count,
                      Float accuracy_threshold,
                      Float ref_objective_function = -1.0,
                      bool test_convergence = false) {
        CAPTURE(cluster_count);

        INFO("create descriptor")
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        check_train_result(kmeans_desc, train_result, ref_centroids, ref_labels, test_convergence);

        INFO("run inference");
        const auto infer_result = infer(kmeans_desc, model, data);
        check_infer_result(kmeans_desc, infer_result, ref_labels, ref_objective_function);
    }

    void exact_checks_with_reordering(const table& data,
                                      const table& initial_centroids,
                                      const table& ref_centroids,
                                      const table& ref_labels,
                                      std::int64_t cluster_count,
                                      std::int64_t max_iteration_count,
                                      Float accuracy_threshold,
                                      Float ref_objective_function = -1.0,
                                      bool test_convergence = false) {
        CAPTURE(cluster_count);

        INFO("create descriptor")
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();

        auto match_map = array<Float>::zeros(cluster_count);
        find_match_centroids(ref_centroids,
                             model.get_centroids(),
                             ref_centroids.get_column_count(),
                             match_map);
        check_train_result(kmeans_desc,
                           train_result,
                           match_map,
                           ref_centroids,
                           ref_labels,
                           test_convergence);
        INFO("run inference");
        const auto infer_result = infer(kmeans_desc, model, data);
        check_infer_result(kmeans_desc,
                           infer_result,
                           match_map,
                           ref_labels,
                           ref_objective_function);
    }

    void dbi_determenistic_checks(const table& data,
                                  std::int64_t cluster_count,
                                  std::int64_t max_iteration_count,
                                  Float accuracy_threshold,
                                  Float ref_dbi,
                                  Float ref_obj_func) {
        CAPTURE(cluster_count);

        INFO("create descriptor")
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        const auto data_rows = row_accessor<const Float>(data).pull({ 0, cluster_count });
        const auto initial_centroids =
            homogen_table::wrap(data_rows.get_data(), cluster_count, data.get_column_count());

        INFO("run training");
        const auto train_result = train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        REQUIRE(te::has_no_nans(model.get_centroids()));

        INFO("run inference");
        const auto infer_result = infer(kmeans_desc, model, data);
        REQUIRE(te::has_no_nans(infer_result.get_labels()));

        Float obj_ref_tol = 1.0e-4;
        Float dbi_ref_tol = 1.0e-4;
        auto dbi = te::davies_bouldin_index(data, model.get_centroids(), infer_result.get_labels());
        CAPTURE(dbi, ref_dbi);
        CAPTURE(infer_result.get_objective_function_value(), ref_obj_func);
        REQUIRE(check_value_with_ref_tol(dbi, ref_dbi, dbi_ref_tol));
        REQUIRE(check_value_with_ref_tol(infer_result.get_objective_function_value(),
                                         ref_obj_func,
                                         obj_ref_tol));
    }

    void train_with_initialization_checks(const table& data,
                                          const table& ref_centroids,
                                          const table& ref_labels,
                                          std::int64_t cluster_count,
                                          std::int64_t max_iteration_count,
                                          Float accuracy_threshold,
                                          kmeans::model<>& model) {
        CAPTURE(cluster_count);

        INFO("create descriptor")
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = train(kmeans_desc, data);
        check_train_result(kmeans_desc, train_result, ref_centroids, ref_labels, false);
        model = train_result.get_model();
    }

    void infer_checks(const table& data,
                      kmeans::model<>& model,
                      const table& ref_labels,
                      Float ref_objective_function = -1.0) {
        CAPTURE(model.get_cluster_count());

        INFO("create descriptor")
        const auto kmeans_desc = get_descriptor(model.get_cluster_count());

        INFO("run inference");
        const auto infer_result = infer(kmeans_desc, model, data);
        check_infer_result(kmeans_desc, infer_result, ref_labels, ref_objective_function);
    }

    void check_train_result(const kmeans::descriptor<Float, Method>& desc,
                            const kmeans::train_result<>& result,
                            const table& ref_centroids,
                            const table& ref_labels,
                            bool test_convergence = false) {
        const auto [centroids, labels, iteration_count] = unpack_result(result);

        check_nans(result);
        const Float strict_rel_tol =
            5.f * std::numeric_limits<Float>::epsilon() * iteration_count * 100;
        check_centroid_match_with_rel_tol(strict_rel_tol, ref_centroids, centroids);
        check_label_match(ref_labels, labels);
        if (test_convergence) {
            INFO("check convergence");
            REQUIRE(iteration_count < desc.get_max_iteration_count());
        }
    }

    void check_train_result(const kmeans::descriptor<Float, Method>& desc,
                            const kmeans::train_result<>& result,
                            const array<Float>& match_map,
                            const table& ref_centroids,
                            const table& ref_labels,
                            bool test_convergence = false) {
        const auto [centroids, labels, iteration_count] = unpack_result(result);

        check_nans(result);
        const Float strict_rel_tol = std::numeric_limits<Float>::epsilon() * iteration_count * 10;
        check_centroid_match_with_rel_tol(match_map, strict_rel_tol, ref_centroids, centroids);
        check_label_match(match_map, ref_labels, labels);

        if (test_convergence) {
            INFO("check convergence");
            REQUIRE(iteration_count < desc.get_max_iteration_count());
        }
    }

    bool check_value_with_ref_tol(Float val, Float ref_val, Float ref_tol) {
        Float max_abs = std::max(fabs(val), fabs(ref_val));
        if (max_abs == 0.0)
            return true;
        return fabs(val - ref_val) / max_abs < ref_tol;
    }

    void check_base_infer_result(const kmeans::descriptor<Float, Method>& desc,
                                 const kmeans::infer_result<>& result,
                                 Float ref_objective_function) {
        const auto [labels, objective_function] = unpack_result(result);

        check_nans(result);

        INFO("check if non-negative objective function value is expected")
        REQUIRE(objective_function >= 0.0);

        Float rel_tol = 1.0e-5;
        if (!(ref_objective_function < 0.0)) {
            CAPTURE(objective_function, ref_objective_function, rel_tol);
            REQUIRE(check_value_with_ref_tol(objective_function, ref_objective_function, rel_tol));
        }
    }

    void check_infer_result(const kmeans::descriptor<Float, Method>& desc,
                            const kmeans::infer_result<>& result,
                            const table& ref_labels,
                            Float ref_objective_function) {
        const auto [labels, objective_function] = unpack_result(result);

        check_base_infer_result(desc, result, ref_objective_function);
        check_label_match(ref_labels, labels);
    }

    void check_infer_result(const kmeans::descriptor<Float, Method>& desc,
                            const kmeans::infer_result<>& result,
                            const array<Float>& match_map,
                            const table& ref_labels,
                            Float ref_objective_function) {
        const auto [labels, objective_function] = unpack_result(result);
        check_base_infer_result(desc, result, ref_objective_function);
        check_label_match(match_map, ref_labels, labels);
    }

    void check_centroid_match_with_rel_tol(Float rel_tol, const table& left, const table& right) {
        INFO("check if centroid shape is expected")
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());

        INFO("check if centroid match is expected")
        const auto left_rows = row_accessor<const Float>(left).pull({ 0, -1 });
        const auto right_rows = row_accessor<const Float>(right).pull({ 0, -1 });
        const Float alpha = std::numeric_limits<Float>::min();
        for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
            const Float l = left_rows[i];
            const Float r = right_rows[i];
            if (fabs(l - r) < alpha)
                continue;
            const Float denom = fabs(l) + fabs(r) + alpha;
            if (fabs(l - r) / denom > rel_tol) {
                CAPTURE(l, r, l - r, rel_tol, (l - r) / denom / rel_tol);
                FAIL("Centroid feature mismatch");
            }
        }
    }

    void check_centroid_match_with_rel_tol(const array<Float>& match_map,
                                           Float rel_tol,
                                           const table& left,
                                           const table& right) {
        INFO("check if centroid shape is expected")
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());

        INFO("check if centroid match is expected")
        const auto left_rows = row_accessor<const Float>(left).pull({ 0, -1 });
        const auto right_rows = row_accessor<const Float>(right).pull({ 0, -1 });
        const Float alpha = std::numeric_limits<Float>::min();
        std::int64_t cluster_count = left.get_row_count();
        std::int64_t feature_count = left.get_column_count();
        for (std::int64_t i = 0; i < cluster_count; i++) {
            for (std::int64_t j = 0; j < feature_count; j++) {
                const Float l = left_rows[match_map[i] * feature_count + j];
                const Float r = right_rows[i * feature_count + j];
                if (fabs(l - r) < alpha)
                    continue;
                const Float denom = fabs(l) + fabs(r) + alpha;
                if (fabs(l - r) / denom > rel_tol) {
                    CAPTURE(l, r);
                    FAIL("Centroid feature mismatch for mapped centroids");
                }
            }
        }
    }

    Float squared_euclidian_distance(std::int64_t x_offset,
                                     const array<Float>& x,
                                     std::int64_t y_offset,
                                     const array<Float>& y,
                                     std::int64_t feature_count) {
        Float sum = 0.0;
        for (std::int64_t i = 0; i < feature_count; i++) {
            Float val = x[x_offset * feature_count + i] - y[y_offset * feature_count + i];
            sum += val * val;
        }
        return sum;
    }

    void find_match_centroids(const table& ref_centroids,
                              const table& centroids,
                              std::int64_t feature_count,
                              array<Float>& match_map) {
        REQUIRE(ref_centroids.get_row_count() == centroids.get_row_count());
        REQUIRE(ref_centroids.get_column_count() == centroids.get_column_count());
        const auto ref_rows = row_accessor<const Float>(ref_centroids).pull({ 0, -1 });
        const auto cur_rows = row_accessor<const Float>(centroids).pull({ 0, -1 });
        std::int64_t cluster_count = centroids.get_row_count();
        auto match_counters = array<std::int64_t>::zeros(cluster_count);
        for (std::int64_t i = 0; i < cluster_count; i++) {
            Float min_distance =
                squared_euclidian_distance(0, ref_rows, i, cur_rows, feature_count);
            for (std::int64_t j = 1; j < cluster_count; j++) {
                Float probe_distance =
                    squared_euclidian_distance(j, ref_rows, i, cur_rows, feature_count);
                if (probe_distance < min_distance) {
                    match_map.get_mutable_data()[i] = j;
                    min_distance = probe_distance;
                }
            }
        }
        for (std::int64_t i = 0; i < cluster_count; i++) {
            std::int64_t match_count = 0;
            for (std::int64_t j = 0; j < cluster_count; j++) {
                match_count += match_map[i] == j ? 1 : 0;
            }
            REQUIRE(match_count == 1);
        }
    }

    void check_label_match(const table& left, const table& right) {
        INFO("check if label shape is expected")
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());
        REQUIRE(left.get_column_count() == 1);
        INFO("check if label match is expected")
        const auto left_rows = row_accessor<const Float>(left).pull({ 0, -1 });
        const auto right_rows = row_accessor<const Float>(right).pull({ 0, -1 });
        for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
            const Float l = left_rows[i];
            const Float r = right_rows[i];
            if (l != r) {
                CAPTURE(l, r);
                FAIL("Label mismatch");
            }
        }
    }

    void check_label_match(const array<Float>& match_map, const table& left, const table& right) {
        INFO("check if label shape is expected")
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());
        REQUIRE(left.get_column_count() == 1);

        INFO("check if label match is expected")
        const auto left_rows = row_accessor<const Float>(left).pull({ 0, -1 });
        const auto right_rows = row_accessor<const Float>(right).pull({ 0, -1 });
        for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
            const Float l = left_rows[i];
            const Float r = right_rows[i];
            if (l != match_map[r]) {
                CAPTURE(l, r, match_map[r]);
                FAIL("Label mismatch for mapped centroids");
            }
        }
    }

    void check_nans(const kmeans::train_result<>& result) {
        const auto [centroids, labels, iteration_count] = unpack_result(result);

        INFO("check if there is no NaN in centroids")
        REQUIRE(te::has_no_nans(centroids));

        INFO("check if there is no NaN in labels")
        REQUIRE(te::has_no_nans(labels));
    }

    void check_nans(const kmeans::infer_result<>& result) {
        const auto [labels, objective_function] = unpack_result(result);

        INFO("check if there is no NaN in objective function values")
        REQUIRE(!std::isnan(objective_function));

        INFO("check if there is no NaN in labels")
        REQUIRE(te::has_no_nans(labels));
    }

private:
    static auto unpack_result(const kmeans::train_result<>& result) {
        const auto centroids = result.get_model().get_centroids();
        const auto labels = result.get_labels();
        const auto iteration_count = result.get_iteration_count();
        return std::make_tuple(centroids, labels, iteration_count);
    }
    static auto unpack_result(const kmeans::infer_result<>& result) {
        const auto labels = result.get_labels();
        const auto objective_function = result.get_objective_function_value();
        return std::make_tuple(labels, objective_function);
    }
};

using kmeans_types = COMBINE_TYPES((float, double), (kmeans::method::lloyd_dense));

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

    model<> model;

    this->train_with_initialization_checks(x, c_final, y, 2, 4, 0.001, model);

    const Float data_infer[] = { 1.0, 1.0,  0.0, 1.0,  1.0,  0.0,  2.0, 2.0,  7.0,
                                 0.0, -1.0, 0.0, -5.0, -5.0, -5.0, 0.0, -2.0, 1.0 };
    const auto x2 = homogen_table::wrap(data_infer, 9, 2);
    Float expected_obj_function = 4;
    this->infer_checks(x, model, y, expected_obj_function);
}
/*
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
*/
/*
// This stress test is commented due to GPU K-Means crash.
// Will be added when the issue is resolved.
TEMPLATE_LIST_TEST_M(kmeans_batch_test,
                     "kmeans partial centroid adjustment test",
                     "[kmeans][batch][nightly]",
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
*/
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

    this->dbi_determenistic_checks(x,
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

    this->dbi_determenistic_checks(x,
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

    this->dbi_determenistic_checks(x,
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

    this->dbi_determenistic_checks(x,
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

    this->dbi_determenistic_checks(x,
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

    this->dbi_determenistic_checks(x,
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

    this->dbi_determenistic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func);
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

    this->dbi_determenistic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func);
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

    this->dbi_determenistic_checks(x,
                                   cluster_count,
                                   max_iteration_count,
                                   0.0,
                                   ref_dbi,
                                   ref_obj_func);
}

} // namespace oneapi::dal::kmeans::test
