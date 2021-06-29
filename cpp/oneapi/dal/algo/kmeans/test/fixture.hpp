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

#pragma once

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
namespace la = dal::test::engine::linalg;

using kmeans_types = COMBINE_TYPES((float, double), (kmeans::method::lloyd_dense));

template <typename TestType, typename Derived>
class kmeans_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using base_t = te::crtp_algo_fixture<TestType, Derived>;
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = kmeans::task::clustering;
    using descriptor_t = kmeans::descriptor<float_t, method_t, task_t>;
    using train_input_t = kmeans::train_input<task_t>;
    using train_result_t = kmeans::train_result<task_t>;
    using infer_input_t = kmeans::infer_input<task_t>;
    using infer_result_t = kmeans::infer_result<task_t>;
    using model_t = kmeans::model<task_t>;

    descriptor_t get_descriptor(std::int64_t cluster_count,
                                std::int64_t max_iteration_count,
                                float_t accuracy_threshold) const {
        return descriptor_t{}
            .set_cluster_count(cluster_count)
            .set_max_iteration_count(max_iteration_count)
            .set_accuracy_threshold(accuracy_threshold);
    }

    descriptor_t get_descriptor(std::int64_t cluster_count) const {
        return descriptor_t{ cluster_count };
    }

    void exact_checks(const table& data,
                      const table& initial_centroids,
                      const table& ref_centroids,
                      const table& ref_labels,
                      std::int64_t cluster_count,
                      std::int64_t max_iteration_count,
                      float_t accuracy_threshold,
                      float_t ref_objective_function = -1.0,
                      bool test_convergence = false) {
        CAPTURE(cluster_count);

        INFO("create descriptor")
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        check_train_result(kmeans_desc, train_result, ref_centroids, ref_labels, test_convergence);

        INFO("run inference");
        const auto infer_result = this->infer(kmeans_desc, model, data);
        check_infer_result(kmeans_desc, infer_result, ref_labels, ref_objective_function);
    }

    void exact_checks_with_reordering(const table& data,
                                      const table& initial_centroids,
                                      const table& ref_centroids,
                                      const table& ref_labels,
                                      std::int64_t cluster_count,
                                      std::int64_t max_iteration_count,
                                      float_t accuracy_threshold,
                                      float_t ref_objective_function = -1.0,
                                      bool test_convergence = false) {
        CAPTURE(cluster_count);

        INFO("create descriptor")
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();

        auto match_map = array<float_t>::zeros(cluster_count);
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
        const auto infer_result = this->infer(kmeans_desc, model, data);
        check_infer_result(kmeans_desc,
                           infer_result,
                           match_map,
                           ref_labels,
                           ref_objective_function);
    }

    void dbi_deterministic_checks(const table& data,
                                  std::int64_t cluster_count,
                                  std::int64_t max_iteration_count,
                                  float_t accuracy_threshold,
                                  float_t ref_dbi,
                                  float_t ref_obj_func,
                                  float_t obj_ref_tol = 1.0e-4,
                                  float_t dbi_ref_tol = 1.0e-4) {
        CAPTURE(cluster_count);

        INFO("create descriptor")
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        const auto data_rows = row_accessor<const float_t>(data).pull({ 0, cluster_count });
        const auto initial_centroids =
            homogen_table::wrap(data_rows.get_data(), cluster_count, data.get_column_count());

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        REQUIRE(te::has_no_nans(model.get_centroids()));

        INFO("run inference");
        const auto infer_result = this->infer(kmeans_desc, model, data);
        REQUIRE(te::has_no_nans(infer_result.get_labels()));

        auto dbi = te::davies_bouldin_index(data, model.get_centroids(), infer_result.get_labels());
        CAPTURE(dbi, ref_dbi);
        CAPTURE(infer_result.get_objective_function_value(), ref_obj_func);
        REQUIRE(check_value_with_ref_tol(dbi, ref_dbi, dbi_ref_tol));
        REQUIRE(check_value_with_ref_tol(infer_result.get_objective_function_value(),
                                         ref_obj_func,
                                         obj_ref_tol));
    }

    void dbi_deterministic_checks_with_centroids(const table& data,
                                                 const table& initial_centroids,
                                                 std::int64_t cluster_count,
                                                 std::int64_t max_iteration_count,
                                                 float_t accuracy_threshold,
                                                 float_t ref_dbi,
                                                 float_t ref_obj_func,
                                                 float_t obj_ref_tol = 1.0e-4,
                                                 float_t dbi_ref_tol = 1.0e-4) {
        CAPTURE(cluster_count);

        INFO("create descriptor")
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        REQUIRE(te::has_no_nans(model.get_centroids()));

        INFO("run inference");
        const auto infer_result = this->infer(kmeans_desc, model, data);
        REQUIRE(te::has_no_nans(infer_result.get_labels()));

        auto dbi = te::davies_bouldin_index(data, model.get_centroids(), infer_result.get_labels());
        CAPTURE(dbi, ref_dbi);
        CAPTURE(infer_result.get_objective_function_value(), ref_obj_func);
        REQUIRE(check_value_with_ref_tol(dbi, ref_dbi, dbi_ref_tol));
        REQUIRE(check_value_with_ref_tol(infer_result.get_objective_function_value(),
                                         ref_obj_func,
                                         obj_ref_tol));
    }

    model_t train_with_initialization_checks(const table& data,
                                             const table& ref_centroids,
                                             const table& ref_labels,
                                             std::int64_t cluster_count,
                                             std::int64_t max_iteration_count,
                                             float_t accuracy_threshold) {
        CAPTURE(cluster_count);

        INFO("create descriptor")
        const auto kmeans_desc =
            get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = this->train(kmeans_desc, data);
        check_train_result(kmeans_desc, train_result, ref_centroids, ref_labels, false);
        return train_result.get_model();
    }

    void infer_checks(const table& data,
                      const model_t& model,
                      const table& ref_labels,
                      float_t ref_objective_function = -1.0) {
        CAPTURE(model.get_cluster_count());

        INFO("create descriptor")
        const auto kmeans_desc = get_descriptor(model.get_cluster_count());

        INFO("run inference");
        const auto infer_result = this->infer(kmeans_desc, model, data);
        check_infer_result(kmeans_desc, infer_result, ref_labels, ref_objective_function);
    }

    void check_train_result(const descriptor_t& desc,
                            const train_result_t& result,
                            const table& ref_centroids,
                            const table& ref_labels,
                            bool test_convergence = false) {
        const auto [centroids, labels, iteration_count] = unpack_result(result);

        check_nans(result);
        const float_t strict_rel_tol =
            5.f * std::numeric_limits<float_t>::epsilon() * iteration_count * 100;
        check_centroid_match_with_rel_tol(strict_rel_tol, ref_centroids, centroids);
        check_label_match(ref_labels, labels);
        if (test_convergence) {
            INFO("check convergence");
            REQUIRE(iteration_count < desc.get_max_iteration_count());
        }
    }

    void check_train_result(const descriptor_t& desc,
                            const infer_result_t& result,
                            const array<float_t>& match_map,
                            const table& ref_centroids,
                            const table& ref_labels,
                            bool test_convergence = false) {
        const auto [centroids, labels, iteration_count] = unpack_result(result);

        check_nans(result);
        const float_t strict_rel_tol =
            std::numeric_limits<float_t>::epsilon() * iteration_count * 10;
        check_centroid_match_with_rel_tol(match_map, strict_rel_tol, ref_centroids, centroids);
        check_label_match(match_map, ref_labels, labels);

        if (test_convergence) {
            INFO("check convergence");
            REQUIRE(iteration_count < desc.get_max_iteration_count());
        }
    }

    bool check_value_with_ref_tol(float_t val, float_t ref_val, float_t ref_tol) {
        float_t max_abs = std::max(fabs(val), fabs(ref_val));
        if (max_abs == 0.0)
            return true;
        CAPTURE(val, ref_val, fabs(val - ref_val) / max_abs, ref_tol);
        return fabs(val - ref_val) / max_abs < ref_tol;
    }

    void check_base_infer_result(const descriptor_t& desc,
                                 const infer_result_t& result,
                                 float_t ref_objective_function) {
        const auto [labels, objective_function] = unpack_result(result);

        check_nans(result);

        INFO("check if non-negative objective function value is expected")
        REQUIRE(objective_function >= 0.0);

        float_t rel_tol = 1.0e-5;
        if (!(ref_objective_function < 0.0)) {
            CAPTURE(objective_function, ref_objective_function, rel_tol);
            REQUIRE(check_value_with_ref_tol(objective_function, ref_objective_function, rel_tol));
        }
    }

    void check_infer_result(const descriptor_t& desc,
                            const infer_result_t& result,
                            const table& ref_labels,
                            float_t ref_objective_function) {
        const auto [labels, objective_function] = unpack_result(result);

        check_base_infer_result(desc, result, ref_objective_function);
        check_label_match(ref_labels, labels);
    }

    void check_infer_result(const descriptor_t& desc,
                            const infer_result_t& result,
                            const array<float_t>& match_map,
                            const table& ref_labels,
                            float_t ref_objective_function) {
        const auto [labels, objective_function] = unpack_result(result);
        check_base_infer_result(desc, result, ref_objective_function);
        check_label_match(match_map, ref_labels, labels);
    }

    void check_centroid_match_with_rel_tol(float_t rel_tol, const table& left, const table& right) {
        INFO("check if centroid shape is expected")
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());

        INFO("check if centroid match is expected")
        const auto left_rows = row_accessor<const float_t>(left).pull({ 0, -1 });
        const auto right_rows = row_accessor<const float_t>(right).pull({ 0, -1 });
        const float_t alpha = std::numeric_limits<float_t>::min();
        for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
            const float_t l = left_rows[i];
            const float_t r = right_rows[i];
            if (fabs(l - r) < alpha)
                continue;
            const float_t denom = fabs(l) + fabs(r) + alpha;
            if (fabs(l - r) / denom > rel_tol) {
                CAPTURE(l, r, l - r, rel_tol, (l - r) / denom / rel_tol);
                FAIL("Centroid feature mismatch");
            }
        }
    }

    void check_centroid_match_with_rel_tol(const array<float_t>& match_map,
                                           float_t rel_tol,
                                           const table& left,
                                           const table& right) {
        INFO("check if centroid shape is expected")
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());

        INFO("check if centroid match is expected")
        const auto left_rows = row_accessor<const float_t>(left).pull({ 0, -1 });
        const auto right_rows = row_accessor<const float_t>(right).pull({ 0, -1 });
        const float_t alpha = std::numeric_limits<float_t>::min();
        std::int64_t cluster_count = left.get_row_count();
        std::int64_t feature_count = left.get_column_count();
        for (std::int64_t i = 0; i < cluster_count; i++) {
            for (std::int64_t j = 0; j < feature_count; j++) {
                const float_t l = left_rows[match_map[i] * feature_count + j];
                const float_t r = right_rows[i * feature_count + j];
                if (fabs(l - r) < alpha)
                    continue;
                const float_t denom = fabs(l) + fabs(r) + alpha;
                if (fabs(l - r) / denom > rel_tol) {
                    CAPTURE(l, r);
                    FAIL("Centroid feature mismatch for mapped centroids");
                }
            }
        }
    }

    float_t squared_euclidian_distance(std::int64_t x_offset,
                                       const array<float_t>& x,
                                       std::int64_t y_offset,
                                       const array<float_t>& y,
                                       std::int64_t feature_count) {
        float_t sum = 0.0;
        for (std::int64_t i = 0; i < feature_count; i++) {
            float_t val = x[x_offset * feature_count + i] - y[y_offset * feature_count + i];
            sum += val * val;
        }
        return sum;
    }

    void find_match_centroids(const table& ref_centroids,
                              const table& centroids,
                              std::int64_t feature_count,
                              array<float_t>& match_map) {
        REQUIRE(ref_centroids.get_row_count() == centroids.get_row_count());
        REQUIRE(ref_centroids.get_column_count() == centroids.get_column_count());
        const auto ref_rows = row_accessor<const float_t>(ref_centroids).pull({ 0, -1 });
        const auto cur_rows = row_accessor<const float_t>(centroids).pull({ 0, -1 });
        std::int64_t cluster_count = centroids.get_row_count();
        auto match_counters = array<std::int64_t>::zeros(cluster_count);
        for (std::int64_t i = 0; i < cluster_count; i++) {
            float_t min_distance =
                squared_euclidian_distance(0, ref_rows, i, cur_rows, feature_count);
            for (std::int64_t j = 1; j < cluster_count; j++) {
                float_t probe_distance =
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
        const auto left_rows = row_accessor<const float_t>(left).pull({ 0, -1 });
        const auto right_rows = row_accessor<const float_t>(right).pull({ 0, -1 });
        for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
            const float_t l = left_rows[i];
            const float_t r = right_rows[i];
            if (l != r) {
                CAPTURE(l, r);
                FAIL("Label mismatch");
            }
        }
    }

    void check_label_match(const array<float_t>& match_map, const table& left, const table& right) {
        INFO("check if label shape is expected")
        REQUIRE(left.get_row_count() == right.get_row_count());
        REQUIRE(left.get_column_count() == right.get_column_count());
        REQUIRE(left.get_column_count() == 1);

        INFO("check if label match is expected")
        const auto left_rows = row_accessor<const float_t>(left).pull({ 0, -1 });
        const auto right_rows = row_accessor<const float_t>(right).pull({ 0, -1 });
        for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
            const float_t l = left_rows[i];
            const float_t r = right_rows[i];
            if (l != match_map[r]) {
                CAPTURE(l, r, match_map[r]);
                FAIL("Label mismatch for mapped centroids");
            }
        }
    }

    void check_nans(const train_result_t& result) {
        const auto [centroids, labels, iteration_count] = unpack_result(result);

        INFO("check if there is no NaN in centroids")
        REQUIRE(te::has_no_nans(centroids));

        INFO("check if there is no NaN in labels")
        REQUIRE(te::has_no_nans(labels));
    }

    void check_nans(const infer_result_t& result) {
        const auto [labels, objective_function] = unpack_result(result);

        INFO("check if there is no NaN in objective function values")
        REQUIRE(!std::isnan(objective_function));

        INFO("check if there is no NaN in labels")
        REQUIRE(te::has_no_nans(labels));
    }

    static auto unpack_result(const train_result_t& result) {
        const auto centroids = result.get_model().get_centroids();
        const auto labels = result.get_labels();
        const auto iteration_count = result.get_iteration_count();
        return std::make_tuple(centroids, labels, iteration_count);
    }

    static auto unpack_result(const infer_result_t& result) {
        const auto labels = result.get_labels();
        const auto objective_function = result.get_objective_function_value();
        return std::make_tuple(labels, objective_function);
    }
};

} // namespace oneapi::dal::kmeans::test
