/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/kmeans/train.hpp"
#include "oneapi/dal/algo/kmeans/infer.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include <limits>

namespace oneapi::dal::kmeans::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class kmeans_batch_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor(std::int64_t cluster_count, std::int64_t max_iteration_count, Float accuracy_threshold) const {
        return kmeans::descriptor<Float, Method>{}
            .set_cluster_count(cluster_count)
            .set_max_iteration_count(max_iteration_count)
            .set_accuracy_threshold(accuracy_threshold);
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    void general_checks(const table& data,
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
        const auto kmeans_desc = get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_result = train(kmeans_desc, data, initial_centroids);
        const auto model = train_result.get_model();
        check_train_result(kmeans_desc, train_result, data, ref_centroids, ref_labels, test_convergence);

        INFO("run inference");
        const auto infer_result = infer(kmeans_desc, model, data);
        check_infer_result(kmeans_desc, infer_result, ref_labels, ref_objective_function);
    }

    void check_train_result(const kmeans::descriptor<Float, Method>& desc,
                            const kmeans::train_result<>& result,
                            const table& data,
                            const table& ref_centroids,
                            const table& ref_labels,
                            bool test_convergence = false) {
        const auto [centroids, labels, iteration_count] = unpack_result(result);

        check_nans(result);
        const Float strict_rel_tol = std::numeric_limits<Float>::epsilon() * iteration_count * 10;
        check_table_match_with_rel_tol(strict_rel_tol, ref_centroids, centroids);
        check_table_match(ref_labels, labels);
        if(test_convergence) {
            INFO("check convergence");
            REQUIRE(iteration_count < desc.get_max_iteration_count());            
        }
    }

    void check_infer_result(const kmeans::descriptor<Float, Method>& desc,
                            const kmeans::infer_result<>& result,
                            const table& ref_labels,
                            Float ref_objective_function) {
        const auto [labels, objective_function] = unpack_result(result);

        check_nans(result);

        check_table_match(ref_labels, labels);

        SECTION("non-negative objective function value is expected") {
            REQUIRE(objective_function >= 0.0);
        }

        Float rel_tol = 1.0e-5;
        if(!(ref_objective_function < 0.0)) {
            REQUIRE(fabs(objective_function - ref_objective_function) / (fabs(objective_function) + (ref_objective_function)) < rel_tol);
        }
    }    


    void check_table_match_with_rel_tol(Float rel_tol, const table& left, const table& right) {
        SECTION("centroid shape is expected") {
            REQUIRE(left.get_row_count() == right.get_row_count());
            REQUIRE(left.get_column_count() == right.get_column_count());
        }
        SECTION("centroid match is expected") {
            const auto left_rows = row_accessor<const Float>(left).pull({ 0, -1 });
            const auto right_rows = row_accessor<const Float>(right).pull({ 0, -1 });
            const Float alpha = std::numeric_limits<Float>::min() * rel_tol;
            bool failed = false;
            for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
                const Float l = left_rows[i];
                const Float r = right_rows[i];
                if(fabs(l - r) < alpha) continue;
                const Float denom = fabs(l) + fabs(r) + alpha;
                failed |= fabs(l - r) / denom < rel_tol;
            }
            REQUIRE(!failed);
        }
    }

    void check_table_match(const table& left, const table& right) {
        SECTION("centroid shape is expected") {
            REQUIRE(left.get_row_count() == right.get_row_count());
            REQUIRE(left.get_column_count() == right.get_column_count());
        }
        SECTION("centroid match is expected") {
            const auto left_rows = row_accessor<const Float>(left).pull({ 0, -1 });
            const auto right_rows = row_accessor<const Float>(right).pull({ 0, -1 });
            bool failed = false;
            for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
                const Float l = left_rows[i];
                const Float r = right_rows[i];
                failed |= l != r;
            }
            REQUIRE(!failed);
        }
    }

    void check_nans(const kmeans::train_result<>& result) {
        const auto [centroids, labels, iteration_count] = unpack_result(result);

        SECTION("there is no NaN in centroids") {
            REQUIRE(te::has_no_nans(centroids));
        }
    }

    void check_nans(const kmeans::infer_result<>& result) {
        const auto [labels, objective_function] = unpack_result(result);

        SECTION("there is no NaN in objective function values") {
            REQUIRE(te::has_no_nans(labels));
        }
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

TEMPLATE_LIST_TEST_M(kmeans_batch_test, "kmeans common tests", "[kmeans][degenerated][batch]", kmeans_types) {
    using oneapi::dal::detail::empty_delete;
    using Float = std::tuple_element_t<0, TestType>;
    Float data[] = {0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 1.0};
    homogen_table x{ data, 5, 3, empty_delete<const Float>() };
    Float labels[] = {0, 1, 2};
    homogen_table y{ labels, 1, 3, empty_delete<const Float>() };
    this->general_checks(x, x, x, y, 3, 2, 0.0, 0.0, false);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test, "kmeans common tests", "[kmeans][relocation][convergence][batch]", kmeans_types) {
    using oneapi::dal::detail::empty_delete;
    using Float = std::tuple_element_t<0, TestType>;

    Float data[] = {0, 0, 0.5, 0, 0.5, 1, 1, 1};
    homogen_table x{ data, 2, 4, empty_delete<const Float>() };

    Float initial_centroids[] = {0.5, 0.5, 3, 3};
    homogen_table c_init{ initial_centroids, 2, 2, empty_delete<const Float>() };

    Float final_centroids[] = {0.25, 0, 0.75, 1};
    homogen_table c_final{ final_centroids, 2, 2, empty_delete<const Float>() };

    std::int64_t labels[] = {0, 0, 1, 1};
    homogen_table y{ labels, 1, 4, empty_delete<const std::int64_t>() };

    Float expected_obj_function = 0.25;
    std::int64_t expected_n_iters = 3;
    this->general_checks(x, c_init, c_final, y, 2, expected_n_iters + 1, 0.0, expected_obj_function, true);
}

TEMPLATE_LIST_TEST_M(kmeans_batch_test, "kmeans common tests", "[kmeans][empty cluster][batch]", kmeans_types) {
    using oneapi::dal::detail::empty_delete;
    using Float = std::tuple_element_t<0, TestType>;

    Float data[] = {-10, -9.5, -9, -8.5, -8, -1, 1, 9, 9.5, 10};
    homogen_table x{ data, 1, 10, empty_delete<const Float>() };

    Float initial_centroids[] = {-10, -10, -10};
    homogen_table c_init{ initial_centroids, 1, 3, empty_delete<const Float>() };

    Float final_centroids[] = {-10, 10, 9.5};
    homogen_table c_final{ final_centroids, 1, 3, empty_delete<const Float>() };

    this->general_checks(x, c_init, c_final, table{}, 3, 1, 0.0);
}


} // namespace oneapi::dal::kmeans::test
