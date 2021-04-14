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
#include <tuple>

#include "oneapi/dal/algo/kmeans/backend/gpu/kmeans_impl.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::kmeans::test {

namespace te = dal::test::engine;
namespace prm = dal::backend::primitives;

template <typename TestType>
class kmeans_impl_test : public te::float_algo_fixture<TestType> {
public:
    using float_t = TestType;

    auto get_descriptor(std::int64_t cluster_count,
                        std::int64_t max_iteration_count,
                        float_t accuracy_threshold) const {
        return kmeans::descriptor<float_t, kmeans::method::lloyd_dense>{}
            .set_cluster_count(cluster_count)
            .set_max_iteration_count(max_iteration_count)
            .set_accuracy_threshold(accuracy_threshold);
    }

    void run_obj_func_check(const prm::ndview<float_t, 1>& min_distances) {
        auto desc = this->get_descriptor(10, 10, 0.0);
        auto row_count = min_distances.get_shape()[0];
        backend::kmeans_impl<float_t> estimator(this->get_queue(), row_count, 2, desc);
        estimator.compute_objective_function(min_distances);
        check_objective_function(min_distances, estimator.get_objective_function());
    }

    void check_objective_function(const prm::ndview<float_t, 1>& min_distances, float_t objective_function_value) {
        auto row_count = min_distances.get_shape()[0];
        auto min_distance_ptr = min_distances.get_data();
        float_t sum = 0.0;
        for(std::int64_t i = 0; i < row_count; i++) {
            sum += min_distance_ptr[i];
        }
        float_t tol = 1.0e-5;
        CAPTURE(sum, objective_function_value);
        REQUIRE(std::fabs(sum - objective_function_value) / std::max(std::fabs(sum), std::fabs(objective_function_value)) < tol);
    }

    void run_counting(const prm::ndview<int32_t, 2> labels, 
                              std::int64_t num_clusters) {
        auto desc = this->get_descriptor(num_clusters, 10, 0.0);
        auto row_count = labels.get_shape()[0];
        auto counters = prm::ndarray<std::int32_t, 1>::empty(this->get_queue(), num_clusters);
        counters.fill(this->get_queue(), 0);
        backend::kmeans_impl<float_t> estimator(this->get_queue(), row_count, 2, desc);
        estimator.count_clusters_impl(labels, counters);
        check_counters(labels, counters, num_clusters, estimator.get_num_empty_clusters());
    }

    void run_reduce_centroids(const prm::ndview<float_t, 2> data, const prm::ndview<int32_t, 2> labels, 
                              std::int64_t num_clusters, std::int64_t num_parts) {
        auto desc = this->get_descriptor(num_clusters, 10, 0.0);
        auto row_count = data.get_shape()[0];
        auto column_count = data.get_shape()[1];
        auto centroids = prm::ndarray<float_t, 2>::empty(this->get_queue(), {num_clusters, column_count});
        auto partial_centroids = prm::ndarray<float_t, 2>::empty(this->get_queue(), 
                                                                {num_clusters * num_parts, column_count});
        auto counters = prm::ndarray<std::int32_t, 1>::empty(this->get_queue(), num_clusters);
        backend::kmeans_impl<float_t> estimator(this->get_queue(), row_count, column_count, desc);
        estimator.count_clusters_impl(labels, counters);
        check_counters(labels, counters, num_clusters, estimator.get_num_empty_clusters());
        estimator.reduce_centroids_impl(data, labels, centroids, partial_centroids, num_parts);
        check_reduced_centroids(data, labels, centroids, counters);
    }    

    void check_counters(const prm::ndview<int32_t, 2>& labels,
                                 const prm::ndview<int32_t, 1>& counters, 
                                 std::int32_t num_clusters,
                                 std::int32_t num_empty_clusters) {
        auto row_count = labels.get_shape()[0];
        auto temp_counters = prm::ndarray<std::int32_t, 1>::empty(this->get_queue(), num_clusters);
        temp_counters.fill(this->get_queue(), 0).wait_and_throw();
        auto temp_counters_ptr = temp_counters.get_mutable_data();
        auto counters_ptr = counters.get_data();
        auto labels_ptr = labels.get_data();
        for(std::int64_t i = 0; i < row_count; i++) {
            const auto cl = labels_ptr[i];
            REQUIRE(cl >= 0);
            REQUIRE(cl < num_clusters);
            temp_counters_ptr[cl] += 1;
        }
        std::int32_t total = 0;
        std::int32_t empties = 0;
        for(std::int64_t i = 0; i < num_clusters; i++) {
            CAPTURE(i);
            REQUIRE(temp_counters_ptr[i] == counters_ptr[i]);
            total += temp_counters_ptr[i];
            empties += std::int32_t(temp_counters_ptr[i] == 0);

        }
        REQUIRE(total == row_count);
        REQUIRE(num_empty_clusters == empties);
    }

    void check_reduced_centroids(const prm::ndview<float_t, 2>& data, 
                                 const prm::ndview<std::int32_t, 2>& labels,
                                 const prm::ndview<float_t, 2>& centroids,
                                 const prm::ndview<std::int32_t, 1>& counters) {
        auto row_count = data.get_shape()[0];
        auto column_count = data.get_shape()[1];
        auto num_clusters = centroids.get_shape()[0];
        auto temp_centroids = prm::ndarray<float_t, 2>::empty(this->get_queue(), {num_clusters, column_count});
        auto temp_centroids_ptr = temp_centroids.get_mutable_data();
        auto data_ptr = data.get_data();
        auto labels_ptr = labels.get_data();
        for(std::int64_t i = 0; i < row_count; i++) {
            const auto cl = labels_ptr[i];
            REQUIRE(cl >= 0);
            REQUIRE(cl < num_clusters);
            for(std::int64_t j = 0; j < column_count; j++) {
                temp_centroids_ptr[cl * column_count + j] = data_ptr[i * column_count + j];
            }
        }
        auto centroids_ptr = centroids.get_data();
        auto counters_ptr = counters.get_data();
        for(std::int64_t i = 0; i < num_clusters; i++) {
            if(counters_ptr[i] == 0) continue;
            for(std::int64_t j = 0; j < column_count; j++) {
                std::cout << i << " " << j << " " << centroids_ptr[i * column_count + j] << std::endl;
                REQUIRE(centroids_ptr[i * column_count + j] == temp_centroids_ptr[i * column_count + j] / counters_ptr[i]);
            }
        }
    }


    void run_selection(const prm::ndview<float_t, 2> data, const prm::ndview<float_t, 2> centroids) {
/*        const std::int64_t row_count = data.get_shape()[0];
        const std::int64_t column_count = data.get_shape()[1];
        kmeans_impl<float_t> estimator(queue, row_count, column_count, params);
        check_labels(data, centroids, labels);*/
    }
};

using kmeans_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(kmeans_impl_test,
                     "objective function unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    using float_t = TestType;

    std::int64_t row_count = 100001;

    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ row_count, 1 }.fill_uniform(0.0, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = prm::ndarray<float_t, 1>::wrap(df_rows.get_data(), row_count);
    this->run_obj_func_check(data_array);
}

TEMPLATE_LIST_TEST_M(kmeans_impl_test,
                     "label counting unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    std::int64_t row_count = 100001;
    std::int64_t cluster_count = 37;

    const auto dfl = GENERATE_DATAFRAME(te::dataframe_builder{ row_count, 1 }.fill_uniform(0, cluster_count - 1));
    const table dfl_table = dfl.get_table(this->get_homogen_table_id());
    const auto dfl_rows = row_accessor<const std::int32_t>(dfl_table).pull(this->get_queue(), { 0, -1 });
    auto labels = prm::ndarray<std::int32_t, 2>::wrap(dfl_rows.get_data(), {row_count, 1});
    this->run_counting(labels, cluster_count);
}

/*
TEMPLATE_LIST_TEST_M(kmeans_impl_test,
                     "centroid reduction unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    using float_t = TestType;

    std::int64_t row_count = 10001;
    std::int64_t cluster_count = 37;
    std::int64_t column_count = 31;
    std::int64_t num_parts = 16;

    const auto dfl = GENERATE_DATAFRAME(te::dataframe_builder{ row_count, 1 }.fill_uniform(0, cluster_count - 1));
    const table dfl_table = dfl.get_table(this->get_homogen_table_id());
    const auto dfl_rows = row_accessor<const std::int32_t>(dfl_table).pull(this->get_queue(), { 0, -1 });
    auto labels = prm::ndarray<std::int32_t, 2>::wrap(dfl_rows.get_data(), {row_count, 1});

    const auto dfd = GENERATE_DATAFRAME(te::dataframe_builder{ row_count, column_count }.fill_uniform(-0.9, 1.7));
    const table dfd_table = dfd.get_table(this->get_homogen_table_id());
    const auto dfd_rows = row_accessor<const float_t>(dfd_table).pull(this->get_queue(), { 0, -1 });
    auto data = prm::ndarray<float_t, 2>::wrap(dfd_rows.get_data(), { row_count, column_count });

    this->run_reduce_centroids(data, labels, cluster_count, num_parts);
}
*/

} // namespace oneapi::dal::kmeans::test
