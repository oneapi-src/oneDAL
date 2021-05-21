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

#include "oneapi/dal/algo/kmeans/backend/gpu/kmeans_impl_fp.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kmeans_impl_int.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::kmeans::backend::test {

namespace te = dal::test::engine;
namespace pr = dal::backend::primitives;
namespace de = dal::detail;

template <typename TestType>
class kmeans_impl_test : public te::float_algo_fixture<TestType> {
public:
    using float_t = TestType;

    void fill_uniform(pr::ndarray<float_t, 2>& val, float_t a, float_t b, std::int64_t seed = 777) {
        std::int32_t elem_count = de::integral_cast<std::int32_t>(val.get_count());
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float_t> distr(a, b);

        // move generation to device when rng is available there
        float_t* val_ptr = de::host_allocator<float_t>().allocate(val.get_count());
        for (std::int32_t el = 0; el < elem_count; el++) {
            val_ptr[el] = distr(rng);
        }
        val.assign(this->get_queue(), val_ptr, val.get_count()).wait_and_throw();
        de::host_allocator<float_t>().deallocate(val_ptr, val.get_count());
    }

    void run_obj_func_check(const pr::ndview<float_t, 2>& closest_distances, float_t tol = 1.0e-5) {
        auto obj_func = pr::ndarray<float_t, 1>::empty(this->get_queue(), 1);
        compute_objective_function(this->get_queue(), closest_distances, obj_func).wait_and_throw();
        check_objective_function(closest_distances, obj_func.get_data()[0], tol);
    }

    void check_objective_function(const pr::ndview<float_t, 2>& closest_distances,
                                  float_t objective_function_value,
                                  float_t tol) {
        auto row_count = closest_distances.get_shape()[0];
        auto min_distance_ptr = closest_distances.get_data();
        float_t sum = 0.0;
        for (std::int64_t i = 0; i < row_count; i++) {
            sum += min_distance_ptr[i];
        }
        CAPTURE(sum, objective_function_value);
        REQUIRE(std::fabs(sum - objective_function_value) /
                    std::max(std::fabs(sum), std::fabs(objective_function_value)) <
                tol);
    }

    void run_counting(const pr::ndview<int32_t, 2>& labels, std::int64_t cluster_count) {
        auto counters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), cluster_count);
        auto empty_clusters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), 1);
        counters.fill(this->get_queue(), 0).wait_and_throw();
        count_clusters(this->get_queue(), labels, cluster_count, counters).wait_and_throw();
        count_empty_clusters(this->get_queue(), cluster_count, counters, empty_clusters)
            .wait_and_throw();
        check_counters(labels, counters, cluster_count, empty_clusters.get_data()[0]);
    }

    void run_partial_reduce(const pr::ndview<float_t, 2>& data,
                            const pr::ndview<int32_t, 2>& labels,
                            std::int64_t cluster_count,
                            std::int64_t part_count,
                            float_t tol = 1.0e-5) {
        auto column_count = data.get_shape()[1];
        auto centroids =
            pr::ndarray<float_t, 2>::empty(this->get_queue(), { cluster_count, column_count });
        centroids.fill(this->get_queue(), 0).wait_and_throw();
        auto partial_centroids =
            pr::ndarray<float_t, 2>::empty(this->get_queue(),
                                           { cluster_count * part_count, column_count });
        partial_centroids.fill(this->get_queue(), 0).wait_and_throw();
        auto counters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), cluster_count);
        auto empty_clusters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), 1);
        counters.fill(this->get_queue(), 0).wait_and_throw();
        count_clusters(this->get_queue(), labels, cluster_count, counters).wait_and_throw();
        count_empty_clusters(this->get_queue(), cluster_count, counters, empty_clusters)
            .wait_and_throw();
        check_counters(labels, counters, cluster_count, empty_clusters.get_data()[0]);

        partial_reduce_centroids(this->get_queue(),
                                 data,
                                 labels,
                                 cluster_count,
                                 part_count,
                                 partial_centroids)
            .wait_and_throw();
        check_partial_centroids(data, labels, partial_centroids, part_count, tol);
    }

    void run_reduce_centroids(const pr::ndview<float_t, 2>& data,
                              const pr::ndview<int32_t, 2>& labels,
                              std::int64_t cluster_count,
                              std::int64_t part_count,
                              float_t tol = 1.0e-5) {
        auto column_count = data.get_shape()[1];
        auto centroids =
            pr::ndarray<float_t, 2>::empty(this->get_queue(), { cluster_count, column_count });
        centroids.fill(this->get_queue(), 0).wait_and_throw();
        auto partial_centroids =
            pr::ndarray<float_t, 2>::empty(this->get_queue(),
                                           { cluster_count * part_count, column_count });
        partial_centroids.fill(this->get_queue(), 0).wait_and_throw();
        auto counters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), cluster_count);
        counters.fill(this->get_queue(), 0).wait_and_throw();
        auto empty_clusters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), 1);
        count_clusters(this->get_queue(), labels, cluster_count, counters).wait_and_throw();
        count_empty_clusters(this->get_queue(), cluster_count, counters, empty_clusters)
            .wait_and_throw();
        check_counters(labels, counters, cluster_count, empty_clusters.get_data()[0]);
        partial_reduce_centroids(this->get_queue(),
                                 data,
                                 labels,
                                 cluster_count,
                                 part_count,
                                 partial_centroids)
            .wait_and_throw();
        check_partial_centroids(data, labels, partial_centroids, part_count, tol);
        merge_reduce_centroids(this->get_queue(),
                               counters,
                               partial_centroids,
                               part_count,
                               centroids)
            .wait_and_throw();
        check_reduced_centroids(data, labels, centroids, counters, tol);
    }

    void run_selection(const pr::ndview<float_t, 2>& data,
                       const pr::ndview<float_t, 2>& centroids,
                       std::int64_t block_rows,
                       float_t tol = 1.0e-5) {
        auto row_count = data.get_shape()[0];
        auto cluster_count = centroids.get_shape()[0];
        auto labels = pr::ndarray<std::int32_t, 2>::empty(this->get_queue(), { row_count, 1 });
        auto closest_distances =
            pr::ndarray<float_t, 2>::empty(this->get_queue(), { row_count, 1 });
        auto distances =
            pr::ndarray<float_t, 2>::empty(this->get_queue(), { block_rows, cluster_count });

        assign_clusters<float_t, pr::squared_l2_metric<float_t>>(this->get_queue(),
                                                                 data,
                                                                 centroids,
                                                                 block_rows,
                                                                 labels,
                                                                 distances,
                                                                 closest_distances,
                                                                 {})
            .wait_and_throw();
        check_assignments(data, centroids, labels, closest_distances, tol);
    }

    void run_candidates(pr::ndview<float_t, 2>& closest_distances, std::int64_t candidate_count) {
        auto candidate_indices =
            pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), candidate_count);
        auto candidate_distances =
            pr::ndarray<float_t, 1>::empty(this->get_queue(), candidate_count);

        find_candidates(this->get_queue(),
                        closest_distances,
                        candidate_count,
                        candidate_indices,
                        candidate_distances,
                        {})
            .wait_and_throw();
        check_candidates(closest_distances,
                         candidate_count,
                         candidate_indices,
                         candidate_distances);
    }

    void check_candidates(const pr::ndview<float_t, 2>& closest_distances,
                          std::int64_t candidate_count,
                          const pr::ndview<std::int32_t, 1>& candidate_indices,
                          const pr::ndview<float_t, 1>& candidate_distances) {
        auto elem_count = closest_distances.get_shape()[0];
        auto closest_distances_ptr = closest_distances.get_data();
        auto candidate_indices_ptr = candidate_indices.get_data();
        auto candidate_distances_ptr = candidate_distances.get_data();
        for (std::int64_t i = 0; i < candidate_count; i++) {
            auto distance = candidate_distances_ptr[i];
            auto index = candidate_indices_ptr[i];
            std::int64_t count = 0;
            for (std::int64_t j = 0; j < elem_count; j++) {
                auto cur_val = closest_distances_ptr[j];
                if (cur_val < distance)
                    count++;
            }
            CAPTURE(i, count);
            REQUIRE(count < candidate_count);
            REQUIRE(index >= 0);
            REQUIRE(index < elem_count);
        }
    }

    void check_assignments(const pr::ndview<float_t, 2>& data,
                           const pr::ndview<float_t, 2>& centroids,
                           const pr::ndview<std::int32_t, 2>& labels,
                           const pr::ndview<float_t, 2>& closest_distances,
                           float_t tol) {
        auto row_count = data.get_shape()[0];
        auto column_count = data.get_shape()[1];
        auto cluster_count = centroids.get_shape()[0];
        auto data_ptr = data.get_data();
        auto centroids_ptr = centroids.get_data();
        auto closest_distances_ptr = closest_distances.get_data();
        auto labels_ptr = labels.get_data();
        for (std::int64_t i = 0; i < row_count; i++) {
            float_t min_distance = dal::detail::limits<float_t>::max();
            std::int32_t min_index = -1;
            for (std::int64_t j = 0; j < cluster_count; j++) {
                float_t distance = 0;
                ;
                for (std::int64_t k = 0; k < column_count; k++) {
                    float_t diff =
                        data_ptr[i * column_count + k] - centroids_ptr[j * column_count + k];
                    distance += diff * diff;
                }
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = j;
                }
            }
            CAPTURE(i, labels_ptr[i]);
            REQUIRE(labels_ptr[i] == min_index);
            auto v1 = closest_distances_ptr[i];
            auto v2 = min_distance;
            CAPTURE(v1, v2);
            auto maxv = std::max(std::fabs(v1), std::fabs(v2));
            if (maxv == 0.0)
                continue;
            REQUIRE(std::fabs(v1 - v2) / maxv < tol);
        }
    }

    void check_counters(const pr::ndview<int32_t, 2>& labels,
                        const pr::ndview<int32_t, 1>& counters,
                        std::int32_t cluster_count,
                        std::int32_t empty_cluster_count) {
        auto row_count = labels.get_shape()[0];
        auto temp_counters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), cluster_count);
        temp_counters.fill(this->get_queue(), 0).wait_and_throw();
        auto temp_counters_ptr = temp_counters.get_mutable_data();
        auto counters_ptr = counters.get_data();
        auto labels_ptr = labels.get_data();
        for (std::int64_t i = 0; i < row_count; i++) {
            const auto cl = labels_ptr[i];
            REQUIRE(cl >= 0);
            REQUIRE(cl < cluster_count);
            temp_counters_ptr[cl] += 1;
        }
        std::int32_t total = 0;
        std::int32_t empties = 0;
        for (std::int64_t i = 0; i < cluster_count; i++) {
            CAPTURE(i);
            REQUIRE(temp_counters_ptr[i] == counters_ptr[i]);
            total += temp_counters_ptr[i];
            empties += std::int32_t(temp_counters_ptr[i] == 0);
        }
        REQUIRE(total == row_count);
        REQUIRE(empty_cluster_count == empties);
    }

    void check_partial_centroids(const pr::ndview<float_t, 2>& data,
                                 const pr::ndview<std::int32_t, 2>& labels,
                                 const pr::ndview<float_t, 2>& partial_centroids,
                                 std::int64_t part_count,
                                 float_t tol) {
        auto row_count = data.get_shape()[0];
        auto column_count = data.get_shape()[1];
        auto cluster_count = partial_centroids.get_shape()[0] / part_count;
        auto temp_partial_centroids =
            pr::ndarray<float_t, 2>::empty(this->get_queue(),
                                           { cluster_count * part_count, column_count });
        temp_partial_centroids.fill(this->get_queue(), 0.0).wait_and_throw();
        auto temp_partial_centroids_ptr = temp_partial_centroids.get_mutable_data();
        auto data_ptr = data.get_data();
        auto labels_ptr = labels.get_data();
        for (std::int64_t i = 0; i < row_count; i++) {
            const auto cl = labels_ptr[i];
            const auto part = i % part_count;
            REQUIRE(cl >= 0);
            REQUIRE(cl < cluster_count);
            for (std::int64_t j = 0; j < column_count; j++) {
                temp_partial_centroids_ptr[part * cluster_count * column_count + cl * column_count +
                                           j] += data_ptr[i * column_count + j];
            }
        }
        auto partial_centroids_ptr = partial_centroids.get_data();
        for (std::int64_t i = 0; i < part_count; i++) {
            for (std::int64_t k = 0; k < cluster_count; k++) {
                for (std::int64_t j = 0; j < column_count; j++) {
                    CAPTURE(i, k, j);
                    auto v1 = partial_centroids_ptr[i * column_count + j];
                    auto v2 = temp_partial_centroids_ptr[i * column_count + j];
                    CAPTURE(v1, v2);
                    auto maxv = std::max(std::fabs(v1), std::fabs(v2));
                    if (maxv == 0.0)
                        continue;
                    REQUIRE(std::fabs(v1 - v2) / maxv < tol);
                }
            }
        }
    }

    void check_reduced_centroids(const pr::ndview<float_t, 2>& data,
                                 const pr::ndview<std::int32_t, 2>& labels,
                                 const pr::ndview<float_t, 2>& centroids,
                                 const pr::ndview<std::int32_t, 1>& counters,
                                 float_t tol) {
        auto row_count = data.get_shape()[0];
        auto column_count = data.get_shape()[1];
        auto cluster_count = centroids.get_shape()[0];
        auto temp_centroids =
            pr::ndarray<float_t, 2>::empty(this->get_queue(), { cluster_count, column_count });
        temp_centroids.fill(this->get_queue(), 0.0).wait_and_throw();
        auto temp_centroids_ptr = temp_centroids.get_mutable_data();
        auto data_ptr = data.get_data();
        auto labels_ptr = labels.get_data();
        for (std::int64_t i = 0; i < row_count; i++) {
            const auto cl = labels_ptr[i];
            REQUIRE(cl >= 0);
            REQUIRE(cl < cluster_count);
            for (std::int64_t j = 0; j < column_count; j++) {
                temp_centroids_ptr[cl * column_count + j] += data_ptr[i * column_count + j];
            }
        }
        auto centroids_ptr = centroids.get_data();
        auto counters_ptr = counters.get_data();
        for (std::int64_t i = 0; i < cluster_count; i++) {
            if (counters_ptr[i] == 0)
                continue;
            for (std::int64_t j = 0; j < column_count; j++) {
                CAPTURE(i, j);
                auto v1 = centroids_ptr[i * column_count + j];
                auto v2 = temp_centroids_ptr[i * column_count + j] / counters_ptr[i];
                CAPTURE(v1, v2);
                auto maxv = std::max(std::fabs(v1), std::fabs(v2));
                if (maxv == 0.0)
                    continue;
                REQUIRE(std::fabs(v1 - v2) / maxv < tol);
            }
        }
    }
};

using kmeans_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(kmeans_impl_test,
                     "objective function unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    using float_t = TestType;

    std::int64_t row_count = 100001;

    const auto df =
        GENERATE_DATAFRAME(te::dataframe_builder{ row_count, 1 }.fill_uniform(0.0, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = pr::ndarray<float_t, 2>::wrap(df_rows.get_data(), { row_count, 1 });
    this->run_obj_func_check(data_array);
}

TEMPLATE_LIST_TEST_M(kmeans_impl_test,
                     "label counting unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    std::int64_t row_count = 100001;
    std::int64_t cluster_count = 37;

    const auto dfl = GENERATE_DATAFRAME(
        te::dataframe_builder{ row_count, 1 }.fill_uniform(0, cluster_count - 1));
    const table dfl_table = dfl.get_table(this->get_homogen_table_id());
    const auto dfl_rows =
        row_accessor<const std::int32_t>(dfl_table).pull(this->get_queue(), { 0, -1 });
    auto labels = pr::ndarray<std::int32_t, 2>::wrap(dfl_rows.get_data(), { row_count, 1 });
    this->run_counting(labels, cluster_count);
}

TEMPLATE_LIST_TEST_M(kmeans_impl_test,
                     "partial reduction unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    using float_t = TestType;

    std::int64_t row_count = 1001;
    std::int64_t cluster_count = 37;
    std::int64_t column_count = 17;
    std::int64_t part_count = 16;

    const auto dfl = GENERATE_DATAFRAME(
        te::dataframe_builder{ row_count, 1 }.fill_uniform(0, cluster_count - 1 + 0.5));
    const table dfl_table = dfl.get_table(this->get_homogen_table_id());
    const auto dfl_rows =
        row_accessor<const std::int32_t>(dfl_table).pull(this->get_queue(), { 0, -1 });
    auto labels = pr::ndarray<std::int32_t, 2>::wrap(dfl_rows.get_data(), { row_count, 1 });

    const auto dfd = GENERATE_DATAFRAME(
        te::dataframe_builder{ row_count, column_count }.fill_uniform(-0.9, 1.7));
    const table dfd_table = dfd.get_table(this->get_homogen_table_id());
    const auto dfd_rows = row_accessor<const float_t>(dfd_table).pull(this->get_queue(), { 0, -1 });
    auto data = pr::ndarray<float_t, 2>::wrap(dfd_rows.get_data(), { row_count, column_count });

    this->run_partial_reduce(data, labels, cluster_count, part_count);
}

TEMPLATE_LIST_TEST_M(kmeans_impl_test,
                     "centroid reduction unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    using float_t = TestType;

    std::int64_t row_count = 10001;
    std::int64_t cluster_count = 37;
    std::int64_t column_count = 33;
    std::int64_t part_count = 16;

    const auto dfl = GENERATE_DATAFRAME(
        te::dataframe_builder{ row_count, 1 }.fill_uniform(0, cluster_count - 1 + 0.5));
    const table dfl_table = dfl.get_table(this->get_homogen_table_id());
    const auto dfl_rows =
        row_accessor<const std::int32_t>(dfl_table).pull(this->get_queue(), { 0, -1 });
    auto labels = pr::ndarray<std::int32_t, 2>::wrap(dfl_rows.get_data(), { row_count, 1 });

    const auto dfd = GENERATE_DATAFRAME(
        te::dataframe_builder{ row_count, column_count }.fill_uniform(-0.9, 1.7));
    const table dfd_table = dfd.get_table(this->get_homogen_table_id());
    const auto dfd_rows = row_accessor<const float_t>(dfd_table).pull(this->get_queue(), { 0, -1 });
    auto data = pr::ndarray<float_t, 2>::wrap(dfd_rows.get_data(), { row_count, column_count });

    this->run_reduce_centroids(data, labels, cluster_count, part_count);
}

TEMPLATE_LIST_TEST_M(kmeans_impl_test,
                     "cluster assignment unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    using float_t = TestType;

    std::int64_t row_count = 10; //10001;
    std::int64_t cluster_count = 3; //37;
    std::int64_t column_count = 2; //33;
    std::int64_t block_rows = 4;

    const auto dfc = GENERATE_DATAFRAME(
        te::dataframe_builder{ cluster_count, column_count }.fill_uniform(-1.0, 2.7));
    const table dfc_table = dfc.get_table(this->get_homogen_table_id());
    const auto dfc_rows = row_accessor<const float_t>(dfc_table).pull(this->get_queue(), { 0, -1 });
    auto centroids =
        pr::ndarray<float_t, 2>::wrap(dfc_rows.get_data(), { cluster_count, column_count });

    const auto dfd = GENERATE_DATAFRAME(
        te::dataframe_builder{ row_count, column_count }.fill_uniform(-0.9, 1.7));
    const table dfd_table = dfd.get_table(this->get_homogen_table_id());
    const auto dfd_rows = row_accessor<const float_t>(dfd_table).pull(this->get_queue(), { 0, -1 });
    auto data = pr::ndarray<float_t, 2>::wrap(dfd_rows.get_data(), { row_count, column_count });

    this->run_selection(data, centroids, block_rows);
}

TEMPLATE_LIST_TEST_M(kmeans_impl_test,
                     "bigger cluster assignment unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    using float_t = TestType;

    std::int64_t row_count = 10001;
    std::int64_t cluster_count = 37;
    std::int64_t column_count = 33;
    std::int64_t block_rows = 1024;

    const auto dfc = GENERATE_DATAFRAME(
        te::dataframe_builder{ cluster_count, column_count }.fill_uniform(-1.0, 2.7));
    const table dfc_table = dfc.get_table(this->get_homogen_table_id());
    const auto dfc_rows = row_accessor<const float_t>(dfc_table).pull(this->get_queue(), { 0, -1 });
    auto centroids =
        pr::ndarray<float_t, 2>::wrap(dfc_rows.get_data(), { cluster_count, column_count });

    const auto dfd = GENERATE_DATAFRAME(
        te::dataframe_builder{ row_count, column_count }.fill_uniform(-0.9, 1.7));
    const table dfd_table = dfd.get_table(this->get_homogen_table_id());
    const auto dfd_rows = row_accessor<const float_t>(dfd_table).pull(this->get_queue(), { 0, -1 });
    auto data = pr::ndarray<float_t, 2>::wrap(dfd_rows.get_data(), { row_count, column_count });

    this->run_selection(data, centroids, block_rows);
}

TEMPLATE_LIST_TEST_M(kmeans_impl_test,
                     "candidate search unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    using float_t = TestType;

    std::int64_t element_count = 10001;
    std::int64_t candidate_count = 17;

    auto closest_distances =
        pr::ndarray<float_t, 2>::empty(this->get_queue(), { element_count, 1 });
    this->fill_uniform(closest_distances, 0.0, 1.75);

    this->run_candidates(closest_distances, candidate_count);
}

} // namespace oneapi::dal::kmeans::backend::test
