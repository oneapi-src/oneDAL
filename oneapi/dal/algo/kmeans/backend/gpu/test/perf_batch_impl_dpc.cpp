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

#include <type_traits>
#include <tuple>

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_integral.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans::backend::test {

namespace te = dal::test::engine;

// template <typename TestType>
// class kmeans_perf_test : public te::policy_fixture {
// public:
//     using float_t = TestType;

//     te::table_id get_homogen_table_id() const {
//         return te::table_id::homogen<float_t>();
//     }

//     void run_objective_function(const pr::ndview<float_t, 2>& closest_distances) {
//         INFO("benchmark objective function");
//         const auto name = fmt::format("Objective function: val_type {}, elem_count {}",
//                                       te::type2str<float_t>::name(),
//                                       closest_distances.get_count());
//         auto obj_func =
//             pr::ndarray<float_t, 1>::empty(this->get_queue(), 1, sycl::usm::alloc::device);
//         BENCHMARK(name.c_str()) {
//             kernels_fp<float_t>::compute_objective_function(this->get_queue(),
//                                                             closest_distances,
//                                                             obj_func)
//                 .wait_and_throw();
//         };
//     }

//     void run_cluster_counters(const pr::ndview<std::int32_t, 2>& responses, std::int64_t cluster_count) {
//         INFO("benchmark cluster counters");
//         const auto name =
//             fmt::format("Cluster counters: val_type {}, cluster_count {}, elem_count {}",
//                         te::type2str<float_t>::name(),
//                         cluster_count,
//                         responses.get_count());
//         auto counters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), cluster_count);
//         auto empty_clusters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), 1);
//         counters.fill(this->get_queue(), 0).wait_and_throw();
//         BENCHMARK(name.c_str()) {
//             auto event = count_clusters(this->get_queue(), responses, cluster_count, counters);
//             count_empty_clusters(this->get_queue(),
//                                  cluster_count,
//                                  counters,
//                                  empty_clusters,
//                                  { event })
//                 .wait_and_throw();
//         };
//     }

//     void run_reduce_centroids(const pr::ndview<float_t, 2>& data,
//                               const pr::ndview<std::int32_t, 2>& responses,
//                               std::int64_t cluster_count,
//                               std::int64_t part_count) {
//         INFO("benchmark centroid reduction");
//         const auto name = fmt::format(
//             "Centroid reduction: val_type {}, cluster_count {}, part_count {}, elem_count {}",
//             te::type2str<float_t>::name(),
//             cluster_count,
//             part_count,
//             responses.get_count());

//         auto column_count = data.get_shape()[1];
//         auto centroids =
//             pr::ndarray<float_t, 2>::empty(this->get_queue(), { cluster_count, column_count });
//         centroids.fill(this->get_queue(), 0).wait_and_throw();
//         auto partial_centroids =
//             pr::ndarray<float_t, 2>::empty(this->get_queue(),
//                                            { cluster_count * part_count, column_count });
//         partial_centroids.fill(this->get_queue(), 0).wait_and_throw();
//         auto counters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), cluster_count);
//         counters.fill(this->get_queue(), 0).wait_and_throw();
//         auto empty_clusters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(), 1);
//         auto event = count_clusters(this->get_queue(), responses, cluster_count, counters);
//         count_empty_clusters(this->get_queue(), cluster_count, counters, empty_clusters, { event })
//             .wait_and_throw();
//         BENCHMARK(name.c_str()) {
//             auto partial_reduce_event =
//                 kernels_fp<float_t>::partial_reduce_centroids(this->get_queue(),
//                                                               data,
//                                                               responses,
//                                                               cluster_count,
//                                                               part_count,
//                                                               partial_centroids);
//             kernels_fp<float_t>::merge_reduce_centroids(this->get_queue(),
//                                                         counters,
//                                                         partial_centroids,
//                                                         part_count,
//                                                         centroids,
//                                                         { partial_reduce_event })
//                 .wait_and_throw();
//         };
//     }
// };

// using kmeans_types = std::tuple<float, double>;

// TEMPLATE_LIST_TEST_M(kmeans_perf_test,
//                      "objective function perf test",
//                      "[kmeans][weekly][perf]",
//                      kmeans_types) {
//     using float_t = TestType;

//     std::int64_t row_count = 1024 * 1024;

//     const auto df =
//         GENERATE_DATAFRAME(te::dataframe_builder{ row_count, 1 }.fill_uniform(0.0, 0.5));
//     const table df_table = df.get_table(this->get_homogen_table_id());
//     const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
//     auto data_array = pr::ndarray<float_t, 2>::wrap(df_rows.get_data(), { row_count, 1 });
//     this->run_objective_function(data_array);
// }

// TEMPLATE_LIST_TEST_M(kmeans_perf_test,
//                      "cluster counters perf test (cluster_count == 2)",
//                      "[kmeans][weekly][perf]",
//                      kmeans_types) {
//     std::int64_t row_count = 1024 * 1024;
//     std::int64_t cluster_count = 2;

//     const auto dfl = GENERATE_DATAFRAME(
//         te::dataframe_builder{ row_count, 1 }.fill_uniform(0, cluster_count - 1));
//     const table dfl_table = dfl.get_table(this->get_homogen_table_id());
//     const auto dfl_rows =
//         row_accessor<const std::int32_t>(dfl_table).pull(this->get_queue(), { 0, -1 });
//     auto responses = pr::ndarray<std::int32_t, 2>::wrap(dfl_rows.get_data(), { row_count, 1 });
//     this->run_cluster_counters(responses, cluster_count);
// }

// TEMPLATE_LIST_TEST_M(kmeans_perf_test,
//                      "cluster counters perf test (cluster_count == 2048)",
//                      "[kmeans][weekly][perf]",
//                      kmeans_types) {
//     std::int64_t row_count = 1024 * 1024;
//     std::int64_t cluster_count = 2048;

//     const auto dfl = GENERATE_DATAFRAME(
//         te::dataframe_builder{ row_count, 1 }.fill_uniform(0, cluster_count - 1));
//     const table dfl_table = dfl.get_table(this->get_homogen_table_id());
//     const auto dfl_rows =
//         row_accessor<const std::int32_t>(dfl_table).pull(this->get_queue(), { 0, -1 });
//     auto responses = pr::ndarray<std::int32_t, 2>::wrap(dfl_rows.get_data(), { row_count, 1 });
//     this->run_cluster_counters(responses, cluster_count);
// }

// TEMPLATE_LIST_TEST_M(kmeans_perf_test,
//                      "centroid reduction perf test",
//                      "[kmeans][weekly][perf]",
//                      kmeans_types) {
//     using float_t = TestType;

//     std::int64_t row_count = 80000;
//     std::int64_t cluster_count = 2048;
//     std::int64_t column_count = 3000;
//     std::int64_t part_count = 64;

//     const auto dfl = GENERATE_DATAFRAME(
//         te::dataframe_builder{ row_count, 1 }.fill_uniform(0, cluster_count - 1 + 0.5));
//     const table dfl_table = dfl.get_table(this->get_homogen_table_id());
//     const auto dfl_rows =
//         row_accessor<const std::int32_t>(dfl_table).pull(this->get_queue(), { 0, -1 });
//     auto responses = pr::ndarray<std::int32_t, 2>::wrap(dfl_rows.get_data(), { row_count, 1 });

//     const auto dfd = GENERATE_DATAFRAME(
//         te::dataframe_builder{ row_count, column_count }.fill_uniform(-0.9, 1.7));
//     const table dfd_table = dfd.get_table(this->get_homogen_table_id());
//     const auto dfd_rows = row_accessor<const float_t>(dfd_table).pull(this->get_queue(), { 0, -1 });
//     auto data = pr::ndarray<float_t, 2>::wrap(dfd_rows.get_data(), { row_count, column_count });

//     this->run_reduce_centroids(data, responses, cluster_count, part_count);
// }

} // namespace oneapi::dal::kmeans::backend::test
