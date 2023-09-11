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

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

#include "oneapi/dal/algo/kmeans/backend/gpu/cluster_updater.hpp"

namespace oneapi::dal::kmeans::backend::test {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
namespace te = dal::test::engine;

template <typename TestType>
class cluster_updater_test : public te::float_algo_fixture<TestType> {
public:
    using float_t = TestType;

    template <std::int64_t dim>
    pr::ndarray<float_t, dim> make_uniform(const pr::ndshape<dim>& shape,
                                           float_t a,
                                           float_t b,
                                           int seed = 7777) {
        ONEDAL_ASSERT(b > a);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float_t> distr(a, b);

        auto val = pr::ndarray<float_t, dim>::empty(shape);
        float_t* val_ptr = val.get_mutable_data();

        for (std::int64_t el = 0; el < shape.get_count(); el++) {
            val_ptr[el] = distr(rng);
        }

        return val.to_device(this->get_queue());
    }

    template <typename Integer, std::int64_t dim>
    pr::ndarray<Integer, dim> make_uniform_int(const pr::ndshape<dim>& shape,
                                               Integer a,
                                               Integer b,
                                               int seed = 7777) {
        ONEDAL_ASSERT(b > a);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<std::int32_t> distr(a, b);

        auto val = pr::ndarray<Integer, dim>::empty(shape);
        Integer* val_ptr = val.get_mutable_data();

        for (std::int64_t el = 0; el < shape.get_count(); el++) {
            val_ptr[el] = Integer(distr(rng));
        }

        return val.to_device(this->get_queue());
    }

    void run_squares_check(const pr::ndview<float_t, 2>& data, float_t tol = 1.0e-7) {
        const auto row_count = data.get_dimension(0);
        auto squares = pr::ndarray<float_t, 1>::empty(this->get_queue(), row_count);
        kernels_fp<float_t>::compute_squares(this->get_queue(), data, squares).wait_and_throw();
        check_squares(data, squares, tol);
    }

    void check_squares(const pr::ndview<float_t, 2>& data,
                       const pr::ndview<float_t, 1>& squares,
                       float_t tol) {
        auto row_count = data.get_dimension(0);
        auto column_count = data.get_dimension(1);
        auto data_ptr = data.get_data();
        auto squares_ptr = squares.get_data();
        ONEDAL_ASSERT(row_count = squares.get_dimension(0));
        for (std::int64_t i = 0; i < row_count; i++) {
            float_t sum = 0.0;
            for (std::int64_t j = 0; j < column_count; j++) {
                float_t val = data_ptr[i * column_count + j];
                sum += val * val;
            }
            float_t ref_sum = squares_ptr[i];
            if (ref_sum == float_t(0)) {
                REQUIRE(sum == float_t(0));
                continue;
            }
            CAPTURE(sum, ref_sum);
            REQUIRE(std::fabs(sum - ref_sum) / std::max(std::fabs(sum), std::fabs(ref_sum)) < tol);
        }
    }

    void run_obj_func_check(const pr::ndarray<float_t, 2>& closest_distances,
                            float_t tol = 1.0e-5) {
        auto obj_func = pr::ndarray<float_t, 1>::empty( //
            this->get_queue(),
            { 1 },
            sycl::usm::alloc::device);

        kernels_fp<float_t>::compute_objective_function(this->get_queue(),
                                                        closest_distances,
                                                        obj_func)
            .wait_and_throw();

        const float_t obj_func_value = obj_func.to_host(this->get_queue()).get_data()[0];
        check_objective_function(closest_distances, obj_func_value, tol);
    }

    void check_objective_function(const pr::ndarray<float_t, 2>& closest_distances,
                                  float_t objective_function_value,
                                  float_t tol) {
        const std::int64_t row_count = closest_distances.get_dimension(0);
        const auto host_closest_distances = closest_distances.to_host(this->get_queue());
        const float_t* min_distance_ptr = host_closest_distances.get_data();

        const float_t sum =
            std::accumulate(min_distance_ptr, min_distance_ptr + row_count, float_t(0));

        CAPTURE(sum, objective_function_value);
        REQUIRE(std::fabs(sum - objective_function_value) /
                    std::max(std::fabs(sum), std::fabs(objective_function_value)) <
                tol);
    }

    void run_counting(const pr::ndarray<std::int32_t, 2>& responses, std::int64_t cluster_count) {
        auto [counters, counters_event] = pr::ndarray<std::int32_t, 1>::zeros( //
            this->get_queue(),
            { cluster_count },
            sycl::usm::alloc::device);

        count_clusters(this->get_queue(), responses, cluster_count, counters, { counters_event })
            .wait_and_throw();

        const std::int64_t empty_cluster_count =
            count_empty_clusters(this->get_queue(), cluster_count, counters);

        check_counters(responses, counters, cluster_count, empty_cluster_count);
    }

    void run_partial_reduce(const pr::ndarray<float_t, 2>& data,
                            const pr::ndarray<std::int32_t, 2>& responses,
                            std::int64_t cluster_count,
                            std::int64_t part_count,
                            float_t tol = 1.0e-5) {
        const std::int64_t column_count = data.get_dimension(1);

        auto [centroids, centroids_event] =
            pr::ndarray<float_t, 2>::zeros(this->get_queue(),
                                           { cluster_count, column_count },
                                           sycl::usm::alloc::device);

        auto [partial_centroids, partial_centroids_event] =
            pr::ndarray<float_t, 2>::zeros(this->get_queue(),
                                           { cluster_count * part_count, column_count },
                                           sycl::usm::alloc::device);

        auto [counters, counters_event] =
            pr::ndarray<std::int32_t, 1>::zeros(this->get_queue(),
                                                { cluster_count },
                                                sycl::usm::alloc::device);

        count_clusters(this->get_queue(),
                       responses,
                       cluster_count,
                       counters,
                       { centroids_event, partial_centroids_event, counters_event })
            .wait_and_throw();

        kernels_fp<float_t>::partial_reduce_centroids(this->get_queue(),
                                                      data,
                                                      responses,
                                                      cluster_count,
                                                      part_count,
                                                      partial_centroids)
            .wait_and_throw();

        check_partial_centroids(data, responses, partial_centroids, part_count, tol);
    }

    void run_reduce_centroids(const pr::ndarray<float_t, 2>& data,
                              const pr::ndarray<std::int32_t, 2>& responses,
                              std::int64_t cluster_count,
                              std::int64_t part_count,
                              float_t tol = 1.0e-5) {
        const std::int64_t column_count = data.get_dimension(1);

        auto [centroids, centroids_event] =
            pr::ndarray<float_t, 2>::zeros(this->get_queue(),
                                           { cluster_count, column_count },
                                           sycl::usm::alloc::device);

        auto [partial_centroids, partial_centroids_event] =
            pr::ndarray<float_t, 2>::zeros(this->get_queue(),
                                           { cluster_count * part_count, column_count },
                                           sycl::usm::alloc::device);

        auto [counters, counters_event] =
            pr::ndarray<std::int32_t, 1>::zeros(this->get_queue(),
                                                { cluster_count },
                                                sycl::usm::alloc::device);

        count_clusters(this->get_queue(),
                       responses,
                       cluster_count,
                       counters,
                       { centroids_event, partial_centroids_event, counters_event })
            .wait_and_throw();

        kernels_fp<float_t>::partial_reduce_centroids(this->get_queue(),
                                                      data,
                                                      responses,
                                                      cluster_count,
                                                      part_count,
                                                      partial_centroids)
            .wait_and_throw();

        kernels_fp<float_t>::merge_reduce_centroids(this->get_queue(),
                                                    counters,
                                                    partial_centroids,
                                                    part_count,
                                                    centroids)
            .wait_and_throw();

        check_reduced_centroids(data, responses, centroids, counters, tol);
    }

    void run_assignments(const pr::ndarray<float_t, 2>& data,
                         const pr::ndarray<float_t, 2>& centroids,
                         std::int64_t block_rows,
                         float_t tol = 1.0e-5) {
        const std::int64_t row_count = data.get_dimension(0);
        const std::int64_t cluster_count = centroids.get_dimension(0);

        auto responses = pr::ndarray<std::int32_t, 2>::empty( //
            this->get_queue(),
            { row_count, 1 },
            sycl::usm::alloc::device);

        auto closest_distances = pr::ndarray<float_t, 2>::empty( //
            this->get_queue(),
            { row_count, 1 },
            sycl::usm::alloc::device);

        auto distances = pr::ndarray<float_t, 2>::empty( //
            this->get_queue(),
            { block_rows, cluster_count },
            sycl::usm::alloc::device);
        auto centroid_squares = pr::ndarray<float_t, 1>::empty( //
            this->get_queue(),
            cluster_count);
        kernels_fp<float_t>::compute_squares(this->get_queue(), centroids, centroid_squares)
            .wait_and_throw();
        auto data_squares = pr::ndarray<float_t, 1>::empty( //
            this->get_queue(),
            row_count);
        kernels_fp<float_t>::compute_squares(this->get_queue(), data, data_squares)
            .wait_and_throw();

        kernels_fp<float_t>::assign_clusters(this->get_queue(),
                                             data,
                                             centroids,
                                             data_squares,
                                             centroid_squares,
                                             block_rows,
                                             responses,
                                             distances,
                                             closest_distances)
            .wait_and_throw();

        check_assignments(data, centroids, responses, closest_distances, tol);
    }

    void check_assignments(const pr::ndarray<float_t, 2>& data,
                           const pr::ndarray<float_t, 2>& centroids,
                           const pr::ndarray<std::int32_t, 2>& responses,
                           const pr::ndarray<float_t, 2>& closest_distances,
                           float_t tol) {
        const std::int64_t row_count = data.get_dimension(0);
        const std::int64_t column_count = data.get_dimension(1);
        const std::int64_t cluster_count = centroids.get_dimension(0);

        const auto host_data = data.to_host(this->get_queue());
        const auto host_centroids = centroids.to_host(this->get_queue());
        const auto host_responses = responses.to_host(this->get_queue());
        const auto host_closest_distances = closest_distances.to_host(this->get_queue());

        auto data_ptr = host_data.get_data();
        auto responses_ptr = host_responses.get_data();
        auto centroids_ptr = host_centroids.get_data();
        auto closest_distances_ptr = host_closest_distances.get_data();

        for (std::int64_t i = 0; i < row_count; i++) {
            float_t min_distance = dal::detail::limits<float_t>::max();
            std::int32_t min_index = -1;
            for (std::int64_t j = 0; j < cluster_count; j++) {
                float_t distance = 0;
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
            CAPTURE(i, responses_ptr[i]);
            REQUIRE(responses_ptr[i] == min_index);
            auto v1 = closest_distances_ptr[i];
            auto v2 = min_distance;
            CAPTURE(v1, v2);
            auto maxv = std::max(std::fabs(v1), std::fabs(v2));
            if (maxv == 0.0) {
                continue;
            }
            REQUIRE(std::fabs(v1 - v2) / maxv < tol);
        }
    }

    void check_counters(const pr::ndarray<std::int32_t, 2>& responses,
                        const pr::ndarray<std::int32_t, 1>& counters,
                        std::int32_t cluster_count,
                        std::int32_t empty_cluster_count) {
        const std::int64_t row_count = responses.get_dimension(0);
        auto temp_counters = pr::ndarray<std::int32_t, 1>::zeros(cluster_count);

        const auto host_responses = responses.to_host(this->get_queue());
        const auto host_counters = counters.to_host(this->get_queue());

        auto responses_ptr = host_responses.get_data();
        auto counters_ptr = host_counters.get_data();
        auto temp_counters_ptr = temp_counters.get_mutable_data();

        for (std::int64_t i = 0; i < row_count; i++) {
            const auto cl = responses_ptr[i];
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

    void check_partial_centroids(const pr::ndarray<float_t, 2>& data,
                                 const pr::ndarray<std::int32_t, 2>& responses,
                                 const pr::ndarray<float_t, 2>& partial_centroids,
                                 std::int64_t part_count,
                                 float_t tol) {
        const std::int64_t row_count = data.get_dimension(0);
        const std::int64_t column_count = data.get_dimension(1);
        const std::int64_t cluster_count = partial_centroids.get_dimension(0) / part_count;

        auto temp_partial_centroids =
            pr::ndarray<float_t, 2>::zeros({ cluster_count * part_count, column_count });

        const auto host_data = data.to_host(this->get_queue());
        const auto host_responses = responses.to_host(this->get_queue());
        const auto host_partial_centroids = partial_centroids.to_host(this->get_queue());

        auto data_ptr = host_data.get_data();
        auto responses_ptr = host_responses.get_data();
        auto partial_centroids_ptr = host_partial_centroids.get_data();
        auto temp_partial_centroids_ptr = temp_partial_centroids.get_mutable_data();

        for (std::int64_t i = 0; i < row_count; i++) {
            const std::int64_t cl = responses_ptr[i];
            const std::int64_t part = i % part_count;
            REQUIRE(cl >= 0);
            REQUIRE(cl < cluster_count);
            for (std::int64_t j = 0; j < column_count; j++) {
                const std::int64_t part_index =
                    part * cluster_count * column_count + cl * column_count + j;
                temp_partial_centroids_ptr[part_index] += data_ptr[i * column_count + j];
            }
        }

        for (std::int64_t i = 0; i < part_count; i++) {
            for (std::int64_t k = 0; k < cluster_count; k++) {
                for (std::int64_t j = 0; j < column_count; j++) {
                    CAPTURE(i, k, j);
                    auto v1 = partial_centroids_ptr[i * column_count + j];
                    auto v2 = temp_partial_centroids_ptr[i * column_count + j];
                    CAPTURE(v1, v2);
                    auto maxv = std::max(std::fabs(v1), std::fabs(v2));
                    if (maxv == 0.0) {
                        continue;
                    }
                    REQUIRE(std::fabs(v1 - v2) / maxv < tol);
                }
            }
        }
    }

    void check_reduced_centroids(const pr::ndarray<float_t, 2>& data,
                                 const pr::ndarray<std::int32_t, 2>& responses,
                                 const pr::ndarray<float_t, 2>& centroids,
                                 const pr::ndarray<std::int32_t, 1>& counters,
                                 float_t tol) {
        const std::int64_t row_count = data.get_dimension(0);
        const std::int64_t column_count = data.get_dimension(1);
        const std::int64_t cluster_count = centroids.get_dimension(0);

        const auto host_data = data.to_host(this->get_queue());
        const auto host_responses = responses.to_host(this->get_queue());
        const auto host_centroids = centroids.to_host(this->get_queue());
        const auto host_counters = counters.to_host(this->get_queue());
        auto temp_centroids = pr::ndarray<float_t, 2>::zeros({ cluster_count, column_count });

        auto data_ptr = host_data.get_data();
        auto responses_ptr = host_responses.get_data();
        auto centroids_ptr = host_centroids.get_data();
        auto counters_ptr = host_counters.get_data();
        auto temp_centroids_ptr = temp_centroids.get_mutable_data();

        for (std::int64_t i = 0; i < row_count; i++) {
            const auto cl = responses_ptr[i];
            REQUIRE(cl >= 0);
            REQUIRE(cl < cluster_count);
            for (std::int64_t j = 0; j < column_count; j++) {
                temp_centroids_ptr[cl * column_count + j] += data_ptr[i * column_count + j];
            }
        }

        for (std::int64_t i = 0; i < cluster_count; i++) {
            if (counters_ptr[i] == 0) {
                continue;
            }
            for (std::int64_t j = 0; j < column_count; j++) {
                CAPTURE(i, j);
                auto v1 = centroids_ptr[i * column_count + j];
                auto v2 = temp_centroids_ptr[i * column_count + j] / counters_ptr[i];
                CAPTURE(v1, v2);
                auto maxv = std::max(std::fabs(v1), std::fabs(v2));
                if (maxv == 0.0) {
                    continue;
                }
                REQUIRE(std::fabs(v1 - v2) / maxv < tol);
            }
        }
    }
};

using kmeans_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(cluster_updater_test,
                     "compute squares unit test",
                     "[kmeans][weekly][unit]",
                     kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    std::int64_t row_count = 17;
    std::int64_t column_count = 1001;

    const auto data = //
        this->template make_uniform<2>({ row_count, column_count }, 0.0, 0.5);

    this->run_squares_check(data);
}

TEMPLATE_LIST_TEST_M(cluster_updater_test, "objective function", "[objective]", kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    const std::int64_t row_count = 100001;

    const auto distances = //
        this->template make_uniform<2>({ row_count, 1 }, 0.0, 0.5);

    this->run_obj_func_check(distances);
}

TEMPLATE_LIST_TEST_M(cluster_updater_test, "response counting", "[count-responses]", kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    const std::int64_t row_count = 100001;
    const std::int64_t cluster_count = 37;

    const auto responses = //
        this->template make_uniform_int<std::int32_t, 2>({ row_count, 1 }, 0, cluster_count - 1);

    this->run_counting(responses, cluster_count);
}

TEMPLATE_LIST_TEST_M(cluster_updater_test, "partial reduction", "[reduction]", kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    const std::int64_t row_count = 1001;
    const std::int64_t cluster_count = 37;
    const std::int64_t column_count = 17;
    const std::int64_t part_count = 16;

    const auto data = //
        this->template make_uniform<2>({ row_count, column_count }, -0.9, 1.7);

    const auto responses = //
        this->template make_uniform_int<std::int32_t, 2>({ row_count, 1 }, 0, cluster_count - 1);

    this->run_partial_reduce(data, responses, cluster_count, part_count);
}

TEMPLATE_LIST_TEST_M(cluster_updater_test, "centroid reduction", "[reduction]", kmeans_types) {
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t row_count = 10001;
    const std::int64_t cluster_count = 37;
    const std::int64_t column_count = 33;
    const std::int64_t part_count = 16;

    const auto data = //
        this->template make_uniform<2>({ row_count, column_count }, -0.9, 1.7);

    const auto responses = //
        this->template make_uniform_int<std::int32_t, 2>({ row_count, 1 }, 0, cluster_count - 1);

    this->run_reduce_centroids(data, responses, cluster_count, part_count);
}

TEMPLATE_LIST_TEST_M(cluster_updater_test, "cluster assignment", "[assignments]", kmeans_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());

    using config_t = std::tuple<std::int64_t, //
                                std::int64_t, //
                                std::int64_t, //
                                std::int64_t>;

    const auto [row_count, cluster_count, column_count, block_rows] =
        GENERATE(config_t{ 10, 3, 2, 4 }, //
                 config_t{ 10001, 37, 33, 1024 });

    const auto centroids = //
        this->template make_uniform<2>({ cluster_count, column_count }, -1.0, 2.7);

    const auto data = //
        this->template make_uniform<2>({ row_count, column_count }, -0.9, 1.7);

    this->run_assignments(data, centroids, block_rows);
}

} // namespace oneapi::dal::kmeans::backend::test
