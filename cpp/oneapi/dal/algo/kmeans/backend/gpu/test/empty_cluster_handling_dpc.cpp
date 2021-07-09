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

#include "oneapi/dal/algo/kmeans/backend/gpu/empty_cluster_handling.hpp"

namespace oneapi::dal::kmeans::backend::test {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;
namespace te = dal::test::engine;

template <typename TestType>
class empty_cluster_handling_test : public te::float_algo_fixture<TestType> {
public:
    using float_t = TestType;

    void fill_uniform(pr::ndarray<float_t, 2>& val, float_t a, float_t b, int seed = 7777) {
        ONEDAL_ASSERT(b > a);

        const std::int64_t elem_count = val.get_count();
        ONEDAL_ASSERT(elem_count > 0);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float_t> distr(a, b);

        auto val_host = bk::make_unique_host<float_t>(elem_count);
        float_t* val_ptr = val_host.get();

        for (std::int64_t el = 0; el < elem_count; el++) {
            val_ptr[el] = distr(rng);
        }

        val.assign(this->get_queue(), val_ptr, elem_count).wait_and_throw();
    }

    void fill_counters_uniform(pr::ndarray<std::int32_t, 1>& counters,
                               std::int64_t a,
                               std::int64_t b,
                               std::int64_t empty_cluster_count,
                               int seed = 7777) {
        ONEDAL_ASSERT(a > 0);
        ONEDAL_ASSERT(b > a);

        const std::int64_t cluster_count = counters.get_count();
        ONEDAL_ASSERT(cluster_count > 0);
        ONEDAL_ASSERT(empty_cluster_count <= cluster_count);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<std::int64_t> distr(a, b);

        auto val_host = bk::make_unique_host<std::int32_t>(cluster_count);
        std::int32_t* val_ptr = val_host.get();
        for (std::int64_t el = 0; el < cluster_count; el++) {
            val_ptr[el] = distr(rng);
        }

        auto indices_host = bk::make_unique_host<std::int32_t>(cluster_count);
        std::int32_t* indices_ptr = indices_host.get();
        for (std::int64_t i = 0; i < cluster_count; i++) {
            indices_ptr[i] = std::int32_t(i);
        }

        std::shuffle(indices_ptr, indices_ptr + cluster_count, rng);

        for (std::int64_t i = 0; i < empty_cluster_count; i++) {
            val_ptr[indices_ptr[i]] = 0;
        }

        counters.assign(this->get_queue(), val_ptr, cluster_count).wait_and_throw();
    }

    void run_find_candidates(const pr::ndarray<float_t, 2>& closest_distances,
                             const pr::ndarray<std::int32_t, 1>& counters,
                             std::int64_t candidate_count) {
        auto [candidates, find_candidates_event] = find_candidates( //
            this->get_queue(),
            candidate_count,
            closest_distances,
            counters);
        find_candidates_event.wait_and_throw();

        check_candidates(closest_distances, candidates);
    }

    void run_fill_empty_clusters(bk::spmd_communicator& comm,
                                 std::int64_t cluster_count,
                                 const pr::ndarray<float_t, 2>& data,
                                 const centroid_candidates<Float>& candidates) {
        const std::int64_t column_count = data.get_dimension(1);

        auto centroids = pr::ndarray<float_t, 2>::empty( //
            this->get_queue(),
            { cluster_count, column_count },
            sycl::usm::alloc::device);

        fill_empty_clusters(this->get_queue(), comm, data, candidates, centroids).wait_and_throw();
    }

    void check_candidates(const pr::ndarray<float_t, 2>& closest_distances,
                          const centroid_candidates<float_t>& candidates) {
        const std::int64_t candidate_count = candidates.get_candidate_count();
        const std::int64_t elem_count = closest_distances.get_dimension(0);

        const auto host_closest_distances = closest_distances.to_host(this->get_queue());
        const auto host_candidate_indices = candidates.get_indices().to_host(this->get_queue());
        const auto host_candidate_distances = candidates.get_distances().to_host(this->get_queue());

        auto closest_distances_ptr = host_closest_distances.get_data();
        auto candidate_indices_ptr = host_candidate_indices.get_data();
        auto candidate_distances_ptr = host_candidate_distances.get_data();

        for (std::int64_t i = 0; i < candidate_count; i++) {
            auto distance = candidate_distances_ptr[i];
            auto index = candidate_indices_ptr[i];
            std::int64_t count = 0;
            for (std::int64_t j = 0; j < elem_count; j++) {
                auto cur_val = -1.0 * closest_distances_ptr[j];
                if (cur_val >= distance)
                    count++;
            }
            CAPTURE(i, count);
            REQUIRE(count <= candidate_count);
            REQUIRE(index >= 0);
            REQUIRE(index < elem_count);
        }
    }
};

using kmeans_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(empty_cluster_handling_test, "find candidates", "[candidates]", kmeans_types) {
    using float_t = TestType;
    SKIP_IF(this->not_float64_friendly());

    const auto [cluster_count, candidate_count] =
        GENERATE(std::tuple<std::int64_t, std::int64_t>{ 11, 2 },
                 std::tuple<std::int64_t, std::int64_t>{ 1001, 17 });

    auto closest_distances = pr::ndarray<float_t, 2>::empty(this->get_queue(),
                                                            { cluster_count, 1 },
                                                            sycl::usm::alloc::device);
    this->fill_uniform(closest_distances, 0.0, 1.75);

    auto counters = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(),
                                                        { cluster_count },
                                                        sycl::usm::alloc::device);
    this->fill_counters_uniform(counters, 10, 100, candidate_count);

    this->test_find_candidates(closest_distances, counters, candidate_count);
}

TEMPLATE_LIST_TEST_M(empty_cluster_handling_test,
                     "fill empty clusters local",
                     "[fill]",
                     kmeans_types) {
    bk::spmd_communicator fake_comm;

    fill_empty_clusters(this->get_queue(), fake_comm, )
}

} // namespace oneapi::dal::kmeans::backend::test
