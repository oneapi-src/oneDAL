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

#include "oneapi/dal/algo/kmeans/test/data.hpp"
#include "oneapi/dal/algo/kmeans/test/fixture.hpp"
#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::kmeans::test {

template <typename TestType>
class kmeans_spmd_test : public kmeans_test<TestType, kmeans_spmd_test<TestType>> {
public:
    using base_t = kmeans_test<TestType, kmeans_spmd_test<TestType>>;
    using float_t = typename base_t::float_t;
    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename... Args>
    train_result_t train_override(Args&&... args) {
        return this->train_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::vector<train_input_t> split_train_input_override(std::int64_t split_count,
                                                          Args&&... args) {
        // Data table is distributed across the ranks, but
        // initial centroids are common for all the ranks
        const train_input_t input{ std::forward<Args>(args)... };

        const auto split_data =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);
        const auto common_centroids = input.get_initial_centroids();

        std::vector<train_input_t> split_input;
        split_input.reserve(split_count);

        for (std::int64_t i = 0; i < split_count; i++) {
            split_input.push_back( //
                train_input_t{ split_data[i] } //
                    .set_initial_centroids(common_centroids) //
            );
        }

        return split_input;
    }

    train_result_t merge_train_result_override(const std::vector<train_result_t>& results) {
        // Responses are distributed accross the ranks, we combine them into one table;
        // Model, iteration_count, objective_function_value are the same for all ranks

        std::vector<table> responses;
        for (const auto& r : results) {
            responses.push_back(r.get_responses());
        }
        const auto full_responses = te::stack_tables_by_rows<float_t>(responses);

        return train_result_t{} //
            .set_responses(full_responses) //
            .set_model(results[0].get_model()) //
            .set_iteration_count(results[0].get_iteration_count()) //
            .set_objective_function_value(results[0].get_objective_function_value());
    }

    void check_if_results_same_on_all_ranks() {
        const auto table_id = this->get_homogen_table_id();
        const auto data = gold_dataset::get_data().get_table(table_id);
        const auto initial_centroids = gold_dataset::get_initial_centroids().get_table(table_id);

        const std::int64_t cluster_count = gold_dataset::get_cluster_count();
        const std::int64_t max_iteration_count = 100;
        const float_t accuracy_threshold = 0.0;

        INFO("create descriptor");
        const auto kmeans_desc =
            this->get_descriptor(cluster_count, max_iteration_count, accuracy_threshold);

        INFO("run training");
        const auto train_results =
            this->train_via_spmd_threads(rank_count_, kmeans_desc, data, initial_centroids);

        SECTION("check if all results bitwise equal on all ranks") {
            ONEDAL_ASSERT(train_results.size() > 0);
            const auto front_centroids = train_results.front().get_model().get_centroids();
            const auto front_iteration_count = train_results.front().get_iteration_count();
            const auto front_objective = train_results.front().get_objective_function_value();

            for (const auto& result : train_results) {
                // We do not check responses as they are expected
                // to be different on each ranks

                SECTION("check centroids") {
                    const auto centroids = result.get_model().get_centroids();
                    te::check_if_tables_equal<float_t>(centroids, front_centroids);
                }

                SECTION("check iterations") {
                    REQUIRE(result.get_iteration_count() == front_iteration_count);
                }

                SECTION("check objective function") {
                    REQUIRE(result.get_objective_function_value() == front_objective);
                }
            }
        }
    }

private:
    std::int64_t rank_count_ = 1;
};

TEMPLATE_LIST_TEST_M(kmeans_spmd_test,
                     "make sure results are the same on all ranks",
                     "[spmd][smoke]",
                     kmeans_types) {
    // SPMD mode is not implemented for CPU. The following `SKIP_IF` should be
    // removed once it's supported for CPU. The same for the rest of tests cases.
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    this->set_rank_count(GENERATE(2, 4));
    this->check_if_results_same_on_all_ranks();
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_test,
                     "distributed kmeans empty clusters test",
                     "[spmd][smoke]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    this->set_rank_count(GENERATE(1, 2));
    this->check_empty_clusters();
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_test,
                     "distributed kmeans smoke train/infer test",
                     "[spmd][smoke]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    this->set_rank_count(GENERATE(1, 2));
    this->check_on_smoke_data();
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_test,
                     "distributed kmeans train/infer on gold data",
                     "[spmd][smoke]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    this->set_rank_count(GENERATE(1, 2, 4, 8));
    this->check_on_gold_data();
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_test,
                     "distributed kmeans block test",
                     "[spmd][block][nightly]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    this->set_rank_count(GENERATE(1, 8));
    this->check_on_large_data_with_one_cluster();
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_test,
                     "distributed higgs: samples=1M, iters=3",
                     "[kmeans][spmd][higgs][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    this->set_rank_count(10);
    const std::int64_t iters = 3;
    const std::string higgs_path = "workloads/higgs/dataset/higgs_1m_test.csv";

    SECTION("clusters=10") {
        this->test_on_dataset(higgs_path, 10, iters, 3.1997724684, 14717484.0);
    }

    SECTION("clusters=100") {
        this->test_on_dataset(higgs_path, 100, iters, 2.7450205195, 10704352.0);
    }

    SECTION("cluster=250") {
        this->test_on_dataset(higgs_path, 250, iters, 2.5923397174, 9335216.0);
    }
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_test,
                     "distributed susy: samples=0.5M, iters=10",
                     "[kmeans][nightly][spmd][susy][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    this->set_rank_count(10);
    const std::int64_t iters = 10;
    const std::string susy_path = "workloads/susy/dataset/susy_test.csv";

    SECTION("clusters=10") {
        this->test_on_dataset(susy_path, 10, iters, 1.7730860782, 3183696.0);
    }

    SECTION("clusters=100") {
        this->test_on_dataset(susy_path, 100, iters, 1.9384844916, 1757022.625);
    }

    SECTION("cluster=250") {
        this->test_on_dataset(susy_path, 250, iters, 1.8950113604, 1400958.5);
    }
}

TEMPLATE_LIST_TEST_M(kmeans_spmd_test,
                     "distributed epsilon: samples=80K, iters=2",
                     "[kmeans][nightly][spmd][epsilon][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->is_sparse_method());

    this->set_rank_count(10);
    const std::int64_t iters = 2;
    const std::string epsilon_path = "workloads/epsilon/dataset/epsilon_80k_train.csv";

    SECTION("clusters=512") {
        this->test_on_dataset(epsilon_path, 512, iters, 6.9367580565, 50128.640625, 1.0e-3);
    }

    // Disabled due to an issue
    /*
    SECTION("clusters=1024") {
        this->test_on_dataset(epsilon_path, 1024, iters, 5.59003873, 49518.75, 1.0e-3);
    }

    SECTION("cluster=2048") {
        this->test_on_dataset(epsilon_path, 2048, iters, 4.3202752143, 48437.6015625, 1.0e-3);
    }
    */
}

} // namespace oneapi::dal::kmeans::test
