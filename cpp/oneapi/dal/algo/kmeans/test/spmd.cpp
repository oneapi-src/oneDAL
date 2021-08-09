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
class kmeans_distr_test : public kmeans_test<TestType, kmeans_distr_test<TestType>> {
public:
    using base_t = kmeans_test<TestType, kmeans_distr_test<TestType>>;
    using float_t = typename base_t::float_t;
    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename... Args>
    train_result_t train_override(Args&&... args) {
        return this->spmd_train_via_threads(rank_count_, std::forward<Args>(args)...);
    }

    std::vector<train_input_t> split_train_input_override(std::int64_t split_count,
                                                          const train_input_t& input) {
        // Data table is distributed across the ranks, but
        // initial centroids are common for all the ranks

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

private:
    std::int64_t rank_count_ = 1;
};

TEMPLATE_LIST_TEST_M(kmeans_distr_test,
                     "distributed kmeans empty clusters test",
                     "[distr]",
                     kmeans_types) {
    // SPMD mode is not implemented for CPU. The following `SKIP_IF` should be
    // removed once it's supported for CPU. The same for the rest of tests cases.
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(GENERATE(1, 2));
    this->check_empty_clusters();
}

TEMPLATE_LIST_TEST_M(kmeans_distr_test,
                     "distributed kmeans smoke train/infer test",
                     "[distr]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(GENERATE(1, 2));
    this->check_on_smoke_data();
}

TEMPLATE_LIST_TEST_M(kmeans_distr_test,
                     "distributed kmeans train/infer on gold data",
                     "[distr]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(GENERATE(1, 2, 4, 8));
    this->check_on_gold_data();
}

TEMPLATE_LIST_TEST_M(kmeans_distr_test,
                     "distributed kmeans block test",
                     "[distr][block][nightly]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(GENERATE(1, 8));
    this->check_on_large_data_with_one_cluster();
}

TEMPLATE_LIST_TEST_M(kmeans_distr_test,
                     "distributed higgs: samples=1M, iters=3",
                     "[kmeans][distr][higgs][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

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

TEMPLATE_LIST_TEST_M(kmeans_distr_test,
                     "distributed susy: samples=0.5M, iters=10",
                     "[kmeans][nightly][distr][susy][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

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

TEMPLATE_LIST_TEST_M(kmeans_distr_test,
                     "distributed epsilon: samples=80K, iters=2",
                     "[kmeans][nightly][distr][epsilon][external-dataset]",
                     kmeans_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->set_rank_count(10);
    const std::int64_t iters = 2;
    const std::string epsilon_path = "workloads/epsilon/dataset/epsilon_80k_train.csv";

    SECTION("clusters=512") {
        this->test_on_dataset(epsilon_path, 512, iters, 6.9367580565, 50128.640625, 1.0e-3);
    }

    SECTION("clusters=1024") {
        this->test_on_dataset(epsilon_path, 1024, iters, 5.59003873, 49518.75, 1.0e-3);
    }

    SECTION("cluster=2048") {
        this->test_on_dataset(epsilon_path, 2048, iters, 4.3202752143, 48437.6015625, 1.0e-3);
    }
}

} // namespace oneapi::dal::kmeans::test
