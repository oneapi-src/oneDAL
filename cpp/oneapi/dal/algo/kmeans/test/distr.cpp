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
#include "oneapi/dal/test/engine/communicator.hpp"
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

        const auto split_data = te::split_table_by_rows<float_t>(input.get_data(), split_count);
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
        // Labels are distributed accross the ranks, we combine them into one table;
        // Model, iteration_count, objective_function_value are the same for all ranks

        std::vector<table> labels;
        for (const auto& r : results) {
            labels.push_back(r.get_labels());
        }
        const auto full_labels = te::stack_tables_by_rows<float_t>(labels);

        return train_result_t{} //
            .set_labels(full_labels) //
            .set_model(results[0].get_model()) //
            .set_iteration_count(results[0].get_iteration_count()) //
            .set_objective_function_value(results[0].get_objective_function_value());
    }

private:
    std::int64_t rank_count_ = 1;
};

TEMPLATE_LIST_TEST_M(kmeans_distr_test, "distributed kmeans smoke", "[distr]", kmeans_types) {
    SKIP_IF(this->not_float64_friendly());

    using float_t = std::tuple_element_t<0, TestType>;

    float_t data[] = { 0.0, 5.0, 0.0, 0.0, 0.0, //
                       1.0, 1.0, 4.0, 0.0, 0.0, //
                       1.0, 0.0, 0.0, 5.0, 1.0 };
    float_t labels[] = { 0, //
                         1, //
                         2 };

    const auto x = homogen_table::wrap(data, 3, 5);
    const auto y = homogen_table::wrap(labels, 3, 1);

    this->set_rank_count(GENERATE(1, 2, 3));
    this->exact_checks(x, x, x, y, 3, 2, 0.0, 0.0, false);
}

// const std::int64_t thread_count = GENERATE(1, 2, 4, 8, 16);
// auto thread_comm = te::thread_communicator{ thread_count };
// auto host_spmd_policy = dal::detail::spmd_policy{ dal::detail::host_policy{}, thread_comm };

// const auto data_df = gold_dataset::get_data();
// const auto data_df_chunks = data_df.split(thread_count);
// const auto initial_centroids_df = gold_dataset::get_initial_centroids();

// thread_comm.execute([=](std::int64_t rank) {
//     const std::int64_t cluster_count = 3;

//     const auto kmeans_desc = kmeans::descriptor<float>{ cluster_count }
//                                  .set_max_iteration_count(100)
//                                  .set_accuracy_threshold(0.0001);

//     const auto table_id = te::table_id::homogen<float>();
//     const auto data = data_df_chunks[rank].get_table(table_id);
//     const auto initial_centroids = initial_centroids_df.get_table(table_id);

//     const auto result = dal::train(host_spmd_policy, kmeans_desc, data, initial_centroids);

//     if (rank == 0) {
//         const auto centroids = result.get_model().get_centroids();
//         const auto C = la::matrix<float>::wrap(centroids);
//         std::cout << C << std::endl;
//     }
// });

} // namespace oneapi::dal::kmeans::test
