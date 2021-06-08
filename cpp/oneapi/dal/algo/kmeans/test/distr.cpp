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

namespace oneapi::dal::kmeans::test {

template <typename TestType>
class kmeans_distr_test : public kmeans_test<TestType> {
public:
    using base_t = kmeans_test<TestType>;
    using descriptor_t = typename base_t::descriptor_t;
    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;

protected:
    train_result_t train_override(const descriptor_t& desc, const train_input_t& input) override {
        const std::int64_t thread_count = GENERATE(1, 2, 4, 8, 16);
        auto comm = te::thread_communicator{ thread_count };
        // auto host_spmd_policy = dal::detail::spmd_policy{ dal::detail::host_policy{}, thread_comm };

        train_result_t result;
        comm.execute([&](std::int64_t rank) {
            const auto train_result = this->spmd_train(comm, desc, input);
            const auto merged_result = merge_train_result(comm, train_result);
            if (dal::detail::is_root_rank(comm)) {
                result = merged_result;
            }
        });

        return result;
    }

private:
    train_result_t merge_train_result(const dal::detail::spmd_communicator& comm,
                                      const train_result_t& result) override {
        const auto local_labels = result.get_labels();
        const auto merged_labels = te::stack_tables_by_rows(comm.gather(local_labels));
        if (dal::detail::is_root_rank(comm)) {
            // Assuming model, iteration_count, objective_function_value
            // are the same for all ranks
            return train_result_t{}
                .set_model(result.get_model())
                .set_labels(merged_labels)
                .set_iteration_count(result.get_iteration_count())
                .set_objective_function_value(result.get_objective_function_value());
        }
        else {
            return train_result_t{};
        }
    }
};

TEMPLATE_LIST_TEST_M(kmeans_distr_test, "distributed kmeans on host", "[distr]", kmeans_types) {
    SKIP_IF(this->not_float64_friendly());

    // TODO: Remove
    SKIP_IF(this->is_float64());

    const std::int64_t thread_count = GENERATE(1, 2, 4, 8, 16);
    auto thread_comm = te::thread_communicator{ thread_count };
    auto host_spmd_policy = dal::detail::spmd_policy{ dal::detail::host_policy{}, thread_comm };

    const auto data_df = gold_dataset::get_data();
    const auto data_df_chunks = data_df.split(thread_count);
    const auto initial_centroids_df = gold_dataset::get_initial_centroids();

    thread_comm.execute([=](std::int64_t rank) {
        const std::int64_t cluster_count = 3;

        const auto kmeans_desc = kmeans::descriptor<float>{ cluster_count }
                                     .set_max_iteration_count(100)
                                     .set_accuracy_threshold(0.0001);

        const auto table_id = te::table_id::homogen<float>();
        const auto data = data_df_chunks[rank].get_table(table_id);
        const auto initial_centroids = initial_centroids_df.get_table(table_id);

        const auto result = dal::train(host_spmd_policy, kmeans_desc, data, initial_centroids);

        if (rank == 0) {
            const auto centroids = result.get_model().get_centroids();
            const auto C = la::matrix<float>::wrap(centroids);
            std::cout << C << std::endl;
        }
    });
}

} // namespace oneapi::dal::kmeans::test
