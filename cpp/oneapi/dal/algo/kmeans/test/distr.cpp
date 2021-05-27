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

#include "oneapi/dal/algo/kmeans/train.hpp"
#include "oneapi/dal/algo/kmeans/infer.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/mocks.hpp"

namespace oneapi::dal::kmeans::test {

namespace te = dal::test::engine;

TEST("distributed kmeans on host") {
    const std::int64_t thread_count = 2;
    auto thread_comm = te::thread_communicator{ thread_count };
    auto host_spmd_policy = dal::detail::spmd_policy{ dal::detail::host_policy{}, thread_comm };

    const auto df = te::dataframe_builder{ thread_count * 1000, 10 }.fill_normal(-1.2, 5.1).build();
    const auto df_chunks = df.split(thread_count);

    std::vector<table> chunks;
    for (const auto& df_chunks : df_chunks) {
        const auto data = df_chunks.get_table(te::table_id::homogen<float>());
        chunks.push_back(data);
    }

    thread_comm.execute([=](std::int64_t rank) {
        const std::int64_t cluster_count = 3;

        const auto kmeans_desc = kmeans::descriptor<float>{ cluster_count }
                                     .set_max_iteration_count(10)
                                     .set_accuracy_threshold(0.001);

        const float data[] = {
            0.0, 5.0, 0.0, 0.0, 0.0, //
            1.0, 1.0, 4.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 5.0, 1.0, //
        };
        const auto x = homogen_table::wrap(data, 3, 5);

        const auto distributed_train_result = dal::train(kmeans_desc, x);
    });
}

} // namespace oneapi::dal::kmeans::test
