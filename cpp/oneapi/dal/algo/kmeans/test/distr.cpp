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

namespace oneapi::dal::kmeans::test {

using te = oneapi::dal::test::engine;

TEST("distributed kmeans") {
    const auto queue = sycl::queue{ sycl::gpu_selector{} };
    const auto communicator = dal::preview::mpi_communicator{ queue };

    const std::int64_t cluster_count = 5;
    const auto kmeans_desc = kmeans::descriptor<float>{ cluster_count }
                                 .set_max_iteration_count(10)
                                 .set_accuracy_threshold(0.001);

    const auto distributed_train_result =
        dal::preview::distributed_train(communicator, kmeans_desc, data);
}

} // namespace oneapi::dal::kmeans::test
