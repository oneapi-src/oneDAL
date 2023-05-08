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

#include <unordered_set>

#include "oneapi/dal/algo/kmeans_init/compute.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

#include "oneapi/dal/algo/kmeans_init/test/fixture.hpp"

namespace oneapi::dal::kmeans_init::test {

namespace te = dal::test::engine;

template <typename TestType>
class kmeans_init_batch_test : public kmeans_init_test<TestType, kmeans_init_batch_test<TestType>> {
};

using kmeans_init_types = _TE_COMBINE_TYPES_2((float, double),
                                              (kmeans_init::method::dense,
                                               kmeans_init::method::random_dense,
                                               kmeans_init::method::plus_plus_dense,
                                               kmeans_init::method::parallel_plus_dense));

TEMPLATE_LIST_TEST_M(kmeans_init_batch_test,
                     "kmeans init dense test random",
                     "[kmeans_init][batch][random]",
                     kmeans_init_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t cluster_count = 4;

    const double data[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                            -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };
    const auto data_table = homogen_table::wrap(data, row_count, column_count);

    this->dense_checks(cluster_count, data_table);
}

} // namespace oneapi::dal::kmeans_init::test
