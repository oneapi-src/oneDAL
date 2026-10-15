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

#include <algorithm>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "oneapi/dal/algo/kmeans_init/compute.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

#include "oneapi/dal/algo/kmeans_init/test/fixture.hpp"

#include "src/algorithms/kmeans/kmeans_init_kernel.h"
#include "src/algorithms/kmeans/kmeans_plusplus_init_impl.i"

namespace oneapi::dal::kmeans_init::test {

namespace te = dal::test::engine;
namespace daal_dm = daal::data_management;
namespace daal_init = daal::algorithms::kmeans::init::internal;

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

constexpr auto daal_cpu = daal::internal::firstCpuType;

// The DAAL tables wrap the test's arrays, which must outlive them.
template <typename Float>
struct dense_helper {
    using float_t = Float;
    using type = daal_init::DataHelperDense<Float, daal_cpu>;
    dense_helper(std::vector<Float>& data, std::size_t row_count, std::size_t column_count)
            : table(daal_dm::HomogenNumericTable<Float>::create(data.data(),
                                                                column_count,
                                                                row_count)) {}
    daal_dm::NumericTablePtr table;
};

// Every value is stored explicitly, zeros included, with one-based indices.
template <typename Float>
struct csr_helper {
    using float_t = Float;
    using type = daal_init::DataHelperCSR<Float, daal_cpu>;
    csr_helper(std::vector<Float>& data, std::size_t row_count, std::size_t column_count)
            : col_indices(row_count * column_count),
              row_offsets(row_count + 1) {
        for (std::size_t i = 0; i < col_indices.size(); ++i) {
            col_indices[i] = i % column_count + 1;
        }
        for (std::size_t r = 0; r <= row_count; ++r) {
            row_offsets[r] = r * column_count + 1;
        }
        table = daal_dm::CSRNumericTable::create(data.data(),
                                                 col_indices.data(),
                                                 row_offsets.data(),
                                                 column_count,
                                                 row_count);
    }
    std::vector<std::size_t> col_indices;
    std::vector<std::size_t> row_offsets;
    daal_dm::NumericTablePtr table;
};

using data_helper_types =
    std::tuple<dense_helper<float>, dense_helper<double>, csr_helper<float>, csr_helper<double>>;

// Each trial's slot must be scored with that trial's own candidate, the previous best trial's
// slot included, and every slot must start from the previous best trial's distances.
TEMPLATE_LIST_TEST("kmeans++ updateMinDistInBlock scores each trial with its own candidate",
                   "[kmeans_init][batch][plus_plus]",
                   data_helper_types) {
    using float_t = typename TestType::float_t;
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 2;
    constexpr std::size_t trial_count = 3;

    std::vector<float_t> data = { 0, 0, 3, 0, 0, 4, 6, 8 };
    const TestType input(data, row_count, column_count);
    typename TestType::type helper(input.table.get());

    // Trial t's candidate is row t.
    const std::vector<float_t> candidates(data.begin(), data.begin() + trial_count * column_count);
    const std::vector<float_t> previous_min_dist = { 1000, 1000, 1000, 50 };

    // min(previous_min_dist, squared distance to row t) for each trial t and row.
    const float_t expected_dist[trial_count][row_count] = { { 0, 9, 16, 50 },
                                                            { 9, 0, 25, 50 },
                                                            { 16, 25, 0, 50 } };
    const float_t expected_potential[trial_count] = { 75, 84, 91 };

    for (std::size_t best_trial = 0; best_trial < trial_count; ++best_trial) {
        CAPTURE(best_trial);
        std::vector<float_t> min_dist(trial_count * row_count, float_t(-1));
        std::copy(previous_min_dist.begin(),
                  previous_min_dist.end(),
                  min_dist.begin() + best_trial * row_count);
        std::vector<float_t> potential(trial_count, float_t(-1));

        const auto status = helper.updateMinDistInBlock(potential.data(),
                                                        1,
                                                        0,
                                                        trial_count,
                                                        best_trial,
                                                        nullptr,
                                                        candidates.data(),
                                                        min_dist.data());
        REQUIRE(status.ok());

        for (std::size_t t = 0; t < trial_count; ++t) {
            CAPTURE(t);
            REQUIRE(potential[t] == expected_potential[t]);
            for (std::int64_t r = 0; r < row_count; ++r) {
                CAPTURE(r);
                REQUIRE(min_dist[t * row_count + r] == expected_dist[t][r]);
            }
        }
    }
}

} // namespace oneapi::dal::kmeans_init::test
