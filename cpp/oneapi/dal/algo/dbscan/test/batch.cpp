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

#include <limits>
#include <cmath>

#include "oneapi/dal/algo/dbscan/compute.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/metrics/clustering.hpp"

namespace oneapi::dal::dbscan::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class dbscan_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor(Float epsilon, std::int64_t min_observations) const {
        return dbscan::descriptor<Float, Method>(epsilon, min_observations);
    }

    void dbi_determenistic_checks(const table& data,
                                  const table& weights,
                                  Float epsilon,
                                  std::int64_t min_observations,
                                  Float ref_dbi,
                                  Float dbi_ref_tol = 1.0e-4) {
        CAPTURE(epsilon, min_observations);

        INFO("create descriptor")
        const auto dbscan_desc = get_descriptor(epsilon, min_observations);

        INFO("run compute");
        const auto compute_result = compute(dbscan_desc, data, weights);

        //        auto dbi = te::davies_bouldin_index(data, compute_result.get_data(), compute_result.get_responses());
        //        CAPTURE(dbi, ref_dbi);
    }

    bool check_value_with_ref_tol(Float val, Float ref_val, Float ref_tol) {
        Float max_abs = std::max(fabs(val), fabs(ref_val));
        if (max_abs == 0.0)
            return true;
        return fabs(val - ref_val) / max_abs < ref_tol;
    }

    Float squared_euclidian_distance(std::int64_t x_offset,
                                     const array<Float>& x,
                                     std::int64_t y_offset,
                                     const array<Float>& y,
                                     std::int64_t feature_count) {
        Float sum = 0.0;
        for (std::int64_t i = 0; i < feature_count; i++) {
            Float val = x[x_offset * feature_count + i] - y[y_offset * feature_count + i];
            sum += val * val;
        }
        return sum;
    }

    void check_nans(const dbscan::compute_result<>& result) {
        const auto [centroids, labels, iteration_count] = unpack_result(result);

        INFO("check if there is no NaN in centroids")
        REQUIRE(te::has_no_nans(centroids));

        INFO("check if there is no NaN in labels")
        REQUIRE(te::has_no_nans(labels));
    }

private:
    static auto unpack_result(const dbscan::compute_result<>& result) {
        /*        const auto centroids = result.get_model().get_centroids();
        const auto labels = result.get_labels();
        const auto iteration_count = result.get_iteration_count();
        return std::make_tuple(centroids, labels, iteration_count);*/
    }
};

using dbscan_types = COMBINE_TYPES((float, double), (dbscan::method::brute_force));

TEMPLATE_LIST_TEST_M(dbscan_batch_test,
                     "dbscan degenerated test",
                     "[dbscan][batch]",
                     dbscan_types) {
    // number of observations is equal to number of centroids (obvious clustering)
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;
    Float data[] = { 0.0, 5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 1.0 };
    const auto x = homogen_table::wrap(data, 3, 5);

    Float weights[] = { 1.0, 1.1, 1, 2 };
    const auto w = homogen_table::wrap(weights, 3, 1);
    this->dbi_determenistic_checks(x, w, 2.0, 2, 1.0);
}

} // namespace oneapi::dal::dbscan::test
