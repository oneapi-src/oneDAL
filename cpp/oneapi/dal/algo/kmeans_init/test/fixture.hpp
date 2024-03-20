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

namespace oneapi::dal::kmeans_init::test {

namespace te = dal::test::engine;

template <typename TestType, typename Derived>
class kmeans_init_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using base_t = te::crtp_algo_fixture<TestType, Derived>;
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using desc_t = kmeans_init::descriptor<float_t, method_t>;

    desc_t get_descriptor(std::int64_t cluster_count) const {
        return kmeans_init::descriptor<float_t, method_t>{ cluster_count };
    }

    bool not_available_on_device() {
        constexpr bool is_parallel_plus_dense =
            std::is_same_v<method_t, kmeans_init::method::parallel_plus_dense>;
        return this->get_policy().is_gpu() && is_parallel_plus_dense;
    }

    void dense_checks(std::int64_t cluster_count, const table& data) {
        CAPTURE(cluster_count);

        INFO("create descriptor");
        const auto desc = get_descriptor(cluster_count);

        INFO("compute");
        const auto compute_results = this->compute(desc, data);
        check_results(cluster_count, data, compute_results.get_centroids());
    }

    void check_results(std::int64_t cluster_count, const table& data, const table& centroids) {
        const std::int64_t row_count = data.get_row_count();
        const std::int64_t column_count = data.get_column_count();

        REQUIRE(centroids.get_row_count() == cluster_count);
        REQUIRE(centroids.get_column_count() == column_count);

        const auto data_array = row_accessor<const float_t>(data).pull();
        const auto centroid_array = row_accessor<const float_t>(centroids).pull();

        const auto* const data_ptr = data_array.get_data();
        const auto* const centroid_ptr = centroid_array.get_data();

        std::set<std::int64_t> indices{};
        for (std::int64_t i = 0; i < cluster_count; ++i) {
            for (std::int64_t j = 0; j < row_count; ++j) {
                bool match = true;
                for (std::int64_t k = 0; k < column_count; ++k) {
                    const auto data_val = data_ptr[j * column_count + k];
                    const auto centroid_val = centroid_ptr[i * column_count + k];

                    if (data_val != centroid_val) {
                        match = false;
                        break;
                    }
                }
                if (match && indices.insert(j).second) {
                    break;
                }
            }
        }
        REQUIRE(dal::detail::integral_cast<std::int64_t>(indices.size()) == cluster_count);
    }
};

} // namespace oneapi::dal::kmeans_init::test
