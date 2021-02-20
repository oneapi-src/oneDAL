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

#include "oneapi/dal/algo/knn/train.hpp"
#include "oneapi/dal/algo/knn/infer.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::knn::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class knn_batch_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor(std::int64_t override_class_count,
                        std::int64_t override_neighbor_count) const {
        return knn::descriptor<Float, Method, knn::task::classification>(
                                        override_class_count, override_neighbor_count);
    }


    void exact_nearest_check(const table& train_data, const table& infer_data,
                                            const knn::infer_result<>& result) {

        const auto [labels] = unpack_result(result);

        SECTION("data shape is expected") {
            REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
            REQUIRE(infer_data.get_row_count() == labels.get_row_count());
            REQUIRE(labels.get_column_count() == 1);
        }

        const auto gtruth = naive_knn_search(train_data, infer_data);



    }

    auto naive_knn_search(const table& train_data, const table& infer_data, 
                                                    std::int64_t k = 1 ) const {

        SECTION("data shape is expected") {
            REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
        }

        auto indices_array = arange(infer_data.get_row_count());
        auto indices_table = homogen_table::wrap(indices_array.get_data());

        const auto n = train_data.get_row_count();
        const auto m = infer_data.get_row_count();
        const auto d = infer_data.get_column_count();

        auto distances = array<Float>::zeros(m * n);

        for(std::int64_t j = 0; j < n; ++j) {
            for(std::int64_t i = 0; i < m; ++i) {
                for(std::int64_t s = 0; s < d; ++s) {
                    distances[j * m + i] = 0;
                }
            }
        }
    }

    auto arange(std::int64_t from, std::int64_t to) const {
        const std::int64_t size = to - from;
        auto arr = array<Float>::zeros(size);
        for(std::int64_t i = 0; i < size; ++i) {
            arr[i] = Float(i);
        }
        return arr;
    }

    auto arange(std::int64_t to) const {
        return arange(0, to);
    }

    void check_label_match(const array<Float>& match_map, const table& left, const table& right) {
        SECTION("label shape is expected") {
            REQUIRE(left.get_row_count() == right.get_row_count());
            REQUIRE(left.get_column_count() == right.get_column_count());
            REQUIRE(left.get_column_count() == 1);
        }
        SECTION("label match is expected") {
            const auto left_rows = row_accessor<const Float>(left).pull({ 0, -1 });
            const auto right_rows = row_accessor<const Float>(right).pull({ 0, -1 });
            for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
                const Float l = left_rows[i];
                const Float r = right_rows[i];
                if (l != match_map[r]) {
                    CAPTURE(l, r, match_map[r]);
                    FAIL("Label mismatch");
                }
            }
        }
    }

    void check_nans(const knn::infer_result<>& result) {
        const auto [labels] = unpack_result(result);

        SECTION("there is no NaN in labels") {
            REQUIRE(te::has_no_nans(labels));
        }
    }

private:
    static auto unpack_result(const knn::infer_result<>& result) {
        const auto labels = result.get_labels();
        return std::make_tuple(labels);
    }

};

using knn_types = COMBINE_TYPES((float, double), (knn::method::brute_force, knn::method::kd_tree));

#define KNN_TEST(name) \
    TEMPLATE_TEST_M(knn_test, name, "[knn][test]", knn::method::brute_force, knn::method::kd_tree)

} // namespace oneapi::dal::kmeans::test
