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
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "oneapi/dal/algo/knn/train.hpp"
#include "oneapi/dal/algo/knn/infer.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::knn::test {

namespace te = dal::test::engine;
namespace de = oneapi::dal::detail;
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

    static constexpr bool is_kd_tree = std::is_same_v<Method, knn::method::kd_tree>;
    static constexpr bool is_brute_force = std::is_same_v<Method, knn::method::brute_force>;

    bool not_available_on_device() {
        return (get_policy().is_gpu() && is_kd_tree) || (get_policy().is_cpu() && is_brute_force);
    }

    void exact_nearest_indices_check(const table& train_data, const table& infer_data,
                                     const knn::infer_result<>& result) {

        const auto [labels] = unpack_result(result);

        const auto gtruth = naive_knn_search(train_data, infer_data);

        SECTION("data shape is expected") {
            REQUIRE(train_data.get_column_count() == infer_data.get_column_count());
            REQUIRE(infer_data.get_row_count() == labels.get_row_count());
            REQUIRE(labels.get_column_count() == 1);
            REQUIRE(infer_data.get_row_count() == gtruth.get_row_count());
            REQUIRE(train_data.get_row_count() == gtruth.get_column_count());
        }

        const auto m = infer_data.get_row_count();
        
        const auto indices = naive_knn_search(train_data, infer_data);

        for(std::int64_t j = 0; j < m; ++j) {
            const auto gt_indices_row = row_accessor<const Float>(indices).pull({ j, j + 1 });
            const auto te_indices_row = row_accessor<const Float>(labels).pull({ j, j + 1 });
            const auto l = gt_indices_row[0], r = te_indices_row[0];
            if (l != r) {
                CAPTURE(l, r);
                FAIL("Indices of nearest neighbors are unequal");
            }
        }


    }

    static auto naive_knn_search(const table& train_data, const table& infer_data) {

        const auto distances_matrix = distances(train_data, infer_data);
        const auto indices_matrix = argsort(distances_matrix);

        return indices_matrix;
    }

    static auto distances(const table& train_data, const table& infer_data) {
        const auto m = train_data.get_row_count();
        const auto n = infer_data.get_row_count();
        const auto d = infer_data.get_column_count();

        auto distances_arr = array<Float>::zeros(m * n);
        auto* distances_ptr = distances_arr.get_mutable_data();

        for(std::int64_t j = 0; j < n; ++j) {
            const auto queue_row = row_accessor<const Float>(infer_data).pull({ j, j + 1 });
            for(std::int64_t i = 0; i < m; ++i) {
                const auto train_row = row_accessor<const Float>(train_data).pull({ i, i + 1 });
                for(std::int64_t s = 0; s < d; ++s) {
                    const auto diff = queue_row[s] - train_row[s]; 
                    distances_ptr[j * m + i] += diff * diff;
                }
            }
        }
        return de::homogen_table_builder{}.reset(distances_arr, n, m).build();
    }

    static auto argsort(const table& distances) {
        const auto m = distances.get_row_count();
        const auto n = distances.get_column_count();
        
        auto indices = array<std::int32_t>::zeros(m * n);
        auto indices_ptr = indices.get_mutable_data();
        for(std::int64_t j = 0; j < n; ++j){
            const auto dist_row = row_accessor<const Float>(distances).pull({ j, j + 1 });
            auto idcs_row = &indices_ptr[j * m];
            std::iota(idcs_row, idcs_row + m, std::int32_t(0));
            const auto compare = 
            [&](std::int32_t x, std::int32_t y) -> bool {
                return dist_row[x] < dist_row[y];
            };
            std::sort(idcs_row, idcs_row + m, compare);
        }
        return de::homogen_table_builder{}.reset(indices, n, m).build();
    }

    static auto arange(std::int64_t from, std::int64_t to) {
        auto indices_arr = array<std::int32_t>::zeros(to - from);
        auto* indices_ptr = indices_arr.get_mutable_data();
        std::iota(indices_ptr, indices_ptr + to - from, std::int32_t(from));
        return de::homogen_table_builder{}.reset(indices_arr, to - from, 1).build();
    }

    static auto arange(std::int64_t to) {
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
    TEMPLATE_LIST_TEST_M(knn_batch_test, name, "[knn][test]", knn_types)

KNN_TEST("knn nearest points test predefined") {
    SKIP_IF(this->not_available_on_device());

    constexpr std::int64_t row_count = 6;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count = row_count * column_count;

    constexpr std::array<float, element_count> train = 
        { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f };

    constexpr std::array<float, element_count> infer = 
        { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f };

    const auto x_train_table = homogen_table::wrap(train.data(), row_count, column_count);
    const auto x_infer_table = homogen_table::wrap(infer.data(), row_count, column_count);
    const auto y_train_table = this->arange(row_count);

    const auto knn_desc = this->get_descriptor( row_count, 1 );

    auto train_result = this->train(knn_desc, x_train_table, y_train_table);
    auto infer_result = this->infer(knn_desc, x_infer_table, train_result.get_model());

    this->exact_nearest_indices_check(x_train_table, x_infer_table, infer_result);
}


} // namespace oneapi::dal::kmeans::test
