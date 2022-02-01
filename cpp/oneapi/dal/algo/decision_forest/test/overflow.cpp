/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/train.hpp"
#include "oneapi/dal/algo/decision_forest/infer.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::decision_forest::test {

namespace df = dal::decision_forest;
namespace te = dal::test::engine;

template <typename TestType>
class df_overflow_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using Task = std::tuple_element_t<2, TestType>;

    bool is_gpu() {
        return get_policy().is_gpu();
    }

    bool is_cpu() {
        return get_policy().is_cpu();
    }

    bool not_available_on_device() {
        constexpr bool is_dense = std::is_same_v<Method, decision_forest::method::dense>;
        return get_policy().is_gpu() && is_dense;
    }

    auto get_default_descriptor() {
        return df::descriptor<Float, Method, Task>{};
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    std::int64_t get_value_to_overflow_int32() {
        return 0xFFFFFFFF;
    }
    std::int64_t get_value_to_overflow_int64() {
        return 0x7FFFFFFFFFFFFFFF;
    }

    auto get_train_data() {
        constexpr std::int64_t row_count_train = 6;
        constexpr std::int64_t column_count = 2;
        constexpr std::int64_t class_count = 2;

        static const float x_train[] = { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f,
                                         +1.f, +1.f, +1.f, +2.f, +2.f, +1.f };
        static const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

        const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
        const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

        return std::make_tuple(x_train_table, y_train_table, class_count);
    }
};

using df_cls_types = _TE_COMBINE_TYPES_3((float, double),
                                         (df::method::dense, df::method::hist),
                                         (df::task::classification));
using df_common_types = _TE_COMBINE_TYPES_3((float, double),
                                            (df::method::dense, df::method::hist),
                                            (df::task::classification, df::task::regression));

#define DF_OVERFLOW_CLS_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_overflow_test, name, "[df][overflow]", df_cls_types)

#define DF_OVERFLOW_TEST(name) \
    TEMPLATE_LIST_TEST_M(df_overflow_test, name, "[df][overflow]", df_common_types)

DF_OVERFLOW_CLS_TEST("train throws if class_count leads to overflow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->is_cpu());
    const auto [x, y, not_used] = this->get_train_data();
    auto desc = this->get_default_descriptor().set_class_count(this->get_value_to_overflow_int32());
    REQUIRE_THROWS_AS(this->train(desc, x, y).get_model(), domain_error);
}

DF_OVERFLOW_CLS_TEST("infer throws if class_count leads to overflow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->is_cpu());
    const auto [x, y, class_count] = this->get_train_data();
    auto desc = this->get_default_descriptor().set_class_count(class_count);
    REQUIRE_THROWS_AS(this->infer(desc.set_class_count(this->get_value_to_overflow_int32()),
                                  this->train(desc, x, y).get_model(),
                                  x),
                      domain_error);
}

DF_OVERFLOW_TEST("train throws if tree_count leads to overflow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->is_cpu());
    const auto [x, y, not_used] = this->get_train_data();
    auto desc = this->get_default_descriptor();
    desc.set_tree_count(this->get_value_to_overflow_int32());
    REQUIRE_THROWS_AS(this->train(desc, x, y), domain_error);
}

DF_OVERFLOW_TEST("train throws if min_observations_in_leaf_node leads to overflow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->is_cpu());
    const auto [x, y, not_used] = this->get_train_data();
    auto desc = this->get_default_descriptor();
    desc.set_min_observations_in_leaf_node(this->get_value_to_overflow_int32());
    REQUIRE_THROWS_AS(this->train(desc, x, y), domain_error);
}

DF_OVERFLOW_TEST("train throws if features_per_node leads to overflow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->is_cpu());
    const auto [x, y, not_used] = this->get_train_data();
    auto desc = this->get_default_descriptor();
    desc.set_features_per_node(this->get_value_to_overflow_int32());
    REQUIRE_THROWS_AS(this->train(desc, x, y), domain_error);
}

DF_OVERFLOW_TEST("train throws if max_bins leads to overflow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->is_cpu());
    const auto [x, y, not_used] = this->get_train_data();
    auto desc = this->get_default_descriptor();
    desc.set_max_bins(this->get_value_to_overflow_int32());
    REQUIRE_THROWS_AS(this->train(desc, x, y), domain_error);
}

DF_OVERFLOW_TEST("train throws if min_bin_size leads to overflow") {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->is_cpu());
    const auto [x, y, not_used] = this->get_train_data();
    auto desc = this->get_default_descriptor();
    desc.set_min_bin_size(this->get_value_to_overflow_int32());
    REQUIRE_THROWS_AS(this->train(desc, x, y), domain_error);
}

} // namespace oneapi::dal::decision_forest::test
