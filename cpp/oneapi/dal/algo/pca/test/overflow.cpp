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

#include "oneapi/dal/algo/pca/train.hpp"
#include "oneapi/dal/algo/pca/infer.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/util/table_mocks.hpp"

namespace oneapi::dal::pca::test {

namespace tu = oneapi::dal::test::util;
namespace te = oneapi::dal::test::engine;

template <typename Method>
class pca_overflow_test : public te::algo_fixture {
public:
    static constexpr std::int64_t row_count = 8;
    static constexpr std::int64_t column_count = 2;
    static constexpr std::int64_t invalid_component_count = 0x7FFFFFFFFFFFFFFF;

    auto get_descriptor_with_invalid_component_count() const {
        return pca::descriptor<float, Method, pca::task::dim_reduction>{}
            .set_component_count(invalid_component_count);
    }

    table get_train_data_with_invalid_column_count() const {
        return homogen_table{ tu::dummy_homogen_table{row_count, invalid_component_count} };
    }

    table get_infer_data() const {
        return homogen_table{ tu::dummy_homogen_table{row_count, column_count} };
    }

    pca::model<> get_model_with_invalid_component_count() const {
        const auto eigenvectors = homogen_table{
            tu::dummy_homogen_table{invalid_component_count, column_count} };
        return pca::model{}.set_eigenvectors(eigenvectors);
    }
};

TYPED_TEST_SUITE_P(pca_overflow_test);

TYPED_TEST_P(pca_overflow_test, train_throws_if_component_count_leads_to_overflow) {
    const auto pca_desc = this->get_descriptor_with_invalid_component_count();
    const auto train_data = this->get_train_data_with_invalid_column_count();

    ASSERT_THROW(this->train(pca_desc, train_data), range_error);
}

TYPED_TEST_P(pca_overflow_test, infer_throws_if_component_count_leads_to_overflow) {
    const auto pca_desc = this->get_descriptor_with_invalid_component_count();
    const auto model = this->get_model_with_invalid_component_count();
    const auto infer_data = this->get_infer_data();

    ASSERT_THROW(this->infer(pca_desc, model, infer_data), range_error);
}

REGISTER_TYPED_TEST_SUITE_P(pca_overflow_test,
                            infer_throws_if_component_count_leads_to_overflow,
                            train_throws_if_component_count_leads_to_overflow);

using pca_methods = testing::Types<pca::method::cov, pca::method::svd>;
INSTANTIATE_TYPED_TEST_SUITE_P(pca,
                               pca_overflow_test,
                               pca_methods);

} // namespace oneapi::dal::pca::test
