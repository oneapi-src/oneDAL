/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "oneapi/dal/test/engine/mocks.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;

template <typename Method>
class pca_overflow_test : public te::algo_fixture {
public:
    static constexpr std::int64_t row_count = 8;
    static constexpr std::int64_t column_count = 2;
    static constexpr std::int64_t invalid_component_count = 0x7FFFFFFFFFFFFFFF;

    bool not_available_on_device() {
        constexpr bool is_svd = std::is_same_v<Method, pca::method::svd>;
        return get_policy().is_gpu() && is_svd;
    }

    auto get_descriptor_with_invalid_component_count() const {
        return pca::descriptor<float, Method, pca::task::dim_reduction>{}.set_component_count(
            invalid_component_count);
    }

    table get_train_data_with_invalid_column_count() const {
        return te::dummy_table{ row_count, invalid_component_count };
    }

    table get_infer_data() const {
        return te::dummy_table{ row_count, column_count };
    }

    pca::model<> get_model_with_invalid_component_count() const {
        const auto eigenvectors = te::dummy_table{ invalid_component_count, column_count };
        return pca::model{}.set_eigenvectors(eigenvectors);
    }
}; // namespace oneapi::dal::pca::test

#define PCA_OVERFLOW_TEST(name) \
    TEMPLATE_TEST_M(pca_overflow_test, name, "[pca][overflow]", pca::method::cov, pca::method::svd)

PCA_OVERFLOW_TEST("train throws if component count leads to overflow") {
    SKIP_IF(this->not_available_on_device());

    const auto pca_desc = this->get_descriptor_with_invalid_component_count();
    const auto train_data = this->get_train_data_with_invalid_column_count();

    REQUIRE_THROWS_AS(this->train(pca_desc, train_data), range_error);
}

PCA_OVERFLOW_TEST("infer throws if component count leads to overflow") {
    SKIP_IF(this->not_available_on_device());

    const auto pca_desc = this->get_descriptor_with_invalid_component_count();
    const auto model = this->get_model_with_invalid_component_count();
    const auto infer_data = this->get_infer_data();

    REQUIRE_THROWS_AS(this->infer(pca_desc, model, infer_data), range_error);
}

} // namespace oneapi::dal::pca::test
