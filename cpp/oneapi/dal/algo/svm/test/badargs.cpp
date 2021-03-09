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

#include <array>

#include "oneapi/dal/algo/svm/infer.hpp"
#include "oneapi/dal/algo/svm/train.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::svm::test {

namespace te = dal::test::engine;

template <typename TestType>
class svm_badarg_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    static constexpr std::int64_t row_count = 8;
    static constexpr std::int64_t column_count = 2;
    static constexpr std::int64_t element_count = row_count * column_count;

    bool not_available_on_device() {
        constexpr bool is_smo = std::is_same_v<Method, svm::method::smo>;
        return get_policy().is_gpu() && is_smo;
    }

    auto get_descriptor() const {
        return svm::descriptor<Float, Method, svm::task::classification>{};
    }

    table get_train_data(std::int64_t override_row_count = row_count,
                         std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count);
        return homogen_table::wrap(train_data_.data(), override_row_count, override_column_count);
    }

    table get_train_labels(std::int64_t override_row_count = row_count) const {
        ONEDAL_ASSERT(override_row_count <= row_count);
        return homogen_table::wrap(train_labels_.data(), override_row_count, 1);
    }

    table get_infer_data(std::int64_t override_row_count = row_count,
                         std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count);
        return homogen_table::wrap(infer_data_.data(), override_row_count, override_column_count);
    }

private:
    static constexpr std::array<Float, element_count> train_data_ = {
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0
    };

    static constexpr std::array<Float, row_count> train_labels_ = { 0.0, 1.0, 0.0, 0.0,
                                                                    1.0, 1.0, 0.0, 1.0 };

    static constexpr std::array<Float, element_count> infer_data_ = {
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0
    };
};

using svm_types = COMBINE_TYPES((float, double), (svm::method::thunder, svm::method::smo));

#define SVM_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(svm_badarg_test, name, "[svm][badarg]", svm_types)

SVM_BADARG_TEST("accepts positive c") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_descriptor().set_c(1.0));
}

SVM_BADARG_TEST("throws if c is negative") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_descriptor().set_c(-1.0), domain_error);
}

SVM_BADARG_TEST("throws if c is zero") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_descriptor().set_c(0), domain_error);
}

SVM_BADARG_TEST("accepts non-negative max_iteration_count") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_descriptor().set_max_iteration_count(10.0));
}

SVM_BADARG_TEST("throws if max_iteration_count is negative") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_descriptor().set_max_iteration_count(-10.0), domain_error);
}

SVM_BADARG_TEST("accepts non-negative accuracy_threshold") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_descriptor().set_accuracy_threshold(0.01));
}

SVM_BADARG_TEST("throws if accuracy_threshold is negative") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_descriptor().set_accuracy_threshold(-0.01), domain_error);
}

SVM_BADARG_TEST("accepts non-negative cache_size") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_descriptor().set_cache_size(100.0));
}

SVM_BADARG_TEST("throws if cache_size is negative") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_descriptor().set_cache_size(-100.0), domain_error);
}

SVM_BADARG_TEST("accepts positive tau") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_NOTHROW(this->get_descriptor().set_tau(1.0));
}

SVM_BADARG_TEST("throws if tau is negative") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_descriptor().set_tau(-1.0), domain_error);
}

SVM_BADARG_TEST("throws if tau is zero") {
    SKIP_IF(this->not_available_on_device());
    REQUIRE_THROWS_AS(this->get_descriptor().set_tau(0), domain_error);
}

SVM_BADARG_TEST("throws if train data is empty") {
    SKIP_IF(this->not_available_on_device());
    const auto svm_desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->train(svm_desc, homogen_table{}, this->get_train_labels()),
                      domain_error);
}

SVM_BADARG_TEST("throws if train labels is empty") {
    SKIP_IF(this->not_available_on_device());
    const auto svm_desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->train(svm_desc, this->get_train_data(), homogen_table{}), domain_error);
}

SVM_BADARG_TEST("throws if train data rows neq train labels rows") {
    SKIP_IF(this->not_available_on_device());
    const auto svm_desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->train(svm_desc, this->get_train_data(), this->get_train_labels(4)),
                      invalid_argument);
}

SVM_BADARG_TEST("throws if train data rows neq train weights rows") {
    SKIP_IF(this->not_available_on_device());
    const auto svm_desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->train(svm_desc,
                                  this->get_train_data(),
                                  this->get_train_labels(),
                                  this->get_train_labels(4)),
                      invalid_argument);
}

SVM_BADARG_TEST("throws if infer data is empty") {
    SKIP_IF(this->not_available_on_device());
    const auto svm_desc = this->get_descriptor();
    const auto model =
        this->train(svm_desc, this->get_train_data(), this->get_train_labels()).get_model();

    REQUIRE_THROWS_AS(this->infer(svm_desc, model, homogen_table{}), domain_error);
}

SVM_BADARG_TEST("throws if infer model support_vectors is empty") {
    SKIP_IF(this->not_available_on_device());
    const auto svm_desc = this->get_descriptor();
    auto model =
        this->train(svm_desc, this->get_train_data(), this->get_train_labels()).get_model();

    REQUIRE_THROWS_AS(
        this->infer(svm_desc, model.set_support_vectors(homogen_table{}), this->get_infer_data()),
        domain_error);
}

SVM_BADARG_TEST("throws if infer model coeffs is empty") {
    SKIP_IF(this->not_available_on_device());
    const auto svm_desc = this->get_descriptor();
    auto model =
        this->train(svm_desc, this->get_train_data(), this->get_train_labels()).get_model();

    REQUIRE_THROWS_AS(
        this->infer(svm_desc, model.set_coeffs(homogen_table{}), this->get_infer_data()),
        domain_error);
}

SVM_BADARG_TEST("throws if infer model support_vectors cols neq infer data cols") {
    SKIP_IF(this->not_available_on_device());
    const auto svm_desc = this->get_descriptor();
    const auto model =
        this->train(svm_desc, this->get_train_data(), this->get_train_labels()).get_model();

    REQUIRE_THROWS_AS(this->infer(svm_desc, model, this->get_infer_data(8, 1)), invalid_argument);
}

SVM_BADARG_TEST("throws if infer model coeffs rows neq support_vector count") {
    SKIP_IF(this->not_available_on_device());
    const auto svm_desc = this->get_descriptor();
    auto model =
        this->train(svm_desc, this->get_train_data(), this->get_train_labels()).get_model();
    const auto support_vector_count = model.get_support_vector_count();
    model.set_coeffs(this->get_infer_data(support_vector_count - 1, 2));

    REQUIRE_THROWS_AS(this->infer(svm_desc, model, this->get_infer_data()), invalid_argument);
}

} // namespace oneapi::dal::svm::test
