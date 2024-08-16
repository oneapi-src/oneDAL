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
    using Task = std::tuple_element_t<2, TestType>;
    static constexpr std::int64_t row_count = 8;
    static constexpr std::int64_t column_count = 2;
    static constexpr std::int64_t element_count = row_count * column_count;

    bool not_available_on_device() {
        constexpr bool is_smo = std::is_same_v<Method, svm::method::smo>;
        constexpr bool is_reg = std::is_same_v<Task, svm::task::regression>;
        constexpr bool is_nu =
            dal::detail::is_one_of_v<Task, svm::task::nu_classification, svm::task::nu_regression>;
        return get_policy().is_gpu() && (is_smo || is_reg || is_nu);
    }

    auto get_descriptor() const {
        return svm::descriptor<Float, Method, Task>{};
    }

    table get_train_data(std::int64_t override_row_count = row_count,
                         std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count);
        return homogen_table::wrap(train_data_.data(), override_row_count, override_column_count);
    }

    table get_train_responses(std::int64_t override_row_count = row_count) const {
        ONEDAL_ASSERT(override_row_count <= row_count);
        return homogen_table::wrap(train_responses_.data(), override_row_count, 1);
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

    static constexpr std::array<Float, row_count> train_responses_ = { 0.0, 1.0, 0.0, 0.0,
                                                                       1.0, 1.0, 0.0, 1.0 };

    static constexpr std::array<Float, element_count> infer_data_ = {
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0
    };
};

#define TEST_NON_NEGATIVE_C                                 \
    {                                                       \
        SKIP_IF(this->not_available_on_device());           \
        REQUIRE_NOTHROW(this->get_descriptor().set_c(1.0)); \
    }

#define TEST_NEGATIVE_C                                                      \
    {                                                                        \
        SKIP_IF(this->not_available_on_device());                            \
        REQUIRE_THROWS_AS(this->get_descriptor().set_c(-1.0), domain_error); \
    }

#define TEST_ZERO_C                                                       \
    {                                                                     \
        SKIP_IF(this->not_available_on_device());                         \
        REQUIRE_THROWS_AS(this->get_descriptor().set_c(0), domain_error); \
    }

#define TEST_NON_NEGATIVE_MAX_ITER_COUNT                                       \
    {                                                                          \
        SKIP_IF(this->not_available_on_device());                              \
        REQUIRE_NOTHROW(this->get_descriptor().set_max_iteration_count(10.0)); \
    }

#define TEST_NEGATIVE_MAX_ITER_COUNT                                                            \
    {                                                                                           \
        SKIP_IF(this->not_available_on_device());                                               \
        REQUIRE_THROWS_AS(this->get_descriptor().set_max_iteration_count(-10.0), domain_error); \
    }

#define TEST_NON_NEGATIVE_ACCURACY_TRESHOLD                                   \
    {                                                                         \
        SKIP_IF(this->not_available_on_device());                             \
        REQUIRE_NOTHROW(this->get_descriptor().set_accuracy_threshold(0.01)); \
    }

#define TEST_NEGATIVE_ACCURACY_TRESHOLD                                                        \
    {                                                                                          \
        SKIP_IF(this->not_available_on_device());                                              \
        REQUIRE_THROWS_AS(this->get_descriptor().set_accuracy_threshold(-0.01), domain_error); \
    }

#define TEST_NON_NEGATIVE_CACHE_SIZE                                   \
    {                                                                  \
        SKIP_IF(this->not_available_on_device());                      \
        REQUIRE_NOTHROW(this->get_descriptor().set_cache_size(100.0)); \
    }

#define TEST_NEGATIVE_CACHE_SIZE                                                        \
    {                                                                                   \
        SKIP_IF(this->not_available_on_device());                                       \
        REQUIRE_THROWS_AS(this->get_descriptor().set_cache_size(-100.0), domain_error); \
    }

#define TEST_NON_NEGATIVE_TAU                                 \
    {                                                         \
        SKIP_IF(this->not_available_on_device());             \
        REQUIRE_NOTHROW(this->get_descriptor().set_tau(1.0)); \
    }

#define TEST_NEGATIVE_TAU                                                      \
    {                                                                          \
        SKIP_IF(this->not_available_on_device());                              \
        REQUIRE_THROWS_AS(this->get_descriptor().set_tau(-1.0), domain_error); \
    }

#define TEST_ZERO_TAU                                                       \
    {                                                                       \
        SKIP_IF(this->not_available_on_device());                           \
        REQUIRE_THROWS_AS(this->get_descriptor().set_tau(0), domain_error); \
    }

#define TEST_NON_NEGATIVE_EPSILON                                 \
    {                                                             \
        SKIP_IF(this->not_available_on_device());                 \
        REQUIRE_NOTHROW(this->get_descriptor().set_epsilon(1.0)); \
    }

#define TEST_NEGATIVE_EPSILON                                                      \
    {                                                                              \
        SKIP_IF(this->not_available_on_device());                                  \
        REQUIRE_THROWS_AS(this->get_descriptor().set_epsilon(-1.0), domain_error); \
    }

#define TEST_ZERO_EPSILON                                       \
    {                                                           \
        SKIP_IF(this->not_available_on_device());               \
        REQUIRE_NOTHROW(this->get_descriptor().set_epsilon(0)); \
    }

#define TEST_NU_FROM_ZERO_TO_ONE                              \
    {                                                         \
        SKIP_IF(this->not_available_on_device());             \
        REQUIRE_NOTHROW(this->get_descriptor().set_nu(0.25)); \
    }

#define TEST_NEGATIVE_NU                                                      \
    {                                                                         \
        SKIP_IF(this->not_available_on_device());                             \
        REQUIRE_THROWS_AS(this->get_descriptor().set_nu(-1.0), domain_error); \
    }

#define TEST_ZERO_NU                                                       \
    {                                                                      \
        SKIP_IF(this->not_available_on_device());                          \
        REQUIRE_THROWS_AS(this->get_descriptor().set_nu(0), domain_error); \
    }

#define TEST_NU_GREATER_THAN_ONE                                             \
    {                                                                        \
        SKIP_IF(this->not_available_on_device());                            \
        REQUIRE_THROWS_AS(this->get_descriptor().set_nu(2.0), domain_error); \
    }

#define TEST_EMPTY_TRAIN_DATA                                                                  \
    {                                                                                          \
        SKIP_IF(this->not_available_on_device());                                              \
        const auto svm_desc = this -> get_descriptor();                                        \
        REQUIRE_THROWS_AS(this->train(svm_desc, homogen_table{}, this->get_train_responses()), \
                          domain_error);                                                       \
    }

#define TEST_EMPTY_TRAIN_RESPONSES                                                        \
    {                                                                                     \
        SKIP_IF(this->not_available_on_device());                                         \
        const auto svm_desc = this -> get_descriptor();                                   \
        REQUIRE_THROWS_AS(this->train(svm_desc, this->get_train_data(), homogen_table{}), \
                          domain_error);                                                  \
    }

#define TEST_TRAIN_DATA_ROWS_NEQ_TRAIN_RESPONSES_ROWS                                    \
    {                                                                                    \
        SKIP_IF(this->not_available_on_device());                                        \
        const auto svm_desc = this -> get_descriptor();                                  \
        REQUIRE_THROWS_AS(                                                               \
            this->train(svm_desc, this->get_train_data(), this->get_train_responses(4)), \
            invalid_argument);                                                           \
    }

#define TEST_TRAIN_DATA_ROWS_NEQ_TRAIN_WEIGHTS_ROWS                  \
    {                                                                \
        SKIP_IF(this->not_available_on_device());                    \
        const auto svm_desc = this -> get_descriptor();              \
        REQUIRE_THROWS_AS(this->train(svm_desc,                      \
                                      this->get_train_data(),        \
                                      this->get_train_responses(),   \
                                      this->get_train_responses(4)), \
                          invalid_argument);                         \
    }

#define TEST_EMPTY_INFER_DATA                                                                      \
    {                                                                                              \
        SKIP_IF(this->not_available_on_device());                                                  \
        const auto svm_desc = this -> get_descriptor();                                            \
        const auto model = this                                                                    \
                           -> train(svm_desc, this->get_train_data(), this->get_train_responses()) \
                               .get_model();                                                       \
        REQUIRE_THROWS_AS(this->infer(svm_desc, model, homogen_table{}), domain_error);            \
    }

#define TEST_IF_SV_EMPTY                                                                          \
    {                                                                                             \
        SKIP_IF(this->not_available_on_device());                                                 \
        const auto svm_desc = this -> get_descriptor();                                           \
        auto model = this -> train(svm_desc, this->get_train_data(), this->get_train_responses()) \
                         .get_model();                                                            \
        REQUIRE_THROWS_AS(this->infer(svm_desc,                                                   \
                                      model.set_support_vectors(homogen_table{}),                 \
                                      this->get_infer_data()),                                    \
                          domain_error);                                                          \
    }

#define TEST_IF_COEFS_EMPTY                                                                       \
    {                                                                                             \
        SKIP_IF(this->not_available_on_device());                                                 \
        const auto svm_desc = this -> get_descriptor();                                           \
        auto model = this -> train(svm_desc, this->get_train_data(), this->get_train_responses()) \
                         .get_model();                                                            \
        REQUIRE_THROWS_AS(                                                                        \
            this->infer(svm_desc, model.set_coeffs(homogen_table{}), this->get_infer_data()),     \
            domain_error);                                                                        \
    }

#define TEST_IF_SV_COLS_NEQ_INFER_DATA_COLS                                                        \
    {                                                                                              \
        SKIP_IF(this->not_available_on_device());                                                  \
        const auto svm_desc = this -> get_descriptor();                                            \
        const auto model = this                                                                    \
                           -> train(svm_desc, this->get_train_data(), this->get_train_responses()) \
                               .get_model();                                                       \
        REQUIRE_THROWS_AS(this->infer(svm_desc, model, this->get_infer_data(8, 1)),                \
                          invalid_argument);                                                       \
    }

#define TEST_IF_COEFFS_ROWS_NEQ_SV_COUNT                                                           \
    {                                                                                              \
        SKIP_IF(this->not_available_on_device());                                                  \
        const auto svm_desc = this -> get_descriptor();                                            \
        auto model = this -> train(svm_desc, this->get_train_data(), this->get_train_responses())  \
                         .get_model();                                                             \
        const auto support_vector_count = model.get_support_vector_count();                        \
        model.set_coeffs(this->get_infer_data(support_vector_count - 1, 2));                       \
        REQUIRE_THROWS_AS(this->infer(svm_desc, model, this->get_infer_data()), invalid_argument); \
    }

using thunder_svm_types = COMBINE_TYPES((float),
                                        (svm::method::thunder),
                                        (svm::task::regression,
                                         svm::task::nu_classification,
                                         svm::task::nu_regression));

#define THUNDER_SVM_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(svm_badarg_test, name, "[svm][badarg]", thunder_svm_types)

THUNDER_SVM_BADARG_TEST("accepts non-negative max_iteration_count")
TEST_NON_NEGATIVE_MAX_ITER_COUNT

THUNDER_SVM_BADARG_TEST("throws if max_iteration_count is negative")
TEST_NEGATIVE_MAX_ITER_COUNT

THUNDER_SVM_BADARG_TEST("accepts non-negative accuracy_threshold")
TEST_NON_NEGATIVE_ACCURACY_TRESHOLD

THUNDER_SVM_BADARG_TEST("throws if accuracy_threshold is negative")
TEST_NEGATIVE_ACCURACY_TRESHOLD

THUNDER_SVM_BADARG_TEST("accepts non-negative cache_size")
TEST_NON_NEGATIVE_CACHE_SIZE

THUNDER_SVM_BADARG_TEST("throws if cache_size is negative")
TEST_NEGATIVE_CACHE_SIZE

THUNDER_SVM_BADARG_TEST("accepts positive tau")
TEST_NON_NEGATIVE_TAU

THUNDER_SVM_BADARG_TEST("throws if tau is negative")
TEST_NEGATIVE_TAU

THUNDER_SVM_BADARG_TEST("throws if tau is zero")
TEST_ZERO_TAU

THUNDER_SVM_BADARG_TEST("throws if train data is empty")
TEST_EMPTY_TRAIN_DATA

THUNDER_SVM_BADARG_TEST("throws if train responses is empty")
TEST_EMPTY_TRAIN_RESPONSES

THUNDER_SVM_BADARG_TEST("throws if train data rows neq train responses rows")
TEST_TRAIN_DATA_ROWS_NEQ_TRAIN_RESPONSES_ROWS

THUNDER_SVM_BADARG_TEST("throws if train data rows neq train weights rows")
TEST_TRAIN_DATA_ROWS_NEQ_TRAIN_WEIGHTS_ROWS

THUNDER_SVM_BADARG_TEST("throws if infer data is empty")
TEST_EMPTY_INFER_DATA

THUNDER_SVM_BADARG_TEST("throws if infer model support_vectors is empty")
TEST_IF_SV_EMPTY

THUNDER_SVM_BADARG_TEST("throws if infer model coeffs is empty")
TEST_IF_COEFS_EMPTY

THUNDER_SVM_BADARG_TEST("throws if infer model support_vectors cols neq infer data cols")
TEST_IF_SV_COLS_NEQ_INFER_DATA_COLS

THUNDER_SVM_BADARG_TEST("throws if infer model coeffs rows neq support_vector count")
TEST_IF_COEFFS_ROWS_NEQ_SV_COUNT

using svc_types = COMBINE_TYPES((float),
                                (svm::method::thunder, svm::method::smo),
                                (svm::task::classification));

#define SVC_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(svm_badarg_test, name, "[svm][badarg]", svc_types)

SVC_BADARG_TEST("accepts positive c")
TEST_NON_NEGATIVE_C

SVC_BADARG_TEST("throws if c is negative")
TEST_NEGATIVE_C

SVC_BADARG_TEST("throws if c is zero")
TEST_ZERO_C

SVC_BADARG_TEST("accepts non-negative max_iteration_count")
TEST_NON_NEGATIVE_MAX_ITER_COUNT

SVC_BADARG_TEST("throws if max_iteration_count is negative")
TEST_NEGATIVE_MAX_ITER_COUNT

SVC_BADARG_TEST("accepts non-negative accuracy_threshold")
TEST_NON_NEGATIVE_ACCURACY_TRESHOLD

SVC_BADARG_TEST("throws if accuracy_threshold is negative")
TEST_NEGATIVE_ACCURACY_TRESHOLD

SVC_BADARG_TEST("accepts non-negative cache_size")
TEST_NON_NEGATIVE_CACHE_SIZE

SVC_BADARG_TEST("throws if cache_size is negative")
TEST_NEGATIVE_CACHE_SIZE

SVC_BADARG_TEST("accepts positive tau")
TEST_NON_NEGATIVE_TAU

SVC_BADARG_TEST("throws if tau is negative")
TEST_NEGATIVE_TAU

SVC_BADARG_TEST("throws if tau is zero")
TEST_ZERO_TAU

SVC_BADARG_TEST("throws if train data is empty")
TEST_EMPTY_TRAIN_DATA

SVC_BADARG_TEST("throws if train responses is empty")
TEST_EMPTY_TRAIN_RESPONSES

SVC_BADARG_TEST("throws if train data rows neq train responses rows")
TEST_TRAIN_DATA_ROWS_NEQ_TRAIN_RESPONSES_ROWS

SVC_BADARG_TEST("throws if train data rows neq train weights rows")
TEST_TRAIN_DATA_ROWS_NEQ_TRAIN_WEIGHTS_ROWS

SVC_BADARG_TEST("throws if infer data is empty")
TEST_EMPTY_INFER_DATA

SVC_BADARG_TEST("throws if infer model support_vectors is empty")
TEST_IF_SV_EMPTY

SVC_BADARG_TEST("throws if infer model coeffs is empty")
TEST_IF_COEFS_EMPTY

SVC_BADARG_TEST("throws if infer model support_vectors cols neq infer data cols")
TEST_IF_SV_COLS_NEQ_INFER_DATA_COLS

SVC_BADARG_TEST("throws if infer model coeffs rows neq support_vector count")
TEST_IF_COEFFS_ROWS_NEQ_SV_COUNT

using svr_types = COMBINE_TYPES((float), (svm::method::thunder), (svm::task::regression));

#define SVR_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(svm_badarg_test, name, "[svm][badarg]", svr_types)

SVR_BADARG_TEST("accepts positive epsilon")
TEST_NON_NEGATIVE_EPSILON

SVR_BADARG_TEST("throws if epsilon is negative")
TEST_NEGATIVE_EPSILON

SVR_BADARG_TEST("accepts zero epsilon")
TEST_ZERO_EPSILON

SVR_BADARG_TEST("accepts positive c")
TEST_NON_NEGATIVE_C

SVR_BADARG_TEST("throws if c is negative")
TEST_NEGATIVE_C

SVR_BADARG_TEST("throws if c is zero")
TEST_ZERO_C

using nusvc_types = COMBINE_TYPES((float), (svm::method::thunder), (svm::task::nu_classification));

#define NUSVC_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(svm_badarg_test, name, "[svm][badarg]", nusvc_types)

NUSVC_BADARG_TEST("accepts nu from zero to one")
TEST_NU_FROM_ZERO_TO_ONE

NUSVC_BADARG_TEST("throws if nu is negative")
TEST_NEGATIVE_NU

NUSVC_BADARG_TEST("throws if nu is zero")
TEST_ZERO_NU

NUSVC_BADARG_TEST("throws if nu is greater than one")
TEST_NU_GREATER_THAN_ONE

using nusvr_types = COMBINE_TYPES((float), (svm::method::thunder), (svm::task::nu_regression));

#define NUSVR_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(svm_badarg_test, name, "[svm][badarg]", nusvr_types)

NUSVR_BADARG_TEST("accepts nu from zero to one")
TEST_NU_FROM_ZERO_TO_ONE

NUSVR_BADARG_TEST("throws if nu is negative")
TEST_NEGATIVE_NU

NUSVR_BADARG_TEST("throws if nu is zero")
TEST_ZERO_NU

NUSVR_BADARG_TEST("throws if nu is greater than one")
TEST_NU_GREATER_THAN_ONE

NUSVR_BADARG_TEST("accepts positive c")
TEST_NON_NEGATIVE_C

NUSVR_BADARG_TEST("throws if c is negative")
TEST_NEGATIVE_C

NUSVR_BADARG_TEST("throws if c is zero")
TEST_ZERO_C

} // namespace oneapi::dal::svm::test
