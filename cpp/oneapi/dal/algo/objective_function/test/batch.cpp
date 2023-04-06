/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/objective_function/compute.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/algo/objective_function/test/fixture.hpp"

namespace oneapi::dal::objective_function::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace obj_fun = oneapi::dal::objective_function;
namespace lg = oneapi::dal::logloss_objective;

template <typename TestType>
class logloss_batch_test : public logloss_test<TestType, logloss_batch_test<TestType>> {
public:
    using base_t = logloss_test<TestType, logloss_batch_test<TestType>>;

    void gen_dimensions() {
        this->n_ = GENERATE(20, 50, 70);
        this->p_ = GENERATE(10, 15);
    }

    void gen_big_dimensions() {
        this->n_ = GENERATE(25000, 50000, 100000);
        this->p_ = 1000;
    }
};

TEMPLATE_LIST_TEST_M(logloss_batch_test,
                     "logloss tests",
                     "[logloss][integration][gpu]",
                     logloss_types) {
    SKIP_IF(this->not_float64_friendly());

    this->gen_dimensions();
    this->gen_input();
    this->set_reg_coefs(1.1, 2.3);
    this->set_intercept_flag(true);
    this->general_checks();
}

TEMPLATE_LIST_TEST_M(logloss_batch_test,
                     "logloss tests - no fit_intercept",
                     "[logloss][integration][gpu]",
                     logloss_types) {
    SKIP_IF(this->not_float64_friendly());

    this->gen_dimensions();
    this->gen_input();
    this->set_reg_coefs(1.1, 2.3);
    this->set_intercept_flag(false);
    this->general_checks();
}

TEMPLATE_LIST_TEST_M(logloss_batch_test,
                     "logloss big test",
                     "[logloss][integration][gpu]",
                     logloss_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->gen_big_dimensions();
    this->gen_input();
    this->set_reg_coefs(1.1, 2.3);
    this->test_big();
}

} // namespace oneapi::dal::objective_function::test
