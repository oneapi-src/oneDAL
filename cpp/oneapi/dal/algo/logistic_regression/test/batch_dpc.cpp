/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/logistic_regression/test/fixture.hpp"

#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::logistic_regression::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename TestType>
class log_reg_batch_test : public log_reg_test<TestType, log_reg_batch_test<TestType>> {
public:
    using base_t = log_reg_test<TestType, log_reg_batch_test<TestType>>;
    using float_t = typename base_t::float_t;
    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;
};

TEMPLATE_LIST_TEST_M(log_reg_batch_test, "LogReg common flow", "[logreg][batch]", log_reg_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->gen_dimensions();
    this->gen_input(true, 0.5);

    this->run_test();
}

TEMPLATE_LIST_TEST_M(log_reg_batch_test,
                     "LogReg common flow - no fit intercept",
                     "[logreg][batch]",
                     log_reg_types) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->gen_dimensions();
    this->gen_input(false, 0.5);

    this->run_test();
}

} // namespace oneapi::dal::logistic_regression::test
