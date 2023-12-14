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

#include "oneapi/dal/algo/logistic_regression/test/spmd_fixture.hpp"

namespace oneapi::dal::logistic_regression::test {

TEMPLATE_LIST_TEST_M(log_reg_spmd_test, "LogReg common flow", "[lr][spmd]", log_reg_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->gen_input(true, 0.5);
    this->set_rank_count(2);

    this->run_test();
}

} // namespace oneapi::dal::logistic_regression::test
