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

#include "fixture.hpp"

namespace oneapi::dal::backend::primitives::test {

TEMPLATE_TEST_M(logloss_test, "gold input test - double", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->test_gold_input();
}

TEMPLATE_TEST_M(logloss_test, "gold input test - double - no fit_intercept", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->test_gold_input(false);
}

TEMPLATE_TEST_M(logloss_test, "gold input test - float", "[logloss]", float) {
    SKIP_IF(this->get_policy().is_cpu());
    this->test_gold_input();
}

TEMPLATE_TEST_M(logloss_test, "gold input test - float - no fit intercept", "[logloss]", float) {
    SKIP_IF(this->get_policy().is_cpu());
    this->test_gold_input(false);
}

TEMPLATE_TEST_M(logloss_test, "test random input - double without L1", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.0f, 1.3f);
}

TEMPLATE_TEST_M(logloss_test,
                "test random input - double without L1 - no fit intercept",
                "[logloss]",
                double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.0f, 1.3f, false);
}

TEMPLATE_TEST_M(logloss_test, "batch test - double", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.0f, 1.3f, true, true);
}

TEMPLATE_TEST_M(logloss_test, "batch test - double - no fit intercept", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.0f, 1.3f, false, true);
}

TEMPLATE_TEST_M(logloss_test, "test random input - double with L1", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.4f, 1.3f);
}

TEMPLATE_TEST_M(logloss_test,
                "test random input - double with L1 -- no fit intercept",
                "[logloss]",
                double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.4f, 1.3f, false);
}

TEMPLATE_TEST_M(logloss_test, "test random input - float", "[logloss]", float) {
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.4f, 1.3f);
}

TEMPLATE_TEST_M(logloss_test, "test random input - float - no fit intercept", "[logloss]", float) {
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.4f, 1.3f, false);
}

TEMPLATE_TEST_M(logloss_test, "sparse data test - float", "[logloss]", float) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_sparse_input();
    this->run_sparse_test(0.0f, true);
}

TEMPLATE_TEST_M(logloss_test, "sparse data test - float - no fit intercept", "[logloss]", float) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_sparse_input();
    this->run_sparse_test(0.0f, false);
}

TEMPLATE_TEST_M(logloss_test, "sparse data test - double", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_sparse_input();
    this->run_sparse_test(1.3f, true);
}

TEMPLATE_TEST_M(logloss_test, "sparse data test - double - no fit intercept", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_sparse_input();
    this->run_sparse_test(1.3f, false);
}

} // namespace oneapi::dal::backend::primitives::test
