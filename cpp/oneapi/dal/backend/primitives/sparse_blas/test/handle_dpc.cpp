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

#include "oneapi/dal/backend/primitives/sparse_blas.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::backend::primitives::test {

TEST("can construct sparse matrix handle") {
    DECLARE_TEST_POLICY(policy);
    // DPC++ Sparse BLAS from micro MKL libs is not supported on CPU
    SKIP_IF(policy.is_cpu());

    try {
        sparse_matrix_handle h(policy.get_queue());
    }
    catch(...) {
        REQUIRE(false);
    }
    SUCCEED();
}

} // namespace oneapi::dal::backend::primitives::test
