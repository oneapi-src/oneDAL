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

#include "oneapi/dal/backend/primitives/sparse_blas/test/fixture.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

using gemm_types = COMBINE_TYPES(
    (float, double),
    (transpose_nontrans, transpose_trans),
    (c_order /*, f_order */), /// oneMKL 2024.0 throws 'unimplemented' exception when the matrix B is transposed
    (c_order /*, f_order */),
    (indexing_zero_based, indexing_one_based));

TEMPLATE_LIST_TEST_M(sparse_blas_test, "ones matrix sparse CSR gemm", "[csr][gemm]", gemm_types) {
    // DPC++ Sparse GEMM from micro MKL libs is not supported on CPU
    SKIP_IF(this->get_policy().is_cpu());

    // Test takes too long time if HW emulates float64
    SKIP_IF(this->not_float64_friendly());

    this->generate_dimensions();
    this->test_gemm();
}

} // namespace oneapi::dal::backend::primitives::test
