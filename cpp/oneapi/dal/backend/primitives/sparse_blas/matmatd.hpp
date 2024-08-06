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

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/misc.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/handle.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

/// Computes a sparse matrix - sparse matrix product with a dense result:
///         C = alpha * op(A) * op(B) + beta * C
/// where `alpha` and `beta` are scalars, A, B - sparse matrices and C - dense matrix.
/// op(A), op(B) are operators defining if the matrices A and B used as is in the computations
/// or is being transposed.
///
/// op(A) is `m` x `k` matrix;
/// op(B) is `k` x `p` matrix;
/// C is `m` x `p` matrix.
///
/// @tparam Float   The type of elements in the matrix A and the vectors x and y.
///                 The `Float` type should be at least `float` or `double`.
/// @tparam co      Data layout in the matrix C.
///                 The `co` value should be `ndorder::c` or `ndorder::f`.
///
/// @param queue        The SYCL* queue object.
/// @param transpose_a  Defines if the sparse matrix A transposed or not.
///                     If `transpose_a` == `transpose::notrans` then op(A) = A.
///                     If `transpose_a` == `transpose::trans` then op(A) = transpose(A).
/// @param transpose_b  Defines if the sparse matrix B transposed or not.
///                     If `transpose_b` == `transpose::notrans` then op(B) =B.
///                     If `transpose_b` == `transpose::trans` then op(B) = transpose(B).
/// @param alpha        Specifies the scalar `alpha`.
/// @param a            Handle to object containing sparse matrix A.
/// @param b            Handle to object containing sparse matrix B.
/// @param beta         Specifies the scalar `beta`.
/// @param c            Dense output matrix that has `m` rows and `p` columns.
/// @param dependencies Events indicating availability of the matrix A and the vectors x and y
///                     for reading or writing.
template <typename Float, ndorder co>
sycl::event matmatd(sycl::queue &queue,
                    transpose transpose_a,
                    transpose transpose_b,
                    const Float alpha,
                    sparse_matrix_handle& a,
                    sparse_matrix_handle& b,
                    const Float beta,
                    ndview<Float, 2, co>& c,
                    const std::vector<sycl::event> &dependencies = {});

#endif // ifdef ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
