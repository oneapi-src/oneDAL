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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/misc.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/handle.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

/// Computes a sparse matrix - dense vector product:
///         y = alpha * op(A) + beta * y
/// where `alpha` and `beta` are scalars, A - sparse matrix and x, y - dense vectors.
/// op(A) is an operator defining if the matrix A used as is in the computations
/// or is being transposed.
///
/// op(A) is `m` x `k` matrix.
///
/// @tparam Float   The type of elements in the matrix A and the vectors x and y.
///                 The `Float` type should be at least `float` or `double`.
///
/// @param queue        The SYCL* queue object.
/// @param transpose_a  Defines if the sparse matrix A transposed or not.
///                     If `transpose_a` == `transpose::notrans` then op(A) = A.
///                     If `transpose_a` == `transpose::trans` then op(A) = transpose(A).
/// @param a            Handle to object containing sparse matrix A.
/// @param x            Dense 1-dimensional input vector that has `k` elements.
/// @param y            Dense 1-dimensional resulting vector that has `m` elements.
/// @param alpha        Specifies the scalar `alpha`.
/// @param beta         Specifies the scalar `beta`.
/// @param dependencies Events indicating availability of the matrix A and the vectors x and y
///                     for reading or writing.
template <typename Float>
sycl::event gemv(sycl::queue& queue,
                 transpose transpose_a,
                 sparse_matrix_handle& a,
                 const ndview<Float, 1>& x,
                 ndview<Float, 1>& y,
                 const Float alpha,
                 const Float beta,
                 const event_vector& dependencies = {});

/// Computes a sparse matrix - dense vector product:
///         y = op(A) * x
/// A - sparse matrix and x, y - dense vectors.
/// op(A) is an operator defining if the matrix A used as is in the computations
/// or is being transposed.
///
/// op(A) is `m` x `k` matrix.
///
/// @tparam Float   The type of elements in the matrix A and the vectors x and y.
///                 The `Float` type should be at least `float` or `double`.
///
/// @param queue        The SYCL* queue object.
/// @param transpose_a  Defines if the sparse matrix A transposed or not.
///                     If `transpose_a` == `transpose::notrans` then op(A) = A.
///                     If `transpose_a` == `transpose::trans` then op(A) = transpose(A).
/// @param a            Handle to object containing sparse matrix A.
/// @param x            Dense 1-dimensional input vector that has `k` elements.
/// @param y            Dense 1-dimensional resulting vector that has `m` elements.
/// @param dependencies Events indicating availability of the matrices A, B and C for reading
///                     or writing.
template <typename Float>
sycl::event gemv(sycl::queue& queue,
                 transpose transpose_a,
                 sparse_matrix_handle& a,
                 const ndview<Float, 1>& x,
                 ndview<Float, 1>& y,
                 const event_vector& dependencies = {}) {
    return gemv<Float>(queue, transpose_a, a, x, y, Float(1), Float(0), dependencies);
}

#endif // ifdef ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
