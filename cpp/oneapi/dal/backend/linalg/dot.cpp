/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/backend/linalg/dot.hpp"
#include "oneapi/dal/backend/mkl/blas.hpp"

namespace oneapi::dal::backend::linalg {

namespace mkl = oneapi::dal::backend::mkl;

template <typename Float>
void dot_op<Float>::operator()(const context_cpu &ctx,
                               const matrix<Float> &a,
                               const matrix<Float> &b,
                               matrix<Float> &c,
                               Float alpha,
                               Float beta) const {
    const bool is_c_trans = (c.get_layout() == layout::row_major);
    if (is_c_trans) {
        const bool is_a_trans = (a.get_layout() == layout::column_major);
        const bool is_b_trans = (b.get_layout() == layout::column_major);
        mkl::gemm<Float>(ctx,
                         is_b_trans,
                         is_a_trans,
                         c.get_column_count(),
                         c.get_row_count(),
                         a.get_column_count(),
                         alpha,
                         b.get_data(),
                         b.get_stride(),
                         a.get_data(),
                         a.get_stride(),
                         beta,
                         c.get_mutable_data(),
                         c.get_stride());
    }
    else {
        const bool is_a_trans = (a.get_layout() == layout::row_major);
        const bool is_b_trans = (b.get_layout() == layout::row_major);
        mkl::gemm<Float>(ctx,
                         is_a_trans,
                         is_b_trans,
                         c.get_row_count(),
                         c.get_column_count(),
                         a.get_column_count(),
                         alpha,
                         a.get_data(),
                         a.get_stride(),
                         b.get_data(),
                         b.get_stride(),
                         beta,
                         c.get_mutable_data(),
                         c.get_stride());
    }
}

#define INSTANTIATE(Float) template struct dot_op<Float>;

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::linalg
