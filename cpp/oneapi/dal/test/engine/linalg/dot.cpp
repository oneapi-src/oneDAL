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

#include "oneapi/dal/test/engine/linalg/dot.hpp"
#include "oneapi/dal/test/engine/mkl/blas.hpp"

namespace oneapi::dal::test::engine::linalg {

namespace mkl = oneapi::dal::test::engine::mkl;

template <typename Float, layout lyt_a, layout lyt_b, layout lyt_c>
void dot_op<Float, lyt_a, lyt_b, lyt_c>::operator()(const matrix<Float, lyt_a> &a,
                                                    const matrix<Float, lyt_b> &b,
                                                    matrix<Float, lyt_c> &c,
                                                    Float alpha,
                                                    Float beta) const {
    ONEDAL_ASSERT(a.get_row_count() == c.get_row_count());
    ONEDAL_ASSERT(a.get_column_count() == b.get_row_count());
    ONEDAL_ASSERT(b.get_column_count() == c.get_column_count());

    constexpr bool is_c_trans = (lyt_c == layout::row_major);
    if constexpr (is_c_trans) {
        constexpr bool is_a_trans = (lyt_a == layout::column_major);
        constexpr bool is_b_trans = (lyt_b == layout::column_major);

        mkl::gemm<Float>(is_b_trans,
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
        constexpr bool is_a_trans = (lyt_a == layout::row_major);
        constexpr bool is_b_trans = (lyt_b == layout::row_major);

        mkl::gemm<Float>(is_a_trans,
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

#define INSTANTIATE(Float, lyt_a, lyt_b, lyt_c) \
    template struct dot_op<Float, layout::lyt_a, layout::lyt_b, layout::lyt_c>;

#define INSTANTIATE_LAYOUTS(Float)                            \
    INSTANTIATE(Float, row_major, row_major, row_major)       \
    INSTANTIATE(Float, row_major, row_major, column_major)    \
    INSTANTIATE(Float, row_major, column_major, row_major)    \
    INSTANTIATE(Float, row_major, column_major, column_major) \
    INSTANTIATE(Float, column_major, row_major, row_major)    \
    INSTANTIATE(Float, column_major, row_major, column_major) \
    INSTANTIATE(Float, column_major, column_major, row_major) \
    INSTANTIATE(Float, column_major, column_major, column_major)

INSTANTIATE_LAYOUTS(float)
INSTANTIATE_LAYOUTS(double)

} // namespace oneapi::dal::test::engine::linalg
