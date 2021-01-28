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

#pragma once

#include "oneapi/dal/test/engine/linalg/matrix.hpp"

namespace oneapi::dal::test::engine::linalg {

template <typename Float, layout lyt_a, layout lyt_b, layout lyt_c>
struct dot_op {
    void operator()(const matrix<Float, lyt_a>& a,
                    const matrix<Float, lyt_b>& b,
                    matrix<Float, lyt_c>& c,
                    Float alpha,
                    Float beta) const;
};

template <typename Float, layout lyt_a, layout lyt_b, layout lyt_c>
inline void dot(const matrix<Float, lyt_a>& a,
                const matrix<Float, lyt_b>& b,
                matrix<Float, lyt_c>& c,
                Float alpha = Float(1),
                Float beta = Float(0)) {
    dot_op<Float, lyt_a, lyt_b, lyt_c>{}(a, b, c, alpha, beta);
}

template <typename Float, layout lyt_a, layout lyt_b, layout lyt_c = layout::row_major>
inline matrix<Float, lyt_c> dot(const matrix<Float, lyt_a>& a,
                                const matrix<Float, lyt_b>& b,
                                Float alpha = Float(1)) {
    auto c = matrix<Float, lyt_c>::empty({ a.get_row_count(), b.get_column_count() });
    dot(a, b, c, alpha);
    return c;
}

} // namespace oneapi::dal::test::engine::linalg
