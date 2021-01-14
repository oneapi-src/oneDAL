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

#pragma once

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/linalg/matrix.hpp"

namespace oneapi::dal::backend::linalg {

template <typename Float>
struct dot_op {
    void operator()(const context_cpu& ctx,
                    const matrix<Float>& a,
                    const matrix<Float>& b,
                    matrix<Float>& c,
                    Float alpha,
                    Float beta) const;
};

template <typename Float>
inline matrix<Float> dot(const matrix<Float>& a, const matrix<Float>& b, Float alpha = Float(1)) {
    auto c = matrix<Float>::empty({ a.get_row_count(), b.get_column_count() });
    dot(a, b, c, alpha);
    return c;
}

template <typename Float>
inline void dot(const matrix<Float>& a,
                const matrix<Float>& b,
                matrix<Float>& c,
                Float alpha = Float(1),
                Float beta = Float(0)) {
    dot_op<Float>{}(context_cpu{}, a, b, c, alpha, beta);
}

} // namespace oneapi::dal::backend::linalg
