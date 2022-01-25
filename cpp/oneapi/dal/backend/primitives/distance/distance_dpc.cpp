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

#include "oneapi/dal/backend/primitives/distance/distance.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, ndorder order1, ndorder order2>
void check_inputs(const ndview<Float, 2, order1>& inp1,
                  const ndview<Float, 2, order2>& inp2,
                  const ndview<Float, 2>& out) {
    ONEDAL_ASSERT(inp1.has_data());
    ONEDAL_ASSERT(inp2.has_data());
    ONEDAL_ASSERT(out.has_mutable_data());
    ONEDAL_ASSERT(inp1.get_dimension(0) == out.get_dimension(0));
    ONEDAL_ASSERT(inp2.get_dimension(0) == out.get_dimension(1));
    ONEDAL_ASSERT(inp1.get_dimension(1) == inp2.get_dimension(1));
}

#define INSTANTIATE(F, A, B)                                    \
    template void check_inputs<F, A, B>(const ndview<F, 2, A>&, \
                                        const ndview<F, 2, B>&, \
                                        const ndview<F, 2>&);

#define INSTANTIATE_F(A, B)  \
    INSTANTIATE(float, A, B) \
    INSTANTIATE(double, A, B)

#define INSTANTIATE_A(B)         \
    INSTANTIATE_F(ndorder::c, B) \
    INSTANTIATE_F(ndorder::f, B)

INSTANTIATE_A(ndorder::c)
INSTANTIATE_A(ndorder::f)

} // namespace oneapi::dal::backend::primitives
