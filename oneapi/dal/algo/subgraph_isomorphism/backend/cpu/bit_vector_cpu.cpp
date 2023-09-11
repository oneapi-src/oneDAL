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

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/bit_vector.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template void or_equal<__CPU_TAG__>(std::uint8_t* vec,
                                    const std::uint8_t* pa,
                                    const std::int64_t size);
template void and_equal<__CPU_TAG__>(std::uint8_t* vec,
                                     const std::uint8_t* pa,
                                     const std::int64_t size);

template void or_equal<__CPU_TAG__>(std::uint8_t* vec,
                                    const std::int64_t* bit_index,
                                    const std::int64_t list_size);
template void and_equal<__CPU_TAG__>(std::uint8_t* vec,
                                     const std::int64_t* bit_index,
                                     const std::int64_t bit_size,
                                     const std::int64_t list_size,
                                     std::int64_t* tmp_array,
                                     const std::int64_t tmp_size);
template void set<__CPU_TAG__>(std::uint8_t* vec, std::int64_t size, const std::uint8_t byte_val);
template class bit_vector<__CPU_TAG__>;

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
