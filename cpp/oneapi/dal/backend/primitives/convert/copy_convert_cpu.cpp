/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <utility>
#include <numeric>
#include <algorithm>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/threading.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/primitives/convert/common.hpp"
#include "oneapi/dal/backend/primitives/convert/copy_convert.hpp"

namespace oneapi::dal::backend::primitives {

/*template <typename CpuType>
void copy_convert(const detail::host_policy& policy,
                  const std::int64_t* input_offsets,
                  const data_type* input_types,
                  const dal::byte_t* input_data,
                  const shape_t& input_shape,
                  const data_type* output_type,
                  dal::byte_t output_data,
                  const shape_t& output_strides) {

}


template void copy_convert<__CPU_TAG__>(const detail::host_policy&,
                                        const std::int64_t*,
                                        const data_type*,
                                        const dal::byte_t*,
                                        const shape_t&,
                                        data_type,
                                        dal::byte_t*,
                                        const shape_t&);*/

} // namespace oneapi::dal::backend::primitives
