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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

#include "oneapi/dal/table/backend/convert/common.hpp"
#include "oneapi/dal/table/backend/convert/copy_convert.hpp"

namespace oneapi::dal::backend {

template <typename CpuType, typename InpType, typename OutType>
void copy_convert(const detail::host_policy& policy,
                  const InpType* inp_ptr,
                  std::int64_t inp_str,
                  OutType* out_ptr,
                  std::int64_t out_str,
                  std::int64_t count);

template <typename CpuType>
void copy_convert(const detail::host_policy& policy,
                  const dal::byte_t* const* inp_ptrs,
                  const data_type* inp_types,
                  const std::int64_t* inp_strs,
                  dal::byte_t* const* out_ptrs,
                  const data_type* out_types,
                  const std::int64_t* out_strs,
                  const shape_t& shape);

#ifdef ONEDAL_DATA_PARALLEL

template <typename InputType>
sycl::event copy_convert(sycl::queue& queue,
                         const InputType* const* inp_pointers,
                         const std::int64_t* inp_strides,
                         dal::byte_t* const* out_pointers,
                         data_type out_type,
                         const std::int64_t* out_strides,
                         const shape_t& shape,
                         const std::vector<sycl::event>& deps = {});

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend
