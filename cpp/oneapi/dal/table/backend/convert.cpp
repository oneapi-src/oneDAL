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

#include "oneapi/dal/table/backend/convert.hpp"
#include "oneapi/dal/backend/interop/data_conversion.hpp"

namespace oneapi::dal::backend {

void convert_vector(const detail::default_host_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t element_count) {
    interop::daal_convert(src, dst, src_type, dst_type, element_count);
}

void convert_vector(const detail::default_host_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t src_stride,
                    std::int64_t dst_stride,
                    std::int64_t element_count) {
    interop::daal_convert(src, dst, src_type, dst_type, src_stride, dst_stride, element_count);
}

#ifdef ONEDAL_DATA_PARALLEL

void convert_vector(const detail::data_parallel_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t element_count) {
    convert_vector(detail::default_host_policy{}, src, dst, src_type, dst_type, element_count);
}

void convert_vector(const detail::data_parallel_policy& policy,
                    const void* src,
                    void* dst,
                    data_type src_type,
                    data_type dst_type,
                    std::int64_t src_stride,
                    std::int64_t dst_stride,
                    std::int64_t element_count) {
    convert_vector(detail::default_host_policy{},
                   src,
                   dst,
                   src_type,
                   dst_type,
                   src_stride,
                   dst_stride,
                   element_count);
}
#endif

} // namespace oneapi::dal::backend
