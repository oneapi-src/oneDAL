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

#include "oneapi/dal/table/backend/interop/data_conversion.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace daal_dm = daal::data_management;

namespace oneapi::dal::backend::interop {

daal_dm::features::IndexNumType get_daal_index_num_type(data_type t) {
    switch (t) {
        case data_type::int32: return daal_dm::features::DAAL_INT32_S;
        case data_type::int64: return daal_dm::features::DAAL_INT64_S;
        case data_type::uint32: return daal_dm::features::DAAL_INT32_U;
        case data_type::uint64: return daal_dm::features::DAAL_INT64_U;
        case data_type::float32: return daal_dm::features::DAAL_FLOAT32;
        case data_type::float64: return daal_dm::features::DAAL_FLOAT64;
        default: return daal_dm::features::DAAL_OTHER_T;
    }
}

daal_dm::internal::ConversionDataType get_daal_conversion_data_type(data_type t) {
    switch (t) {
        case data_type::int32: return daal_dm::internal::DAAL_INT32;
        case data_type::float32: return daal_dm::internal::DAAL_SINGLE;
        case data_type::float64: return daal_dm::internal::DAAL_DOUBLE;
        default: return daal_dm::internal::DAAL_OTHER;
    }
}

template <typename DownCast, typename UpCast, typename... Args>
void daal_convert_dispatcher(data_type src_type,
                             data_type dest_type,
                             DownCast&& dcast,
                             UpCast&& ucast,
                             Args&&... args) {
    auto from_type = get_daal_index_num_type(src_type);
    auto to_type = get_daal_conversion_data_type(dest_type);

    auto check_types = [](auto from_type, auto to_type) {
        if (from_type == daal_dm::features::DAAL_OTHER_T || to_type == daal_dm::internal::DAAL_OTHER) {
            throw internal_error("unsupported conversion types");
        }
    };

    if (get_daal_conversion_data_type(dest_type) == daal_dm::internal::DAAL_OTHER &&
        get_daal_conversion_data_type(src_type) != daal_dm::internal::DAAL_OTHER) {
        from_type = get_daal_index_num_type(dest_type);
        to_type = get_daal_conversion_data_type(src_type);

        check_types(from_type, to_type);
        dcast(from_type, to_type)(std::forward<Args>(args)...);
    }
    else {
        check_types(from_type, to_type);
        ucast(from_type, to_type)(std::forward<Args>(args)...);
    }
}

void daal_convert(const void* src,
                  void* dst,
                  data_type src_type,
                  data_type dst_type,
                  std::int64_t size) {
    daal_convert_dispatcher(src_type,
                            dst_type,
                            daal_dm::internal::getVectorDownCast,
                            daal_dm::internal::getVectorUpCast,
                            size,
                            src,
                            dst);
}

void daal_convert(const void* src,
                  void* dst,
                  data_type src_type,
                  data_type dst_type,
                  std::int64_t src_stride,
                  std::int64_t dst_stride,
                  std::int64_t size) {
    daal_convert_dispatcher(src_type,
                            dst_type,
                            daal_dm::internal::getVectorStrideDownCast,
                            daal_dm::internal::getVectorStrideUpCast,
                            size,
                            src,
                            src_stride,
                            dst,
                            dst_stride);
}

} // namespace oneapi::dal::backend::interop
