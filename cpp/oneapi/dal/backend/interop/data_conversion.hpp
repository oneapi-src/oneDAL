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

#include <daal/include/data_management/data/internal/conversion.h>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::backend::interop {

// TODO: Remove using namespace
using namespace daal::data_management;

inline features::IndexNumType getIndexNumType(data_type t) {
    switch (t) {
        case data_type::int32: return features::DAAL_INT32_S;
        case data_type::int64: return features::DAAL_INT64_S;
        case data_type::uint32: return features::DAAL_INT32_U;
        case data_type::uint64: return features::DAAL_INT64_U;
        case data_type::float32: return features::DAAL_FLOAT32;
        case data_type::float64: return features::DAAL_FLOAT64;
        default: return features::DAAL_OTHER_T;
    }
}

inline data_type get_dal_data_type(features::IndexNumType t) {
    switch (t) {
        case features::DAAL_INT32_S: return data_type::int32;
        case features::DAAL_INT64_S: return data_type::int64;
        case features::DAAL_INT32_U: return data_type::uint32;
        case features::DAAL_INT64_U: return data_type::uint64;
        case features::DAAL_FLOAT32: return data_type::float32;
        case features::DAAL_FLOAT64: return data_type::float64;
        default: return data_type::float32;
    }
}

inline internal::ConversionDataType getConversionDataType(data_type t) {
    switch (t) {
        case data_type::int32: return internal::DAAL_INT32;
        case data_type::float32: return internal::DAAL_SINGLE;
        case data_type::float64: return internal::DAAL_DOUBLE;
        default: return internal::DAAL_OTHER;
    }
}

template <typename DownCast, typename UpCast, typename... Args>
inline void daal_convert_dispatcher(data_type src_type,
                                    data_type dst_type,
                                    DownCast&& dcast,
                                    UpCast&& ucast,
                                    Args&&... args) {
    auto from_type = getIndexNumType(src_type);
    auto to_type = getConversionDataType(dst_type);

    auto check_types = [](auto from_type, auto to_type) {
        if (from_type == features::DAAL_OTHER_T || to_type == internal::DAAL_OTHER) {
            throw invalid_argument(dal::detail::error_messages::unsupported_conversion_types());
        }
    };

    if (getConversionDataType(dst_type) == internal::DAAL_OTHER &&
        getConversionDataType(src_type) != internal::DAAL_OTHER) {
        from_type = getIndexNumType(dst_type);
        to_type = getConversionDataType(src_type);

        check_types(from_type, to_type);
        dcast(from_type, to_type)(std::forward<Args>(args)...);
    }
    else {
        check_types(from_type, to_type);
        ucast(from_type, to_type)(std::forward<Args>(args)...);
    }
}

inline void daal_convert(const void* src,
                         void* dst,
                         data_type src_type,
                         data_type dst_type,
                         std::int64_t element_count) {
    daal_convert_dispatcher(src_type,
                            dst_type,
                            internal::getVectorDownCast,
                            internal::getVectorUpCast,
                            element_count,
                            src,
                            dst);
}

inline void daal_convert(const void* src,
                         void* dst,
                         data_type src_type,
                         data_type dst_type,
                         std::int64_t src_stride,
                         std::int64_t dst_stride,
                         std::int64_t element_count) {
    daal_convert_dispatcher(src_type,
                            dst_type,
                            internal::getVectorStrideDownCast,
                            internal::getVectorStrideUpCast,
                            element_count,
                            src,
                            src_stride,
                            dst,
                            dst_stride);
}

} // namespace oneapi::dal::backend::interop
