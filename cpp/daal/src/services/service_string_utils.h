/* file: service_string_utils.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#ifndef __KERNEL_SERVICE_STRING_UTILS_H__
#define __KERNEL_SERVICE_STRING_UTILS_H__

#include <cstdio>
#include <cstring>

#include "services/daal_string.h"
#include "src/services/service_defines.h"
#include "src/externals/service_service.h"

namespace daal
{
namespace services
{
namespace internal
{
template <class T>
void toStringBuffer(T number, char * buffer)
{}

template <>
void toStringBuffer<int>(int value, char * buffer)
{
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, value);
}

template <>
void toStringBuffer<double>(double value, char * buffer)
{
    daal::services::daal_int_to_string(buffer, DAAL_MAX_STRING_SIZE, value);
}

template <>
void toStringBuffer<String>(String value, char * buffer)
{
    daal::internal::ServiceInst::serv_strncpy_s(buffer, DAAL_MAX_STRING_SIZE, value.c_str(), DAAL_MAX_STRING_SIZE - value.length());
}

} // namespace internal
} // namespace services
} // namespace daal

#endif
