/* file: error_handling_helpers.h */
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

#ifndef __SERVICES_INTERNAL_ERROR_HANDLING_HELPERS_H__
#define __SERVICES_INTERNAL_ERROR_HANDLING_HELPERS_H__

#include "services/error_handling.h"
#include "services/daal_shared_ptr.h"

namespace daal
{
namespace services
{
namespace internal
{
inline void tryAssignStatus(Status * status, const Status & statusToAssign)
{
    if (status)
    {
        *status |= statusToAssign;
    }
}

inline void tryAssignStatusAndThrow(Status * status, const Status & statusToAssign)
{
    if (status)
    {
        *status |= statusToAssign;
        services::throwIfPossible(*status);
    }
    else
    {
        services::throwIfPossible(statusToAssign);
    }
}

template <typename T>
inline SharedPtr<T> wrapShared(T * object, Status * status = NULL)
{
    if (!object)
    {
        tryAssignStatus(status, ErrorMemoryAllocationFailed);
    }
    return SharedPtr<T>(object);
}

template <typename T>
inline SharedPtr<T> wrapSharedAndTryThrow(T * object, Status * status = NULL)
{
    if (!object)
    {
        tryAssignStatusAndThrow(status, ErrorMemoryAllocationFailed);
    }
    return SharedPtr<T>(object);
}

} // namespace internal
} // namespace services
} // namespace daal

#endif
