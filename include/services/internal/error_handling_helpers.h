/* file: error_handling_helpers.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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

inline void tryAssignStatus(Status *status, const Status &statusToAssign)
{
    if (status) { *status |= statusToAssign; }
}

inline void tryAssignStatusAndThrow(Status *status, const Status &statusToAssign)
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

template<typename T>
inline SharedPtr<T> wrapShared(T *object, Status *status = NULL)
{
    if (!object)
    {
        tryAssignStatus(status, ErrorMemoryAllocationFailed);
    }
    return SharedPtr<T>(object);
}

template<typename T>
inline SharedPtr<T> wrapSharedAndTryThrow(T *object, Status *status = NULL)
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
