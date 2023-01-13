/* file: error_handling_sycl.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_ERROR_HANDLING_SYCL_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_ERROR_HANDLING_SYCL_H__

#ifndef DAAL_SYCL_INTERFACE
    #error "DAAL_SYCL_INTERFACE must be defined to include this file"
#endif

#include <CL/cl.h>
#include <sycl/sycl.hpp>

#include "services/error_handling.h"
#ifndef DAAL_DISABLE_LEVEL_ZERO
    #include "services/internal/sycl/level_zero_common.h"
#endif

#define DAAL_CHECK_OPENCL(cl_error, status)                   \
    {                                                         \
        if (cl_error != CL_SUCCESS)                           \
        {                                                     \
            status |= convertOpenClErrorToErrorPtr(cl_error); \
            return;                                           \
        }                                                     \
    }

#ifndef DAAL_DISABLE_LEVEL_ZERO
    #define DAAL_CHECK_LEVEL_ZERO(ze_error, status)                  \
        {                                                            \
            if (ze_error != ZE_RESULT_SUCCESS)                       \
            {                                                        \
                status |= convertLevelZeroErrorToErrorPtr(ze_error); \
                return;                                              \
            }                                                        \
        }
#endif // DAAL_DISABLE_LEVEL_ZERO

/// \cond INTERNAL
namespace daal
{
namespace services
{
namespace internal
{
namespace sycl
{
namespace interface1
{
inline String getOpenClErrorDescription(cl_int clError)
{
#define OPENCL_ERROR_CASE(x) \
case x: return String(#x);
    switch (clError)
    {
        OPENCL_ERROR_CASE(CL_BUILD_PROGRAM_FAILURE);
        OPENCL_ERROR_CASE(CL_COMPILER_NOT_AVAILABLE);
        OPENCL_ERROR_CASE(CL_DEVICE_NOT_AVAILABLE);
        OPENCL_ERROR_CASE(CL_DEVICE_NOT_FOUND);
        OPENCL_ERROR_CASE(CL_IMAGE_FORMAT_MISMATCH);
        OPENCL_ERROR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        OPENCL_ERROR_CASE(CL_INVALID_ARG_INDEX);
        OPENCL_ERROR_CASE(CL_INVALID_ARG_SIZE);
        OPENCL_ERROR_CASE(CL_INVALID_ARG_VALUE);
        OPENCL_ERROR_CASE(CL_INVALID_BINARY);
        OPENCL_ERROR_CASE(CL_INVALID_BUFFER_SIZE);
        OPENCL_ERROR_CASE(CL_INVALID_BUILD_OPTIONS);
        OPENCL_ERROR_CASE(CL_INVALID_COMMAND_QUEUE);
        OPENCL_ERROR_CASE(CL_INVALID_CONTEXT);
        OPENCL_ERROR_CASE(CL_INVALID_DEVICE);
        OPENCL_ERROR_CASE(CL_INVALID_DEVICE_TYPE);
        OPENCL_ERROR_CASE(CL_INVALID_EVENT);
        OPENCL_ERROR_CASE(CL_INVALID_EVENT_WAIT_LIST);
        OPENCL_ERROR_CASE(CL_INVALID_GL_OBJECT);
        OPENCL_ERROR_CASE(CL_INVALID_GLOBAL_OFFSET);
        OPENCL_ERROR_CASE(CL_INVALID_HOST_PTR);
        OPENCL_ERROR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        OPENCL_ERROR_CASE(CL_INVALID_IMAGE_SIZE);
        OPENCL_ERROR_CASE(CL_INVALID_KERNEL_NAME);
        OPENCL_ERROR_CASE(CL_INVALID_KERNEL);
        OPENCL_ERROR_CASE(CL_INVALID_KERNEL_ARGS);
        OPENCL_ERROR_CASE(CL_INVALID_KERNEL_DEFINITION);
        OPENCL_ERROR_CASE(CL_INVALID_MEM_OBJECT);
        OPENCL_ERROR_CASE(CL_INVALID_OPERATION);
        OPENCL_ERROR_CASE(CL_INVALID_PLATFORM);
        OPENCL_ERROR_CASE(CL_INVALID_PROGRAM);
        OPENCL_ERROR_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
        OPENCL_ERROR_CASE(CL_INVALID_QUEUE_PROPERTIES);
        OPENCL_ERROR_CASE(CL_INVALID_SAMPLER);
        OPENCL_ERROR_CASE(CL_INVALID_VALUE);
        OPENCL_ERROR_CASE(CL_INVALID_WORK_DIMENSION);
        OPENCL_ERROR_CASE(CL_INVALID_WORK_GROUP_SIZE);
        OPENCL_ERROR_CASE(CL_INVALID_WORK_ITEM_SIZE);
        OPENCL_ERROR_CASE(CL_MAP_FAILURE);
        OPENCL_ERROR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        OPENCL_ERROR_CASE(CL_MEM_COPY_OVERLAP);
        OPENCL_ERROR_CASE(CL_OUT_OF_HOST_MEMORY);
        OPENCL_ERROR_CASE(CL_OUT_OF_RESOURCES);
        OPENCL_ERROR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
    }
    return String("Unknown OpenCL error");

#undef OPENCL_ERROR_CASE
}

inline ErrorPtr convertOpenClErrorToErrorPtr(cl_int clError)
{
    return Error::create(ErrorID::ErrorExecutionContext, ErrorDetailID::OpenCL, getOpenClErrorDescription(clError));
}

#ifndef DAAL_DISABLE_LEVEL_ZERO
inline String getLevelZeroErrorDescription(ze_result_t zeError)
{
    #define LEVEL_ZERO_ERROR_CASE(x) \
    case x: return String(#x);
    switch (zeError)
    {
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_SUCCESS);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_NOT_READY);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_DEVICE_LOST);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_MODULE_LINK_FAILURE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_NOT_AVAILABLE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_UNINITIALIZED);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_UNSUPPORTED_VERSION);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_ARGUMENT);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_NULL_HANDLE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_NULL_POINTER);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_SIZE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_UNSUPPORTED_SIZE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_ENUMERATION);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_NAME);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_OVERLAPPING_REGIONS);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_UNKNOWN);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_FORCE_UINT32);
    }
    return String("Unknown LevelZero error");

    #undef LEVEL_ZERO_ERROR_CASE
}

inline ErrorPtr convertLevelZeroErrorToErrorPtr(ze_result_t zeError)
{
    return Error::create(ErrorID::ErrorExecutionContext, ErrorDetailID::LevelZero, getLevelZeroErrorDescription(zeError));
}
#endif // DAAL_DISABLE_LEVEL_ZERO

inline Status convertSyclExceptionToStatus(const std::exception & ex)
{
    return Error::create(ErrorID::ErrorExecutionContext, ErrorDetailID::Sycl, String(ex.what()));
}

template <typename TryBody, typename CatchBody>
DAAL_FORCEINLINE auto catchSyclExceptions(Status & status, TryBody && tryBody, CatchBody && catchBody) -> decltype(tryBody())
{
    try
    {
        return tryBody();
    }
    catch (const std::bad_alloc &)
    {
        status |= ErrorMemoryAllocationFailed;
        return catchBody();
    }
    catch (const std::exception & ex)
    {
        status |= convertSyclExceptionToStatus(ex);
        return catchBody();
    }
    catch (...)
    {
        status |= UnknownError;
        return catchBody();
    }
}

template <typename Body>
DAAL_FORCEINLINE Status catchSyclExceptions(Body && body)
{
    Status status;
    return catchSyclExceptions(
        status,
        [&]() {
            body();
            return status;
        },
        [&]() { return status; });
}

} // namespace interface1

using interface1::convertOpenClErrorToErrorPtr;
using interface1::convertSyclExceptionToStatus;
using interface1::catchSyclExceptions;

} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
/// \endcond

#endif
