/* file: error_handling.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#ifndef __DAAL_ONEAPI_INTERNAL_ERROR_HANDLING_H__
#define __DAAL_ONEAPI_INTERNAL_ERROR_HANDLING_H__

#include <CL/cl.h>
#include <CL/sycl.hpp>

#if defined(_WIN32) || defined(_WIN64)
    #define DAAL_DISABLE_LEVEL_ZERO
#endif

#ifndef DAAL_DISABLE_LEVEL_ZERO
    #include <ze_api.h>
#endif //DAAL_ENABLE_LEVEL_ZERO

#include "services/internal/error_handling_helpers.h"
#include "services/error_indexes.h"
#include "services/daal_string.h"

#define DAAL_CHECK_OPENCL(cl_error, statusPtr, ...)                     \
    {                                                                   \
        if (cl_error != CL_SUCCESS)                                     \
        {                                                               \
            if (statusPtr != nullptr)                                   \
            {                                                           \
                statusPtr->add(convertOpenClErrorToErrorPtr(cl_error)); \
            }                                                           \
            return __VA_ARGS__;                                         \
        }                                                               \
    }

#ifndef DAAL_DISABLE_LEVEL_ZERO
    #define DAAL_CHECK_LEVEL_ZERO(ze_error, statusPtr, ...)                    \
        {                                                                      \
            if (ze_error != ZE_RESULT_SUCCESS)                                 \
            {                                                                  \
                if (statusPtr != nullptr)                                      \
                {                                                              \
                    statusPtr->add(convertLevelZeroErrorToErrorPtr(ze_error)); \
                }                                                              \
                return __VA_ARGS__;                                            \
            }                                                                  \
        }
#endif //DAAL_ENABLE_LEVEL_ZERO

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace interface1
{
inline services::String getOpenClErrorDescription(cl_int clError)
{
#define OPENCL_ERROR_CASE(x) \
case x: return services::String(#x);
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
    return services::String("Unknown OpenCL error");

#undef OPENCL_ERROR_CASE
}

inline services::ErrorPtr convertOpenClErrorToErrorPtr(cl_int clError)
{
    return services::Error::create(services::ErrorID::ErrorExecutionContext, services::ErrorDetailID::OpenCL, getOpenClErrorDescription(clError));
}

#ifndef DAAL_DISABLE_LEVEL_ZERO
inline services::String getLevelZeroErrorDescription(ze_result_t zeError)
{
    #define LEVEL_ZERO_ERROR_CASE(x) \
    case x: return services::String(#x);
    switch (zeError)
    {
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_SUCCESS);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_NOT_READY);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_DEVICE_LOST);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_NOT_AVAILABLE);
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
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_OVERLAPPING_REGIONS);
        LEVEL_ZERO_ERROR_CASE(ZE_RESULT_ERROR_UNKNOWN);
    }
    return services::String("Unknown LevelZero error");

    #undef LEVEL_ZERO_ERROR_CASE
}

inline services::ErrorPtr convertLevelZeroErrorToErrorPtr(ze_result_t zeError)
{
    return services::Error::create(services::ErrorID::ErrorExecutionContext, services::ErrorDetailID::LevelZero,
                                   getLevelZeroErrorDescription(zeError));
}
#endif //DAAL_ENABLE_LEVEL_ZERO

inline void convertSyclExceptionToStatus(cl::sycl::exception const & e, services::Status * statusPtr)
{
    if (statusPtr != NULL)
    {
        statusPtr->add(services::Error::create(services::ErrorID::ErrorExecutionContext, services::ErrorDetailID::Sycl, services::String(e.what())));
    }
}
} // namespace interface1

using interface1::convertOpenClErrorToErrorPtr;
using interface1::convertSyclExceptionToStatus;

} // namespace internal
} // namespace oneapi
} // namespace daal

#endif
