/* file: daal_ze_module_helper.h */
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

/*
//++
// Helper for ze_module management
//--
*/

#ifdef DAAL_SYCL_INTERFACE
    #ifndef __DAAL_ONEAPI_INTERNAL_DAAL_ZE_MODULE_HELPER_H__
        #define __DAAL_ONEAPI_INTERNAL_DAAL_ZE_MODULE_HELPER_H__

        #include <CL/cl.h>
        #include "sycl/internal/daal_level_zero_common.h"

        #ifndef DAAL_DISABLE_LEVEL_ZERO

            #include "sycl/internal/error_handling.h"
            #include "services/daal_shared_ptr.h"
            #include "services/internal/dynamic_lib_helper.h"

            #define _P(...)              \
                do                       \
                {                        \
                    printf(__VA_ARGS__); \
                    printf("\n");        \
                    fflush(0);           \
                } while (0)

namespace daal
{
namespace oneapi
{
namespace internal
{
namespace interface1
{
            #ifdef __linux__

static const char * zeLoaderName = "libze_loader.so";
static const int libLoadFlags    = RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL;

            #elif defined(_WIN64)

static const char * zeLoaderName = "ze_loader.dll";
static const int libLoadFlags    = 0;

            #else

                #error "Level Zero support is unavailable for this platform"

            #endif
static const char * zeModuleCreateFuncName  = "zeModuleCreate";
static const char * zeModuleDestroyFuncName = "zeModuleDestroy";

class ZeModuleHelper : public Base
{
public:
    ZeModuleHelper()                       = delete;
    ZeModuleHelper(const ZeModuleHelper &) = delete;
    ZeModuleHelper(ze_context_handle_t zeContext, ze_device_handle_t zeDevice, cl_program clProgram, services::Status * status = nullptr)
        : _zeContext(zeContext), _zeDevice(zeDevice), _binarySize(0), _pBinary(nullptr)
    {
        services::Status localStatus;

        static services::internal::DynamicLibHelper zeLib(zeLoaderName, libLoadFlags, &localStatus);
        if (!localStatus.ok())
        {
            services::internal::tryAssignStatus(status, localStatus);
            return;
        }

        static zeModuleCreateFT stZeModuleCreateF = zeLib.getSymbol<zeModuleCreateFT>(zeModuleCreateFuncName, &localStatus);
        if (!localStatus.ok())
        {
            services::internal::tryAssignStatus(status, localStatus);
            return;
        }

        _zeModuleCreateF = stZeModuleCreateF;

        DAAL_CHECK_OPENCL(clGetProgramInfo(clProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &_binarySize, NULL), status);

        _pBinary = (unsigned char *)daal::services::daal_malloc(_binarySize);
        if (_pBinary == nullptr)
        {
            services::internal::tryAssignStatus(status, services::ErrorMemoryAllocationFailed);
            return;
        }

        DAAL_CHECK_OPENCL(clGetProgramInfo(clProgram, CL_PROGRAM_BINARIES, sizeof(_pBinary), &_pBinary, NULL), status);
    }

    ~ZeModuleHelper() { daal::services::daal_free(_pBinary); }

    void zeModuleCreate(ze_module_handle_t * moduleLevelZero, services::Status * status = nullptr)
    {
        ze_module_desc_t zeDesc;
        zeDesc.stype        = ZE_STRUCTURE_TYPE_MODULE_DESC;
        zeDesc.format       = ZE_MODULE_FORMAT_NATIVE;
        zeDesc.inputSize    = _binarySize;
        zeDesc.pInputModule = _pBinary;
        zeDesc.pBuildFlags  = "";
        zeDesc.pConstants   = nullptr;

        DAAL_CHECK_LEVEL_ZERO(_zeModuleCreateF(_zeContext, _zeDevice, &zeDesc, moduleLevelZero, nullptr), status);
    }

private:
    ze_context_handle_t _zeContext;
    ze_device_handle_t _zeDevice;
    zeModuleCreateFT _zeModuleCreateF;
    unsigned char * _pBinary;
    size_t _binarySize;
};

typedef services::SharedPtr<ZeModuleHelper> ZeModuleHelperPtr;

} // namespace interface1
} // namespace internal
} // namespace oneapi
} // namespace daal

        #endif // DAAL_DISABLE_LEVEL_ZERO

    #endif // __DAAL_ONEAPI_INTERNAL_DAAL_ZE_MODULE_HELPER_H__
#endif     // DAAL_SYCL_INTERFACE
