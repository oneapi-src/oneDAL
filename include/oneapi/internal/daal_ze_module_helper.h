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

        #include "oneapi/internal/daal_level_zero_common.h"

        #ifndef DAAL_DISABLE_LEVEL_ZERO

            #include "oneapi/internal/error_handling.h"
            #include "services/daal_shared_ptr.h"
            #include "services/internal/dynamic_lib_helper.h"

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

static const char * zeLoaderName = "libze_loader.dll";
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
    ZeModuleHelper(ze_device_handle_t zeDevice, size_t binarySize, const uint8_t * pBinary, services::Status * status = nullptr)
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

        static zeModuleDestroyFT stZeModuleDestroy = zeLib.getSymbol<zeModuleDestroyFT>(zeModuleDestroyFuncName, &localStatus);
        if (!localStatus.ok())
        {
            services::internal::tryAssignStatus(status, localStatus);
            return;
        }

        _zeModuleDestroyF = stZeModuleDestroy;

        ze_module_desc_t desc { ZE_MODULE_DESC_VERSION_CURRENT };
        desc.format       = ZE_MODULE_FORMAT_NATIVE;
        desc.inputSize    = binarySize;
        desc.pInputModule = pBinary;
        desc.pBuildFlags  = "";
        desc.pConstants   = nullptr;

        DAAL_CHECK_LEVEL_ZERO(_zeModuleCreateF(zeDevice, &desc, &_moduleLevelZero, nullptr), status);
    }

    ~ZeModuleHelper() { _zeModuleDestroyF(_moduleLevelZero); }

    ze_module_handle_t get() { return _moduleLevelZero; }

private:
    ze_module_handle_t _moduleLevelZero;
    zeModuleCreateFT _zeModuleCreateF;
    zeModuleDestroyFT _zeModuleDestroyF;
};

typedef services::SharedPtr<ZeModuleHelper> ZeModuleHelperPtr;

} // namespace interface1
} // namespace internal
} // namespace oneapi
} // namespace daal

        #endif // DAAL_DISABLE_LEVEL_ZERO

    #endif // __DAAL_ONEAPI_INTERNAL_DAAL_ZE_MODULE_HELPER_H__
#endif     // DAAL_SYCL_INTERFACE
