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
    #ifndef __DAAL_SERVICES_INTERNAL_SYCL_DAAL_ZE_MODULE_HELPER_H__
        #define __DAAL_SERVICES_INTERNAL_SYCL_DAAL_ZE_MODULE_HELPER_H__

        #include "services/internal/sycl/daal_level_zero_common.h"

        #ifndef DAAL_DISABLE_LEVEL_ZERO

            #include <CL/sycl.hpp>
            #include "services/internal/sycl/error_handling.h"
            #include "services/daal_shared_ptr.h"
            #include "services/internal/dynamic_lib_helper.h"

            #if (defined(__SYCL_COMPILER_VERSION) && (__SYCL_COMPILER_VERSION >= 20200701))
                #include <CL/sycl/backend/level_zero.hpp>
            #else
                #include <CL/sycl/backend/Intel_level0.hpp>
            #endif

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
    ZeModuleHelper(cl::sycl::queue & deviceQueue, size_t binarySize, const uint8_t * pBinary, Status & status)
        : _program(deviceQueue.get_context())
    {
        static DynamicLibHelper zeLib(zeLoaderName, libLoadFlags, status);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        static zeModuleCreateFT stZeModuleCreateF = zeLib.getSymbol<zeModuleCreateFT>(zeModuleCreateFuncName, status);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        _zeModuleCreateF = stZeModuleCreateF;

        ze_module_desc_t desc;
        desc.stype        = ZE_STRUCTURE_TYPE_MODULE_DESC;
        desc.format       = ZE_MODULE_FORMAT_NATIVE;
        desc.inputSize    = binarySize;
        desc.pInputModule = pBinary;
        desc.pBuildFlags  = "";
        desc.pConstants   = nullptr;

        ze_module_handle_t _moduleLevelZero;
        DAAL_CHECK_LEVEL_ZERO(
            _zeModuleCreateF(deviceQueue.get_context().get_native<cl::sycl::backend::level_zero>(),
                             deviceQueue.get_device().get_native<cl::sycl::backend::level_zero>(), &desc, &_moduleLevelZero, nullptr),
            status);

        _program = cl::sycl::level_zero::make<cl::sycl::program>(deviceQueue.get_context(), _moduleLevelZero);
    }

    ~ZeModuleHelper() = default;

    cl::sycl::program getZeProgram() { return _program; }

private:
    cl::sycl::program _program;
    zeModuleCreateFT _zeModuleCreateF;
};

typedef services::SharedPtr<ZeModuleHelper> ZeModuleHelperPtr;

} // namespace interface1
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal

        #endif // DAAL_DISABLE_LEVEL_ZERO

    #endif // __DAAL_SERVICES_INTERNAL_SYCL_DAAL_ZE_MODULE_HELPER_H__
#endif     // DAAL_SYCL_INTERFACE
