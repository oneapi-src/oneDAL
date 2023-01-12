/* file: level_zero_module_sycl.h */
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

#ifndef __DAAL_SERVICES_INTERNAL_SYCL_DAAL_ZE_MODULE_SYCL_H__
#define __DAAL_SERVICES_INTERNAL_SYCL_DAAL_ZE_MODULE_SYCL_H__

#ifndef DAAL_SYCL_INTERFACE
    #error "DAAL_SYCL_INTERFACE must be defined to include this file"
#endif

#ifdef DAAL_DISABLE_LEVEL_ZERO
    #error "DAAL_DISABLE_LEVEL_ZERO must be undefined to include this file"
#endif

#include <sycl/sycl.hpp>

#include "services/daal_shared_ptr.h"
#include "services/internal/dynamic_lib_helper.h"
#include "services/internal/sycl/error_handling_sycl.h"
#include "services/internal/sycl/level_zero_common.h"

#if (defined(__SYCL_COMPILER_VERSION) && (__SYCL_COMPILER_VERSION >= 20211025))
    #include <ext/oneapi/backend/level_zero.hpp>
#elif (defined(__SYCL_COMPILER_VERSION) && (__SYCL_COMPILER_VERSION >= 20200701))
    #include <CL/sycl/backend/level_zero.hpp>
#else
    #include <CL/sycl/backend/Intel_level0.hpp>
#endif

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
#define DAAL_LEVEL_ZERO_VERSION_SUFF          ".1"
#define DAAL_LEVEL_ZERO_LIB_VERSIONED_NAME(n) #n DAAL_LEVEL_ZERO_VERSION_SUFF

#ifdef __linux__
static const char * zeLoaderName = DAAL_LEVEL_ZERO_LIB_VERSIONED_NAME(libze_loader.so);
static const int libLoadFlags    = RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL;
#elif defined(_WIN64)
static const char * zeLoaderName = "ze_loader.dll";
static const int libLoadFlags    = LOAD_LIBRARY_SEARCH_SYSTEM32;
#else
    #error "Level Zero support is unavailable for this platform"
#endif

static const char * zeModuleCreateFuncName  = "zeModuleCreate";
static const char * zeModuleDestroyFuncName = "zeModuleDestroy";
static const char * zeKernelCreateFuncName  = "zeKernelCreate";
static const char * zeKernelDestroyFuncName = "zeKernelDestroy";
class ZeModule;

class ZeKernel : public Base
{
    friend ZeModule;

public:
    ZeKernel(const ZeModule &)             = delete;
    ZeKernel & operator=(const ZeModule &) = delete;

    ze_kernel_handle_t get() const { return _kernelLevelZero; }

private:
    explicit ZeKernel(ze_module_handle_t moduleLevelZero, const char * kernelName, Status & status)
    {
        static DynamicLibHelper zeLib(zeLoaderName, libLoadFlags, status);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        static zeKernelCreateFT stZeKernelCreateF = zeLib.getSymbol<zeKernelCreateFT>(zeKernelCreateFuncName, status);
        DAAL_CHECK_STATUS_RETURN_VOID_IF_FAIL(status);

        _zeKernelCreateF = stZeKernelCreateF;

        ze_kernel_desc_t desc;
        desc.stype       = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        desc.pNext       = nullptr;
        desc.flags       = ze_kernel_flags_t(0);
        desc.pKernelName = kernelName;

        DAAL_CHECK_LEVEL_ZERO(_zeKernelCreateF(moduleLevelZero, &desc, &_kernelLevelZero), status);
    }

    zeKernelCreateFT _zeKernelCreateF;

    ze_kernel_handle_t _kernelLevelZero;
};

typedef SharedPtr<ZeKernel> ZeKernelPtr;

class ZeModule : public Base
{
public:
    static SharedPtr<ZeModule> create(::sycl::queue & deviceQueue, size_t binarySize, const uint8_t * pBinary, Status & status)
    {
        auto ptr = new ZeModule(deviceQueue, binarySize, pBinary, status);
        if (!status)
        {
            if (ptr) delete ptr;
            ptr = nullptr;
        }
        else if (!ptr)
            status |= ErrorMemoryAllocationFailed;
        return SharedPtr<ZeModule>(ptr);
    }

    ZeModule(const ZeModule &)             = delete;
    ZeModule & operator=(const ZeModule &) = delete;

    ZeKernelPtr createKernel(const char * kernelName, Status & status)
    {
        auto ptr = new ZeKernel(_moduleLevelZero, kernelName, status);
        if (!status)
        {
            if (ptr) delete ptr;
            ptr = nullptr;
        }
        else if (!ptr)
            status |= ErrorMemoryAllocationFailed;
        return ZeKernelPtr(ptr);
    }

    ze_module_handle_t get() const { return _moduleLevelZero; }

private:
    explicit ZeModule(::sycl::queue & deviceQueue, size_t binarySize, const uint8_t * pBinary, Status & status)
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
        desc.pNext        = nullptr;

        DAAL_CHECK_LEVEL_ZERO(_zeModuleCreateF(::sycl::get_native< ::sycl::backend::ext_oneapi_level_zero>(deviceQueue.get_context()),
                                               ::sycl::get_native< ::sycl::backend::ext_oneapi_level_zero>(deviceQueue.get_device()), &desc,
                                               &_moduleLevelZero, nullptr),
                              status);
    }

    zeModuleCreateFT _zeModuleCreateF;

    ze_module_handle_t _moduleLevelZero;
};

typedef SharedPtr<ZeModule> ZeModulePtr;

} // namespace interface1
} // namespace sycl
} // namespace internal
} // namespace services
} // namespace daal
/// \endcond

#endif
