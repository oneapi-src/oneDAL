/* file: dynamic_lib_helper.h */
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
// Helper for dynamic libs management  
//--
*/

#ifndef __DAAL_DYNAMIC_LIB_HELPER_H__
#define __DAAL_DYNAMIC_LIB_HELPER_H__

#ifdef __linux__

    #include <dlfcn.h>

namespace daal
{
namespace services
{
namespace internal
{
/** @ingroup services_internal
 * @{
 */

/**
 * <a name="DAAL-CLASS-SERVICES__INTERNAL__DYNAMICLIBHELPER"></a>
 * \brief  Class that provides routines for managing dynamic library
 */

class DynamicLibHelper final
{
public:
    DynamicLibHelper()                         = delete;
    DynamicLibHelper(const DynamicLibHelper &) = delete;
    DynamicLibHelper(const char * libName, int flag, services::Status * status = nullptr)
    {
        services::Status localStatus;
        _handle = dlopen(libName, flag);

        if (!_handle)
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorCanNotLoadDynamicLibrary);
            return;
        }
    }

    ~DynamicLibHelper() { dlclose(_handle); };

    template <typename T>
    T getSymbol(const char * symName, services::Status * status = nullptr)
    {
        void * sym   = dlsym(_handle, symName);
        char * error = dlerror();

        if (nullptr != error)
        {
            services::internal::tryAssignStatusAndThrow(status, services::ErrorCanNotLoadDynamicLibrarySymbol);
            return nullptr;
        }

        return (T)sym;
    }

private:
    void * _handle;
};

} // namespace internal
} // namespace services
} // namespace daal

#endif // __linux__
#endif // __DAAL_DYNAMIC_LIB_HELPER_H__
