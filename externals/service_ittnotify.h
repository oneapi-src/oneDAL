/* file: service_ittnotify.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Wrappers for common ittnotify functions.
//--
*/

#ifndef __SERVICE_ITTNOTIFY_H__
#define __SERVICE_ITTNOTIFY_H__

#ifdef __DAAL_ITTNOTIFY_ENABLE__
    #include <ittnotify.h>

namespace daal
{
namespace internal
{
namespace ittnotify
{
class Domain
{
public:
    explicit Domain(const char * name) : _itt_domain(__itt_domain_create(name)) {}

    __itt_domain * Get() const { return _itt_domain; }

private:
    __itt_domain * _itt_domain;
};

class StringHandle
{
public:
    explicit StringHandle(const char * name) : _handle(__itt_string_handle_create(name)) {}

    __itt_string_handle * Get() const { return _handle; }

private:
    __itt_string_handle * _handle;
};

inline void Pause()
{
    __itt_pause();
}

inline void Resume()
{
    __itt_resume();
}

inline void TaskBegin(const Domain & domain, const StringHandle & handle)
{
    __itt_task_begin(domain.Get(), __itt_null, __itt_null, handle.Get());
}

inline void TaskEnd(const Domain & domain)
{
    __itt_task_end(domain.Get());
}

class ScopedTask
{
public:
    ScopedTask(const Domain & domain, const StringHandle & handle) : _domain(domain) { TaskBegin(domain, handle); }

    ~ScopedTask() { TaskEnd(_domain); }

private:
    const Domain & _domain;
};

} // namespace ittnotify
} // namespace internal
} // namespace daal

    // There must be only one domain on the translation unit regarding to this macro
    #define DAAL_ITTNOTIFY_DOMAIN(name) static daal::internal::ittnotify::Domain __ittnotify_domain(#name)

    #define DAAL_ITTNOTIFY_SCOPED_TASK(name)                                            \
        static daal::internal::ittnotify::StringHandle __ittnotify_stringhandle(#name); \
        daal::internal::ittnotify::ScopedTask __ittnotify_task(__ittnotify_domain, __ittnotify_stringhandle)

#else
    #include "service_profiler.h"

    #define DAAL_ITTNOTIFY_CONCAT2(x, y) x##y
    #define DAAL_ITTNOTIFY_CONCAT(x, y)  DAAL_ITTNOTIFY_CONCAT2(x, y)

    #define DAAL_ITTNOTIFY_UNIQUE_ID __LINE__

    #define DAAL_ITTNOTIFY_DOMAIN(name)
    #define DAAL_ITTNOTIFY_SCOPED_TASK(name) \
        daal::internal::ProfilerTask DAAL_ITTNOTIFY_CONCAT(__profiler_taks__, DAAL_ITTNOTIFY_UNIQUE_ID) = daal::internal::Profiler::startTask(#name);

#endif // __DAAL_ITTNOTIFY_ENABLE__
#endif // __SERVICE_ITTNOTIFY_H__
