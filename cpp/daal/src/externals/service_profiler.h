/* file: service_profiler.h */
/*******************************************************************************
* Copyright 2019 Intel Corporation
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
//  Profiler for time measurement of kernels
//--
*/

#ifndef __SERVICE_PROFILER_H__
#define __SERVICE_PROFILER_H__

// #define ONEDAL_KERNEL_PROFILER

#ifdef ONEDAL_KERNEL_PROFILER
#include <ittnotify.h>
#endif

#define DAAL_ITTNOTIFY_CONCAT2(x, y) x##y
#define DAAL_ITTNOTIFY_CONCAT(x, y)  DAAL_ITTNOTIFY_CONCAT2(x, y)

#define DAAL_ITTNOTIFY_UNIQUE_ID __LINE__

#define DAAL_ITTNOTIFY_SCOPED_TASK(name) \
    daal::internal::ProfilerTask DAAL_ITTNOTIFY_CONCAT(__profiler_taks__, DAAL_ITTNOTIFY_UNIQUE_ID) = daal::internal::Profiler::startTask(#name);

namespace daal
{
namespace internal
{
class ProfilerTask
{
public:
    ProfilerTask(const char * taskName);
    ~ProfilerTask();

private:
    const char * _taskName;
#ifdef ONEDAL_KERNEL_PROFILER
    __itt_string_handle* _handle;
    __itt_domain* _domain;
#endif
};

// This class is a stub in the library. Its redefinition will be in Bechmarks
class Profiler
{
public:
    static ProfilerTask startTask(const char * taskName);
    static void endTask(const char * taskName);
#ifdef ONEDAL_KERNEL_PROFILER
    static Profiler* getInstance() {
        static Profiler  instance;
        return &instance;
    }

    static __itt_domain* getDomain() {
        return (getInstance())->_domain;
    }
private:
    Profiler() {
        _domain = __itt_domain_create("oneDAL");
    }
    ~Profiler() {}
    __itt_domain* _domain;
#endif
};

} // namespace internal
} // namespace daal

#endif // __SERVICE_PROFILER_H__
