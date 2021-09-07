/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#pragma once

#ifdef ONEDAL_DATA_PARALLEL
#include <CL/sycl.hpp>
#endif

namespace oneapi::dal::detail {

class profiler_task {
public:
    profiler_task(const char * taskName);
    #ifdef ONEDAL_DATA_PARALLEL
    profiler_task(const char * taskName,  sycl::queue& q);
    #endif
    ~profiler_task();

private:
    const char * _taskName;
    #ifdef ONEDAL_DATA_PARALLEL
    sycl::queue task_queue;
    #endif
};

class profiler {
public:
    static profiler_task start_task(const char*);
    #ifdef ONEDAL_DATA_PARALLEL
    static profiler_task start_task(const char*, sycl::queue& q);
    #endif
    static void end_task(const char * taskName);
};

} // namespace oneapi::dal::detail
