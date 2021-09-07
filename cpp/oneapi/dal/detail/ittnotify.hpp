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

#ifdef __ONEDAL_ITTNOTIFY_ENABLE__
#include <ittnotify.h>

namespace oneapi::dal::detail {

class domain {
public:
    explicit domain(const char* name) : _itt_domain(__itt_domain_create(name)) {}

    __itt_domain* get() const {
        return _itt_domain;
    }

private:
    __itt_domain* _itt_domain;
};

class string_handle {
public:
    explicit string_handle(const char* name) : _handle(__itt_string_handle_create(name)) {}

    __itt_string_handle* get() const {
        return _handle;
    }

private:
    __itt_string_handle* _handle;
};

inline void pause() {
    __itt_pause();
}

inline void resume() {
    __itt_resume();
}

inline void task_begin(const domain& domain, const string_handle& handle) {
    __itt_task_begin(domain.get(), __itt_null, __itt_null, handle.get());
}

inline void task_end(const domain& domain) {
    __itt_task_end(domain.get());
}

class scoped_task {
public:
    scoped_task(const domain& domain, const string_handle& handle) : _domain(domain) {
        t(domain, handle);
    }

    ~scoped_task() {
        task_end(_domain);
    }

private:
    const domain& _domain;
};

} // namespace oneapi::dal::detail

// There must be only one domain on the translation unit regarding to this macro
#define ONEDAL_ITTNOTIFY_DOMAIN(name) static oneapi::dal::detail::domain __ittnotify_domain(#name)

#define ONEDAL_PROFILER_TASK(name)                                             \
    static oneapi::dal::detail::string_handle __ittnotify_stringhandle(#name); \
    oneapi::dal::detail::scoped_task __ittnotify_task(__ittnotify_domain, __ittnotify_stringhandle)

#else
#include "oneapi/dal/detail/profiler.hpp"

#define ONEDAL_PROFILER_CONCAT2(x, y) x##y
#define ONEDAL_PROFILER_CONCAT(x, y)  ONEDAL_PROFILER_CONCAT2(x, y)

#define ONEDAL_PROFILER_UNIQUE_ID __LINE__

#define ONEDAL_ITTNOTIFY_DOMAIN(name)
#define ONEDAL_PROFILER_TASK(...)                                                           \
    oneapi::dal::detail::profiler_task ONEDAL_PROFILER_CONCAT(__profiler_task__,            \
                                                              ONEDAL_ITTNOTIFY_UNIQUE_ID) = \
        oneapi::dal::detail::profiler::start_task(__VA_ARGS__);

#endif // __ONEDAL_ITTNOTIFY_ENABLE__
