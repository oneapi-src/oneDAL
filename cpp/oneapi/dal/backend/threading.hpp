/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "tbb/tbb.h"
#include <tbb/task_arena.h>
#include <tbb/task_scheduler_observer.h>
#include "src/threading/service_thread_pinner.h"
#include "src/services/service_topo.h"

namespace oneapi::dal::backend {
struct threading_policy {
    bool thread_pinning;
    int max_threads_per_core;

    threading_policy(bool thread_pinning_ = false, int max_threads_per_core_ = 0)
            : thread_pinning(thread_pinning_),
              max_threads_per_core(max_threads_per_core_) {}
};

tbb::task_arena* create_task_arena(const threading_policy& policy);

} // namespace oneapi::dal::backend
