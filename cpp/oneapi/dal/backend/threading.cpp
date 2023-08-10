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
#define TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION 1

#include "oneapi/dal/backend/threading.hpp"


namespace oneapi::dal::backend {

tbb::task_arena* task_executor::create_task_arena(const oneapi::dal::detail::threading_policy& policy) {
    if (policy.max_threads_per_core) {
        static tbb::task_arena task_arena{ tbb::task_arena::constraints{}.set_max_threads_per_core(
            policy.max_threads_per_core) };
        if (policy.thread_pinning) {
            using daal::services::internal::thread_pinner_t;
            thread_pinner_t* thread_pinner_ptr =
                daal::services::internal::getThreadPinnerFromTaskArena(true,
                                                          read_topology,
                                                          delete_topology,
                                                          task_arena);
            return thread_pinner_ptr->get_task_arena();
        }
        return &task_arena;
    }
    else {
        static tbb::task_arena task_arena{};
        if (policy.thread_pinning) {
            using daal::services::internal::thread_pinner_t;
            thread_pinner_t* thread_pinner_ptr =
                daal::services::internal::getThreadPinnerFromTaskArena(true,
                                                          read_topology,
                                                          delete_topology,
                                                          task_arena);
            return thread_pinner_ptr->get_task_arena();
        }
        return &task_arena;
    }
}

} // namespace oneapi::dal::backend
