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
#include "src/threading/service_thread_pinner.h"
#include "src/services/service_topo.h"

namespace oneapi::dal::backend {

#if !defined(DAAL_THREAD_PINNING_DISABLED)

struct task_daal : public daal::services::internal::thread_pinner_task_t {
    task_daal(task& f) : task_oneapi(f) {}
    void operator()() final {
        task_oneapi();
    }
    task& task_oneapi;
};

class thread_pinner_impl {
public:
    thread_pinner_impl() {
        daal_thread_pinner = std::shared_ptr<daal::services::internal::thread_pinner_t>(
            daal::services::internal::getThreadPinner(true, read_topology, delete_topology));
    }
    thread_pinner_impl(tbb::task_arena& task_arena) {
        daal_thread_pinner = std::shared_ptr<daal::services::internal::thread_pinner_t>(
            daal::services::internal::getThreadPinnerFromTaskArena(true,
                                                                   read_topology,
                                                                   delete_topology,
                                                                   task_arena));
    }
    void execute(task& task_) {
        auto task_daal_ = task_daal(task_);
        daal_thread_pinner->execute(task_daal_);
    }

private:
    std::shared_ptr<daal::services::internal::thread_pinner_t> daal_thread_pinner;
};

#endif

tbb::task_arena* task_executor::create_task_arena() {
    auto constraints = tbb::task_arena::constraints();
    if (policy_.max_threads_per_core) {
        constraints.set_max_threads_per_core(policy_.max_threads_per_core);
    }
    if (policy_.max_concurrency) {
        constraints.set_max_concurrency(policy_.max_concurrency);
    }
    static tbb::task_arena task_arena{ constraints };
#if !defined(DAAL_THREAD_PINNING_DISABLED)
    if (policy_.thread_pinning) {
        using daal::services::internal::thread_pinner_t;
        thread_pinner_t* thread_pinner_ptr =
            daal::services::internal::getThreadPinnerFromTaskArena(true,
                                                                   read_topology,
                                                                   delete_topology,
                                                                   task_arena);
        return thread_pinner_ptr->get_task_arena();
    }
#endif
    return &task_arena;
}

} // namespace oneapi::dal::backend
