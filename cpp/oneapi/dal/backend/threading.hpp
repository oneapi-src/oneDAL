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
// #include <tbb/task_arena.h>
// #include <tbb/task_scheduler_observer.h>
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

template<typename Functor>
class NonVoidTaskAdapter : public daal::services::internal::thread_pinner_task_t {
public:
    typedef std::invoke_result_t<Functor> result_t;

    explicit NonVoidTaskAdapter(Functor&& func, result_t* place)
        : functor{ std::forward<Functor>(func) }, placement{ place } {}

    void operator() () final {
        auto temp = functor();
        new (placement) result_t{ std::move(temp) };
    }

private:
    Functor&& functor;
    result_t* placement;
};

template<typename Functor>
class VoidTaskAdapter : public daal::services::internal::thread_pinner_task_t {
public:
    explicit VoidTaskAdapter(Functor&& func) 
        : functor{ std::forward<Functor>(func) } {}

    void operator() () final {
        functor();
    }

private:
    Functor&& functor;
};

/*template<typename Functor>
auto wrap_functor(Functor&& func) {
  using res_t = std::result_of_t<Functor>;

  if constexpr (std::is_same_v<res_t, void>) {
    return VoidTaskAdapter<Functor>(std::forward<Functor>(functor));
  }
  else {

  }
}*/

class task_executor {
    threading_policy policy_;
    tbb::task_arena *task_arena_;
    daal::services::internal::thread_pinner_t *thread_pinner_;
    tbb::task_arena* create_task_arena(const threading_policy& policy);
public:
    template<typename F> 
    auto execute(F&& f) -> decltype(f());
    task_executor(threading_policy &policy) {
      policy_ = policy;
      task_arena_ = create_task_arena(policy_);
    }
};

template<typename F> 
auto task_executor::execute(F&& f) -> decltype(f()){
    if (this->policy_.thread_pinning) {
        using res_t = decltype(f());
        constexpr auto is_void = std::is_same_v<res_t, void>;
        if constexpr (is_void) {
          VoidTaskAdapter wrapper(std::forward<F>(f));
          thread_pinner_->execute(wrapper);
          return;
        }
        else {
          res_t result;
          NonVoidTaskAdapter wrapper(std::forward<F>(f), &result);
          thread_pinner_->execute(wrapper);
          return result;
        }
    }
    else {
        return this->task_arena_->execute(f);
    }
}

} // namespace oneapi::dal::backend
