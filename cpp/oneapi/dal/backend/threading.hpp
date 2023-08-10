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
#include "src/threading/service_thread_pinner.h"
#include "src/services/service_topo.h"
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::backend {

using detail::threading_policy;

template<typename F>
class non_void_task : public daal::services::internal::thread_pinner_task_t {
public:
    typedef std::invoke_result_t<F> result_t;

    explicit non_void_task(F&& functor, result_t* result_ptr)
        : functor_{ std::forward<F>(functor) }, result_ptr_{ result_ptr } {}

    void operator()() final {
        auto result = functor_();
        new (result_ptr_) result_t{ std::move(result) };
    }

private:
    F&& functor_;
    result_t* result_ptr_;
};

template<typename F>
class void_task : public daal::services::internal::thread_pinner_task_t {
public:
    explicit void_task(F&& functor) 
        : functor_{ std::forward<F>(functor) } {}

    void operator()() final {
        functor_();
    }

private:
    F&& functor_;
};

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
        using result_t = decltype(f());
        constexpr auto is_void = std::is_same_v<result_t, void>;
        if constexpr (is_void) {
          void_task wrapper(std::forward<F>(f));
          thread_pinner_->execute(wrapper);
          return;
        }
        else {
          result_t result;
          non_void_task wrapper(std::forward<F>(f), &result);
          thread_pinner_->execute(wrapper);
          return result;
        }
    }
    else {
        return this->task_arena_->execute(f);
    }
}

} // namespace oneapi::dal::backend
