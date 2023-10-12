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
#if !defined(__APPLE__)
#include "tbb/tbb.h"
#include "oneapi/dal/detail/policy.hpp"

namespace oneapi::dal::backend {

using detail::threading_policy;

struct task {
    virtual void operator()() = 0;
};

template <typename F>
class non_void_task : public task {
public:
    typedef std::invoke_result_t<F> result_t;

    explicit non_void_task(F&& functor, result_t* result_ptr)
            : functor_{ std::forward<F>(functor) },
              result_ptr_{ result_ptr } {}

    void operator()() final {
        auto result = functor_();
        new (result_ptr_) result_t{ std::move(result) };
    }

private:
    F&& functor_;
    result_t* result_ptr_;
};

template <typename F>
class void_task : public task {
public:
    explicit void_task(F&& functor) : functor_{ std::forward<F>(functor) } {}

    void operator()() final {
        functor_();
    }

private:
    F&& functor_;
};

#if !defined(DAAL_THREAD_PINNING_DISABLED)

class thread_pinner_impl;

class thread_pinner {
    friend detail::pimpl_accessor;

public:
    thread_pinner();
    thread_pinner(tbb::task_arena& task_arena);
    thread_pinner(const thread_pinner&) = default;
    thread_pinner(thread_pinner&&) = default;
    template <typename F>
    auto execute(F&& task_) -> decltype(task_()) {
        using result_t = decltype(task_());
        if constexpr (std::is_same_v<void, result_t>) {
            void_task wrapper(std::forward<F>(task_));
            execute(std::move(wrapper));
            return;
        }
        else {
            result_t result;
            non_void_task wrapper(std::forward<F>(task_), &result);
            execute(std::move(wrapper));
            return result;
        }
    }

private:
    void execute(const task& task_) const;
    detail::pimpl<thread_pinner_impl> impl_;
};

#endif

class task_executor {
    threading_policy policy_;
    tbb::task_arena* task_arena_;
#if !defined(DAAL_THREAD_PINNING_DISABLED)
    thread_pinner* thread_pinner_;
#endif
    tbb::task_arena* create_task_arena();

public:
    template <typename F>
    auto execute(F&& f) -> decltype(f());
    task_executor(threading_policy policy) {
        policy_ = policy;
        task_arena_ = create_task_arena();
    }
};

template <typename F>
auto task_executor::execute(F&& f) -> decltype(f()) {
#if !defined(DAAL_THREAD_PINNING_DISABLED)
    if (this->policy_.thread_pinning) {
        using result_t = decltype(f());
        if constexpr (std::is_same_v<result_t, void>) {
            void_task wrapper(std::forward<F>(f));
            thread_pinner_->execute(std::move(wrapper));
            return;
        }
        else {
            result_t result;
            non_void_task wrapper(std::forward<F>(f), &result);
            thread_pinner_->execute(std::move(wrapper));
            return result;
        }
    }
#endif
    return this->task_arena_->execute(f);
}

} // namespace oneapi::dal::backend

#endif
