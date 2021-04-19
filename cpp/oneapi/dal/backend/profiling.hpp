/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/backend/common.hpp"

#ifdef ONEDAL_ENABLE_PROFILING
#include <chrono>
#include <string>
#include <iomanip>
#include <iostream>
#endif

#ifdef ONEDAL_ENABLE_PROFILING

#define ONEDAL_TIMER_BEGIN(domain, id)                            \
    oneapi::dal::backend::timer id##__timer__{ #domain "." #id }; \
    id##__timer__.start();

#define ONEDAL_TIMER_END(id) id##__timer__.stop().print();

#define ONEDAL_TIMER(domain, id) \
    if (auto id##__timer__ =     \
            oneapi::dal::backend::timer_guard<oneapi::dal::backend::timer>{ #domain "." #id })
#else
#define ONEDAL_TIMER_BEGIN(id)
#define ONEDAL_TIMER_END(id)
#define ONEDAL_TIMER(id)
#endif

namespace oneapi::dal::backend {

#ifdef ONEDAL_ENABLE_PROFILING

class timer {
public:
    using clock_t = std::chrono::high_resolution_clock;
    using time_point_t = typename clock_t::time_point;

    explicit timer(const std::string& name) : name_(name) {}

    timer& start() {
        time_begin_ = clock_t::now();
        return *this;
    }

    timer& stop() {
        time_end_ = clock_t::now();
        return *this;
    }

    timer& print() {
        std::cout << name_ << ": " //
                  << std::setprecision(2) //
                  << duration_milliseconds() << "ms" //
                  << std::endl
                  << std::flush;
        return *this;
    }

    double duration_nanoseconds() {
        using std::chrono::nanoseconds;
        using std::chrono::duration_cast;
        return duration_cast<nanoseconds>(time_end_ - time_begin_).count();
    }

    double duration_microseconds() {
        return duration_nanoseconds() / decimal;
    }

    double duration_milliseconds() {
        return duration_microseconds() / decimal;
    }

    double duration_seconds() {
        return duration_milliseconds() / decimal;
    }

private:
    time_point_t time_begin_ = time_point_t::min();
    time_point_t time_end_ = time_point_t::min();
    std::string name_;

    static constexpr double decimal = 1000.0;
};

template <typename Timer>
class timer_guard {
public:
    template <typename... Args>
    timer_guard(Args&&... args) : timer_(std::forward<Args>(args)...) {
        timer_.start();
    }

    constexpr operator bool() {
        return true;
    }

    ~timer_guard() {
        timer_.stop();
        timer_.print();
    }

private:
    Timer timer_;
};

#endif

} // namespace oneapi::dal::backend
