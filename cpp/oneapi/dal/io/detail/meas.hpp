/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <chrono>
#include <type_traits>

template <typename Function, typename... Arguments>
inline double measure(Function func, Arguments&... args) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    func(std::forward<Arguments>(args)...);
    const auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}
template <typename Function, typename... Arguments>
inline std::pair<double, typename std::result_of<Function(Arguments...)>::type> measure_with_result(
    Function func,
    Arguments&... args) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    auto function_result = func(std::forward<Arguments>(args)...);
    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto time_delta = std::chrono::duration<double>(end_time - start_time).count();
    return std::pair(time_delta, function_result);
}

// #ifndef CR_INIT()
#define CR_INIT()                                        \
    auto t0 = std::chrono::high_resolution_clock::now(); \
    auto t1 = std::chrono::high_resolution_clock::now();
#define CR_ST() t0 = std::chrono::high_resolution_clock::now();
#define CR_END(name)                                                                \
    t1 = std::chrono::high_resolution_clock::now();                                 \
    {                                                                               \
        std::chrono::duration<double> elapsed = t1 - t0;                            \
        std::cout << "Elapsed time " << #name << ": " << elapsed.count() << " s\n"; \
    }
// #endif
