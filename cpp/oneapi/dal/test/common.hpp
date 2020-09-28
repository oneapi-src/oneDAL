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

#include <tuple>
#include <iostream>
#include <type_traits>

#include <fmt/core.h>
#include <catch2/catch.hpp>

#include "oneapi/dal/train.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/compute.hpp"
#include "oneapi/dal/test/macro.hpp"

// Workaround DPC++ Compiler's warning on unused
// variable declared by Catch2's TEST_CASE macro
#ifdef __clang__
#define ONEDAL_TEST_DISABLE_UNUSED_VARIABLE \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wunused-variable\"") \

#define ONEDAL_TEST_ENABLE_UNUSED_VARIABLE \
    _Pragma("clang diagnostic pop")

#undef TEST_CASE
#define TEST_CASE(...) \
    ONEDAL_TEST_DISABLE_UNUSED_VARIABLE \
    INTERNAL_CATCH_TESTCASE(__VA_ARGS__) \
    ONEDAL_TEST_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE
#define TEMPLATE_TEST_CASE(...) \
    ONEDAL_TEST_DISABLE_UNUSED_VARIABLE \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE(__VA_ARGS__) \
    ONEDAL_TEST_ENABLE_UNUSED_VARIABLE
#endif

#define _TS_STRINGIFY_ALGO_TYPES(Float, Method) \
    " - (" _TS_STRINGIFY(Float) ", " _TS_STRINGIFY(Method) ")"

#define _TS_STRINGIFY_ALGO_TAGS(Float, Method) \
    _TS_STRINGIFY([Float][Method])

#define _TS_HANDLE_ALGO_TYPES_(ctx, x) \
    _TS_HANDLE_ALGO_TYPES(_TS_GET_0(ctx), _TS_GET_1(ctx), \
                          _TS_GET_0(x),   _TS_GET_1(x))

#define _TS_HANDLE_ALGO_TYPES(test_case, function, Float, Method) \
    TEST_CASE(test_case _TS_STRINGIFY_ALGO_TYPES(Float, Method), \
                        _TS_STRINGIFY_ALGO_TAGS(Float, Method)) { \
        function<Float, Method>(); \
    }

#define REGISTER_ALGO_TEST_CASE(test_case, function, float_list, method_list) \
    _TS_FOR_EACH((test_case, function), _TS_HANDLE_ALGO_TYPES_, _TS_COMB(float_list, method_list))

#define ALGO_TEST_CASE(test_case, float_list, method_list) \
    template <typename Float, typename Method> \
    void test_algo_ ## __LINE__(); \
    REGISTER_ALGO_TEST_CASE(test_case, test_algo_ ## __LINE__, float_list, method_list) \
    template <typename Float, typename Method> \
    void test_algo_ ## __LINE__()

#ifdef ONEAPI_DAL_DATA_PARALLEL
#define DECLARE_TEST_POLICY(policy_name) \
    oneapi::dal::test::device_test_policy policy_name
#else
#define DECLARE_TEST_POLICY(policy_name) \
    oneapi::dal::test::host_test_policy policy_name
#endif

namespace oneapi::dal::test {

template <typename Float>
inline double get_tolerance(double double_tol, double float_tol) {
    if constexpr (std::is_same_v<std::decay_t<Float>, double>) {
        return double_tol;
    }
    return float_tol;
}

class host_test_policy {};

template <typename... Args>
inline auto train(host_test_policy& policy, Args&& ...args) {
    return dal::train(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto infer(host_test_policy& policy, Args&& ...args) {
    return dal::infer(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto compute(host_test_policy& policy, Args&& ...args) {
    return dal::compute(std::forward<Args>(args)...);
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
class device_test_policy {
public:
    device_test_policy(const sycl::queue& queue)
        : queue_(queue) {}

    device_test_policy()
        : queue_(sycl::gpu_selector{}) {}

    sycl::queue& get_queue() {
        return queue_;
    }

private:
    sycl::queue queue_;
};

template <typename... Args>
inline auto train(device_test_policy& policy, Args&& ...args) {
    return dal::train(policy.get_queue(), std::forward<Args>(args)...);
}

template <typename... Args>
inline auto infer(device_test_policy& policy, Args&& ...args) {
    return dal::infer(policy.get_queue(), std::forward<Args>(args)...);
}

template <typename... Args>
inline auto compute(device_test_policy& policy, Args&& ...args) {
    return dal::compute(policy.get_queue(), std::forward<Args>(args)...);
}
#endif

} // oneapi::dal::test
