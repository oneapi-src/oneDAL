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

#include <catch2/catch.hpp>

#include <tuple>
#include <memory>
#include <iostream>
#include <type_traits>

#include "oneapi/dal/train.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/compute.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/test/engine/macro.hpp"

// Disable clang-format as it dramatically
// affects redability of macro definitions

// Workaround DPC++ Compiler's warning on unused
// variable declared by Catch2's TEST_CASE macro
#ifdef __clang__
#define _TS_DISABLE_UNUSED_VARIABLE  \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wunused-variable\"")

#define _TS_ENABLE_UNUSED_VARIABLE _Pragma("clang diagnostic pop")

#undef TEST_CASE
#define TEST_CASE(...)                   \
    _TS_DISABLE_UNUSED_VARIABLE          \
    INTERNAL_CATCH_TESTCASE(__VA_ARGS__) \
    _TS_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE
#define TEMPLATE_TEST_CASE(...)                    \
    _TS_DISABLE_UNUSED_VARIABLE                    \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE(__VA_ARGS__) \
    _TS_ENABLE_UNUSED_VARIABLE

#undef TEMPLATE_TEST_CASE_METHOD
#define TEMPLATE_TEST_CASE_METHOD(...)                    \
    _TS_DISABLE_UNUSED_VARIABLE                    \
    INTERNAL_CATCH_TEMPLATE_TEST_CASE_METHOD(__VA_ARGS__) \
    _TS_ENABLE_UNUSED_VARIABLE
#endif // __clang__

#ifdef ONEDAL_DATA_PARALLEL
#define DECLARE_TEST_POLICY(policy_name) oneapi::dal::test::engine::device_test_policy policy_name
#else
#define DECLARE_TEST_POLICY(policy_name) oneapi::dal::test::engine::host_test_policy policy_name
#endif

namespace oneapi::dal::test::engine {

template <typename Float>
inline double get_tolerance(double double_tol, double float_tol) {
    if constexpr (std::is_same_v<std::decay_t<Float>, double>) {
        return double_tol;
    }
    return float_tol;
}

class host_test_policy {};

template <typename... Args>
inline auto train(host_test_policy& policy, Args&&... args) {
    return dal::train(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto infer(host_test_policy& policy, Args&&... args) {
    return dal::infer(std::forward<Args>(args)...);
}

template <typename... Args>
inline auto compute(host_test_policy& policy, Args&&... args) {
    return dal::compute(std::forward<Args>(args)...);
}

#ifdef ONEDAL_DATA_PARALLEL
class test_queue_provider {
public:
    static test_queue_provider& get_instance();

    sycl::queue& get_global_queue() {
        if (!queue_) {
            throw internal_error{ "Test queue provider is not initialized" };
        }
        return *queue_;
    }

    void init(const sycl::queue& queue) {
        queue_.reset(new sycl::queue{ queue });
    }

    void reset() {
        queue_.reset();
    }

private:
    test_queue_provider() = default;

    std::unique_ptr<sycl::queue> queue_;
};

inline const sycl::queue& get_global_queue() {
    return test_queue_provider::get_instance().get_global_queue();
}

class device_test_policy {
public:
    device_test_policy(const sycl::queue& queue) : queue_(queue) {}

    device_test_policy() : queue_(get_global_queue()) {}

    sycl::queue& get_queue() {
        return queue_;
    }

private:
    sycl::queue queue_;
};

template <typename... Args>
inline auto train(device_test_policy& policy, Args&&... args) {
    return dal::train(policy.get_queue(), std::forward<Args>(args)...);
}

template <typename... Args>
inline auto infer(device_test_policy& policy, Args&&... args) {
    return dal::infer(policy.get_queue(), std::forward<Args>(args)...);
}

template <typename... Args>
inline auto compute(device_test_policy& policy, Args&&... args) {
    return dal::compute(policy.get_queue(), std::forward<Args>(args)...);
}
#endif

class policy_fixture {
public:
    auto& get_policy() {
        return policy_;
    }

private:
    DECLARE_TEST_POLICY(policy_);
};

class algo_fixture : public policy_fixture {
public:
    template <typename... Args>
    auto train(Args&&... args) {
        return oneapi::dal::test::engine::train(get_policy(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto infer(Args&&... args) {
        return oneapi::dal::test::engine::infer(get_policy(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto compute(Args&&... args) {
        return oneapi::dal::test::engine::compute(get_policy(), std::forward<Args>(args)...);
    }
};

} // namespace oneapi::dal::test::engine
