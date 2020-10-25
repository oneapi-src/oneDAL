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
#include <memory>
#include <iostream>
#include <type_traits>

#include "oneapi/dal/train.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/compute.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/test/macro.hpp"

// Disable clang-format as it dramatically
// affects redability of macro definitions

#ifdef ONEDAL_DATA_PARALLEL
#define DECLARE_TEST_POLICY(policy_name) oneapi::dal::test::device_test_policy policy_name
#else
#define DECLARE_TEST_POLICY(policy_name) oneapi::dal::test::host_test_policy policy_name
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

    const sycl::queue& get_global_queue() const {
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

} // namespace oneapi::dal::test
