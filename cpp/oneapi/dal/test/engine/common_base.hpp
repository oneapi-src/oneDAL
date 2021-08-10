#pragma once

#ifdef ONEDAL_DATA_PARALLEL
#define DECLARE_TEST_POLICY(policy_name) oneapi::dal::test::engine::device_test_policy policy_name
#else
#define DECLARE_TEST_POLICY(policy_name) oneapi::dal::test::engine::host_test_policy policy_name
#endif

#include <tuple>
#include <memory>
#include <iostream>
#include <type_traits>

#include <fmt/core.h>

#include "oneapi/dal/train.hpp"
#include "oneapi/dal/infer.hpp"
#include "oneapi/dal/compute.hpp"
#include "oneapi/dal/exceptions.hpp"
//#include "oneapi/dal/test/engine/catch.hpp"
#include "oneapi/dal/test/engine/macro.hpp"
#include "oneapi/dal/test/engine/type_traits.hpp"

namespace oneapi::dal::test::engine {

class host_test_policy {
public:
    bool is_cpu() const {
        return true;
    }

    bool is_gpu() const {
        return false;
    }

    bool has_native_float64() const {
        return true;
    }
};

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

    sycl::queue& get_global_queue() const {
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
    device_test_policy(const sycl::queue& queue) {
        init(queue);
    }

    device_test_policy() {}

    sycl::queue& get_queue() {
        if (!queue_) {
            init(get_global_queue());
        }
        return *(queue_);
    }

    bool is_cpu() {
        return get_queue().get_device().is_cpu() || get_queue().get_device().is_host();
    }

    bool is_gpu() {
        return get_queue().get_device().is_gpu();
    }

    bool has_native_float64();

private:
    void init(const sycl::queue& queue) {
        queue_ = std::make_unique<sycl::queue>(queue);
    }

    std::unique_ptr<sycl::queue> queue_;
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

template <typename T>
struct type2str {
    static const char* name() {
        return "Unknown";
    }
};

#define INSTANTIATE_TYPE_MAP(T)       \
    template <>                       \
    const char* type2str<T>::name() { \
        return #T;                    \
    }

#endif

} // namespace oneapi::dal::test::engine