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

#ifdef ONEDAL_DATA_PARALLEL
#include <CL/sycl.hpp>
#endif

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::backend {

#ifdef ONEDAL_DATA_PARALLEL

inline bool is_known_usm_pointer_type(const sycl::queue& queue, const void* pointer) {
    auto pointer_type = sycl::get_pointer_type(pointer, queue.get_context());
    return pointer_type != sycl::usm::alloc::unknown;
}

inline void* malloc(const sycl::queue& queue, std::size_t size, const sycl::usm::alloc& alloc) {
    auto ptr = sycl::malloc(size, queue, alloc);
    if (ptr == nullptr) {
        if (alloc == sycl::usm::alloc::shared || alloc == sycl::usm::alloc::host) {
            throw dal::host_bad_alloc();
        }
        else if (alloc == sycl::usm::alloc::device) {
            throw dal::device_bad_alloc();
        }
        else {
            throw dal::invalid_argument(detail::error_messages::unknown_usm_pointer_type());
        }
    }
    return ptr;
}

inline void free(const sycl::queue& queue, void* pointer) {
    ONEDAL_ASSERT(pointer == nullptr || is_known_usm_pointer_type(queue, pointer));
    sycl::free(pointer, queue);
}

template <typename T>
inline T* malloc(const sycl::queue& queue, std::int64_t count, const sycl::usm::alloc& alloc) {
    ONEDAL_ASSERT_MUL_OVERFLOW(std::size_t, sizeof(T), count);
    const std::size_t bytes_count = sizeof(T) * count;
    return static_cast<T*>(malloc(queue, bytes_count, alloc));
}

template <typename T>
class usm_deleter {
public:
    explicit usm_deleter(const sycl::queue& queue) : queue_(queue) {}

    void operator()(T* ptr) const {
        free(queue_, ptr);
    }

    sycl::queue& get_queue() {
        return queue_;
    }

    const sycl::queue& get_queue() const {
        return queue_;
    }

private:
    sycl::queue queue_;
};

#endif

} // namespace oneapi::dal::backend
