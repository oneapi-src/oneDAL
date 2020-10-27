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

#include "oneapi/dal/detail/memory.hpp"

namespace oneapi::dal {

template <typename T>
class empty_delete {
public:
    void operator()(T*) const noexcept {}
};

template <typename T, typename Policy>
class default_delete {
public:
    explicit default_delete(const Policy& policy) : policy_(policy) {}

    void operator()(T* data) const {
        detail::free(policy_, data);
    }

private:
    std::remove_reference_t<Policy> policy_;
};

template <typename T>
inline auto make_default_delete(const detail::default_host_policy& policy) {
    return default_delete<T, detail::default_host_policy>{ policy };
}

#ifdef ONEDAL_DATA_PARALLEL

template <typename T>
inline auto make_default_delete(const detail::data_parallel_policy& policy) {
    return default_delete<T, detail::data_parallel_policy>{ policy };
}

#endif

} // namespace oneapi::dal
