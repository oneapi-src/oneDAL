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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/memory.hpp"

namespace oneapi::dal::v1 {

void* base::operator new(std::size_t size) {
    return detail::malloc(detail::default_host_policy{}, size);
}

void* base::operator new(std::size_t, void* where) {
    return where;
}

void* base::operator new[](std::size_t size) {
    return detail::calloc(detail::default_host_policy{}, size);
}

void* base::operator new[](std::size_t, void* where) {
    return where;
}

void base::operator delete(void* ptr, std::size_t) {
    detail::free(detail::default_host_policy{}, ptr);
}

void base::operator delete[](void* ptr, std::size_t) {
    detail::free(detail::default_host_policy{}, ptr);
}

} // namespace oneapi::dal::v1
