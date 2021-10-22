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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::preview::spmd {

namespace v1 {

class communication_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    const char* what() const noexcept override {
        return std::runtime_error::what();
    }
};

enum class reduce_op { max, min, sum };

} // namespace v1

using v1::communication_error;
using v1::reduce_op;

namespace device_memory_access {
namespace v1 {

struct usm {};
struct none {};
} // namespace v1

using v1::usm;
using v1::none;

} // namespace device_memory_access

template <typename T>
using enable_if_device_memory_accessible_t =
    std::enable_if_t<dal::detail::is_one_of_v<T, device_memory_access::usm>>;
template <typename T>
using enable_if_device_memory_not_accessible_t =
    std::enable_if_t<dal::detail::is_one_of_v<T, device_memory_access::none>>;

} // namespace oneapi::dal::preview::spmd
