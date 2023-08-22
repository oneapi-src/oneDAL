/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/spmd/common.hpp"

namespace oneapi::dal::preview::spmd {

error_holder::error_holder(const std::exception_ptr& actual)
        : std::runtime_error(oneapi::dal::detail::error_messages::spmd_error_holder_message()),
          actual_exception_(actual) {}

const char* error_holder::what() const noexcept {
    return std::runtime_error::what();
}

std::exception_ptr error_holder::get_actual() const {
    return actual_exception_;
}

void error_holder::rethrow_actual() const {
    if (actual_exception_) {
        std::rethrow_exception(actual_exception_);
    }
}

const char* coworker_error::what() const noexcept {
    return std::runtime_error::what();
}

const char* communication_error::what() const noexcept {
    return std::runtime_error::what();
}

} // namespace oneapi::dal::preview::spmd
