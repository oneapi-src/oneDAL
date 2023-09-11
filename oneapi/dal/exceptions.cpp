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

#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal {

const char* invalid_argument::what() const noexcept {
    return std::invalid_argument::what();
}

const char* uninitialized_optional_result::what() const noexcept {
    return std::logic_error::what();
}

const char* domain_error::what() const noexcept {
    return std::domain_error::what();
}

const char* out_of_range::what() const noexcept {
    return std::out_of_range::what();
}

const char* unimplemented::what() const noexcept {
    return std::logic_error::what();
}

const char* unsupported_device::what() const noexcept {
    return std::logic_error::what();
}

const char* range_error::what() const noexcept {
    return std::range_error::what();
}

const char* system_error::what() const noexcept {
    return std::system_error::what();
}

const std::error_code& system_error::code() const noexcept {
    return std::system_error::code();
}

const char* internal_error::what() const noexcept {
    return std::runtime_error::what();
}

const char* host_bad_alloc::what() const noexcept {
    return std::bad_alloc::what();
}

const char* device_bad_alloc::what() const noexcept {
    return std::bad_alloc::what();
}

} // namespace oneapi::dal
