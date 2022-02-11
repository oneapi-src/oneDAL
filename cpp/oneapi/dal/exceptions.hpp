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

#include <new>
#include <stdexcept>
#include <system_error>

#include "oneapi/dal/common.hpp"

namespace oneapi::dal {
namespace v1 {

class ONEDAL_EXPORT exception {
public:
    virtual ~exception() = default;
    virtual const char* what() const noexcept = 0;
};

class ONEDAL_EXPORT logic_error : public exception {};
class ONEDAL_EXPORT runtime_error : public exception {};

class ONEDAL_EXPORT invalid_argument : public logic_error, public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
    const char* what() const noexcept override;
};

class ONEDAL_EXPORT uninitialized_optional_result : public logic_error, public std::logic_error {
public:
    using std::logic_error::logic_error;
    const char* what() const noexcept override;
};

class ONEDAL_EXPORT domain_error : public logic_error, public std::domain_error {
public:
    using std::domain_error::domain_error;
    const char* what() const noexcept override;
};

class ONEDAL_EXPORT out_of_range : public logic_error, public std::out_of_range {
public:
    using std::out_of_range::out_of_range;
    const char* what() const noexcept override;
};

class ONEDAL_EXPORT unimplemented : public logic_error, public std::logic_error {
public:
    using std::logic_error::logic_error;
    const char* what() const noexcept override;
};

class ONEDAL_EXPORT unsupported_device : public logic_error, public std::logic_error {
public:
    using std::logic_error::logic_error;
    const char* what() const noexcept override;
};

class ONEDAL_EXPORT range_error : public runtime_error, public std::range_error {
public:
    using std::range_error::range_error;
    const char* what() const noexcept override;
};

class ONEDAL_EXPORT system_error : public runtime_error, public std::system_error {
public:
    using std::system_error::system_error;
    const char* what() const noexcept override;
    const std::error_code& code() const noexcept;
};

class ONEDAL_EXPORT internal_error : public runtime_error, public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    const char* what() const noexcept override;
};

class ONEDAL_EXPORT bad_alloc : public exception, public std::bad_alloc {};

class ONEDAL_EXPORT host_bad_alloc : public bad_alloc {
public:
    host_bad_alloc() noexcept = default;
    const char* what() const noexcept override;
};

class ONEDAL_EXPORT device_bad_alloc : public bad_alloc {
public:
    device_bad_alloc() noexcept = default;
    const char* what() const noexcept override;
};

} // namespace v1

using v1::exception;
using v1::logic_error;
using v1::runtime_error;
using v1::invalid_argument;
using v1::uninitialized_optional_result;
using v1::domain_error;
using v1::out_of_range;
using v1::unimplemented;
using v1::unsupported_device;
using v1::range_error;
using v1::system_error;
using v1::internal_error;
using v1::bad_alloc;
using v1::host_bad_alloc;
using v1::device_bad_alloc;

} // namespace oneapi::dal
