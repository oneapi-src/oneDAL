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

#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::detail {

#define DECLARE_MESSAGE(id) static const char* id() noexcept

class error_messages {
public:
    error_messages() = delete;
};

#undef DECLARE_MESSAGE

} // namespace oneapi::dal::detail
