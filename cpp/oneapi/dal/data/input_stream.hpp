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

class ONEAPI_DAL_EXPORT input_stream {
public:
    input_stream(const char * file_name) {
        const size_t len = strlen(file_name);
        file_name_ = new char[len + 1];
        detail::memcpy(detail::default_host_policy{}, file_name_, file_name, sizeof(char) * len);
        file_name_[len] = '\0';
    }

    input_stream(const input_stream& other) : input_stream(other.get_file_name()) {}

    ~input_stream() {
        delete [] file_name_;
    }

    const char * get_file_name() const {
        return file_name_;
    }

private:
    char * file_name_;
};

} // namespace oneapi::dal
