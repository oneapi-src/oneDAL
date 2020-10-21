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
    error_messages(const error_messages&) = delete;
    error_messages& operator=(const error_messages&) = delete;

    DECLARE_MESSAGE(input_data_is_empty);
    DECLARE_MESSAGE(input_labels_are_empty);
    DECLARE_MESSAGE(input_data_row_count_is_not_equal_to_input_labels_row_count);
    DECLARE_MESSAGE(bootstrap_is_incompatible_with_variable_importance_mode);
};

#undef DECLARE_MESSAGE

#define DEFINE_EXCEPTION(ExceptionType, id)              \
    static ExceptionType id() noexcept {                 \
        return ExceptionType(error_messages::id());      \
    }                                                    \
                                                         \
    static void report_##id##_if_false(bool condition) { \
        if (!condition) {                                \
            throw id();                                  \
        }                                                \
    }

class exception_generator {
public:
    exception_generator() = delete;
    exception_generator(const exception_generator&) = delete;
    exception_generator& operator=(const exception_generator&) = delete;

    DEFINE_EXCEPTION(domain_error, input_data_is_empty)
    DEFINE_EXCEPTION(domain_error, input_labels_are_empty)
    DEFINE_EXCEPTION(invalid_argument, input_data_row_count_is_not_equal_to_input_labels_row_count)
    DEFINE_EXCEPTION(invalid_argument, bootstrap_is_incompatible_with_variable_importance_mode)
};

#undef DEFINE_EXCEPTION

} // namespace oneapi::dal::detail
