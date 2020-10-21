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

#include "oneapi/dal/detail/error_reporting.hpp"

#define DEFINE_MESSAGE(id, text)                \
    static const char* _##id = text;            \
    const char* error_messages::id() noexcept { \
        return _##id;                           \
    }

namespace oneapi::dal::detail {

DEFINE_MESSAGE(input_data_is_empty, "Input data is empty")
DEFINE_MESSAGE(input_labels_are_empty, "Input labels are empty")
DEFINE_MESSAGE(input_data_row_count_is_not_equal_to_input_labels_row_count,
               "Input data row count is not equal to input labels row count")
DEFINE_MESSAGE(bootstrap_is_incompatible_with_variable_importance_mode,
               "Parameter `bootstrap` is incompatible with requested variable importance mode")

} // namespace oneapi::dal::detail
