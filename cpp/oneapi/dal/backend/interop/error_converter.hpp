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

#include <daal/include/services/error_handling.h>
#include <daal/include/services/internal/status_to_error_id.h>

namespace oneapi::dal::backend::interop {

void status_to_exception(const daal::services::Status& s);

template <class StatusConverter>
inline void status_to_exception(const daal::services::Status& s, StatusConverter alg_converter) {
    if (s) {
        return;
    }
    alg_converter(s);
    status_to_exception(s);
}

} // namespace oneapi::dal::backend::interop
