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

#include <daal/include/services/env_detect.h>
#include <daal/src/services/service_defines.h>

namespace oneapi::dal::backend::interop {

#ifdef ONEAPI_DAL_DATA_PARALLEL

struct execution_context_guard {
    explicit execution_context_guard(const sycl::queue &queue) {
        daal::services::internal::SyclExecutionContext ctx(queue);
        daal::services::Environment::getInstance()->setDefaultExecutionContext(ctx);
    }

    ~execution_context_guard() {
        daal::services::Environment::getInstance()->setDefaultExecutionContext(
            daal::services::internal::CpuExecutionContext());
    }
};

#endif

} // namespace oneapi::dal::backend::interop
