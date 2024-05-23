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

#include "oneapi/dal/backend/interop/common_dpc.hpp"

#include <unordered_map>
#include <daal/include/services/env_detect.h>
#include <daal/src/services/service_defines.h>
#include <daal/include/services/internal/execution_context.h>
#include <daal/include/services/internal/utilities.h>

namespace oneapi::dal::backend::interop {

using daal_sycl_ex_ctx_t = daal::services::internal::SyclExecutionContext;

class execution_context_cache {
public:
    static execution_context_cache& get_instance() {
        static execution_context_cache cache;
        return cache;
    }

    ~execution_context_cache() {
        // We do not delete map entries intentionally as deallocation order of global object is not
        // defined. The desctructors of objects stored by DAAL `SyclExecutionContext` likely require
        // access to dynamic libraries, which may be unloaded before the call to this desctructor.
        // The workaround leads to memory leak, however does not affect user application as always
        // happens at the end. This will be no longer required once kernels are migrated to DPC++.
    }

    daal_sycl_ex_ctx_t lookup(const sycl::queue& queue) {
        const std::size_t hash = std::hash<sycl::queue>{}(queue);
        const auto it = map_.find(hash);
        if (it == map_.end()) {
            const auto ctx = new daal_sycl_ex_ctx_t{ queue };
            map_.emplace(hash, ctx);
            return *ctx;
        }
        return *it->second;
    }

    void cleanup() {
        map_.clear();
    }

private:
    execution_context_cache() = default;
    std::unordered_map<std::size_t, daal_sycl_ex_ctx_t*> map_;
};

execution_context_guard::execution_context_guard(const sycl::queue& queue) {
    auto ctx = execution_context_cache::get_instance().lookup(queue);
    daal::services::Environment::getInstance().setDefaultExecutionContext(ctx);
}

execution_context_guard::~execution_context_guard() {
    daal::services::Environment::getInstance().setDefaultExecutionContext(
        daal::services::internal::CpuExecutionContext());
}

} // namespace oneapi::dal::backend::interop
