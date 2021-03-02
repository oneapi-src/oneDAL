/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <unordered_map>
#include "oneapi/dal/backend/interop/common_dpc.hpp"

namespace oneapi::dal::backend::interop {

using daal_sycl_ex_ctx_t = daal::services::internal::SyclExecutionContext;

class execution_context_cache {
public:
    static execution_context_cache& get_instance() {
        static execution_context_cache cache;
        return cache;
    }

    daal_sycl_ex_ctx_t lookup(const sycl::queue& queue) {
        if (!is_enabled()) {
            return daal_sycl_ex_ctx_t{ queue };
        }

        const auto it = map_.find(queue);
        if (it == map_.end()) {
            const auto ctx = daal_sycl_ex_ctx_t{ queue };
            map_.emplace(queue, ctx);
            return ctx;
        }
        return it->second;
    }

    void cleanup() {
        map_.clear();
    }

    void enable() {
        is_enabled_ = true;
    }

    bool is_enabled() const {
        return is_enabled_;
    }

private:
    execution_context_cache() = default;

    bool is_enabled_ = false;
    std::unordered_map<sycl::queue, daal_sycl_ex_ctx_t> map_;
};

execution_context_guard::execution_context_guard(const sycl::queue& queue) {
    auto ctx = execution_context_cache::get_instance().lookup(queue);
    daal::services::Environment::getInstance()->setDefaultExecutionContext(ctx);
}

execution_context_guard::~execution_context_guard() {
    daal::services::Environment::getInstance()->setDefaultExecutionContext(
        daal::services::internal::CpuExecutionContext());
}

void enable_daal_sycl_execution_context_cache() {
    execution_context_cache::get_instance().enable();
}

void cleanup_daal_sycl_execution_context_cache() {
    execution_context_cache::get_instance().cleanup();
}

} // namespace oneapi::dal::backend::interop
