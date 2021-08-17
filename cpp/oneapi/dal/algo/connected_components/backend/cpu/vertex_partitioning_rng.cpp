/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include "oneapi/dal/algo/connected_components/backend/cpu/vertex_partitioning_rng.hpp"

#include <daal/src/externals/service_rng.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>

#include "oneapi/dal/array.hpp"

using namespace oneapi::dal::backend;

namespace oneapi::dal::preview::connected_components::backend {

struct uniform_dispatcher {
    template <typename... Args>
    static void uniform_by_cpu(Args&&... args) {
        dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
            daal::internal::RNGs<size_t, daal::sse2> rng;
            auto res = rng.uniform(std::forward<Args>(args)...);
            //    oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)::value >
            if (res) {
                using msg = dal::detail::error_messages;
                throw internal_error(msg::failed_to_generate_random_numbers());
            }
        });
    }
};

void generate_uniformly(size_t* result_array, std::int64_t count, size_t a, size_t b) {
    using msg = dal::detail::error_messages;
    daal::algorithms::engines::EnginePtr engine =
        daal::algorithms::engines::mt19937::Batch<>::create(777777);
    if (engine.get() == nullptr) {
        throw internal_error(msg::failed_to_generate_random_numbers());
    }
    auto engine_impl =
        dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&(*engine));
    if (engine_impl == nullptr) {
        throw internal_error(msg::failed_to_generate_random_numbers());
    }
    uniform_dispatcher::uniform_by_cpu(count, result_array, engine_impl->getState(), a, b);
}

} // namespace oneapi::dal::preview::connected_components::backend