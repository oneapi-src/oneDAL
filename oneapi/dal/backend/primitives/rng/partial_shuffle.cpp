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

#include "oneapi/dal/backend/primitives/rng/partial_shuffle.hpp"
#include "oneapi/dal/backend/primitives/rng/utils.hpp"

#include <daal/include/algorithms/engines/mt19937/mt19937.h>

#include <tuple>

namespace oneapi::dal::backend::primitives {

void partial_fisher_yates_shuffle(ndview<std::int64_t, 1>& result_array,
                                  std::int64_t top,
                                  std::int64_t seed) {
    using msg = dal::detail::error_messages;
    daal::algorithms::engines::EnginePtr engine =
        daal::algorithms::engines::mt19937::Batch<>::create(seed);
    if (engine.get() == nullptr) {
        throw internal_error(msg::failed_to_generate_random_numbers());
    }
    auto engine_impl =
        dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&(*engine));
    if (engine_impl == nullptr) {
        throw internal_error(msg::failed_to_generate_random_numbers());
    }
    const auto casted_top = dal::detail::integral_cast<std::size_t>(top);
    const std::int64_t count = result_array.get_count();
    const auto casted_count = dal::detail::integral_cast<std::size_t>(count);
    ONEDAL_ASSERT(casted_count < casted_top);
    auto indices_ptr = result_array.get_mutable_data();

    std::int64_t k = 0;
    std::size_t value = 0;
    for (std::size_t i = 0; i < casted_count; i++) {
        uniform_dispatcher::uniform_by_cpu(1, &value, engine_impl->getState(), i, casted_top);
        for (std::size_t j = i; j > 0; j--) {
            if (value == dal::detail::integral_cast<std::size_t>(indices_ptr[j - 1])) {
                value = j - 1;
            }
        }
        if (value >= casted_top)
            continue;
        indices_ptr[i] = dal::detail::integral_cast<std::int64_t>(value);
        k++;
    }
    ONEDAL_ASSERT(k == count);
}

void partial_fisher_yates_shuffle(ndview<std::int64_t, 1>& result_array, std::int64_t top) {
    partial_fisher_yates_shuffle(result_array, top, 7777);
}

} // namespace oneapi::dal::backend::primitives
