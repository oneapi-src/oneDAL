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

#pragma once

#include <daal/src/externals/service_rng.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>
#include <daal/src/algorithms/engines/engine_types_internal.h>

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Type = std::size_t, typename... Args>
inline void uniform_by_cpu(Args&&... args) {
    dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
        int res =
            daal::internal::
                RNGs<Type, oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value>{}
                    .uniform(std::forward<Args>(args)...);
        if (res) {
            using msg = dal::detail::error_messages;
            throw internal_error(msg::failed_to_generate_random_numbers());
        }
    });
}

template <typename Type = std::size_t, typename... Args>
inline void uniform_without_replacement_by_cpu(Args&&... args) {
    dispatch_by_cpu(context_cpu{}, [&](auto cpu) {
        int res =
            daal::internal::
                RNGs<Type, oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value>{}
                    .uniformWithoutReplacement(std::forward<Args>(args)...);
        if (res) {
            using msg = dal::detail::error_messages;
            throw internal_error(msg::failed_to_generate_random_numbers());
        }
    });
}

} // namespace oneapi::dal::backend::primitives
