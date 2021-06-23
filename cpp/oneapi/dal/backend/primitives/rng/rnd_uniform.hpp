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

#include <limits>

#include <daal/src/externals/service_rng.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename T>
class rnd_uniform {
public:
    rnd_uniform() : engine_(daal::algorithms::engines::mt19937::Batch<>::create(777)) {}
    void generate(sycl::queue& queue, ndview<T, 1>& result_array, T a, T b) {
        static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
        ONEDAL_ASSERT(*engine_ != nullptr);
        auto engine_impl =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&(*engine_));

        auto* values = result_array.get_mutable_data();
        auto count = result_array.get_count();
        const size_t count_as_size_t = dal::detail::integral_cast<std::size_t>(count);

        daal::internal::RNGs<size_t, daal::sse2> rng;
        rng.uniform(count_as_size_t, values, engine_impl->getState(), a, b);
    }
    daal::algorithms::engines::EnginePtr engine_;
};

#endif

} // namespace oneapi::dal::backend::primitives
