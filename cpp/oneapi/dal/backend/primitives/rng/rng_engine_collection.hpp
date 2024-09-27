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

#include "oneapi/dal/backend/primitives/rng/rng.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include <vector>

#include <daal/include/algorithms/engines/mt2203/mt2203.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>
#include "oneapi/dal/backend/primitives/rng/utils.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Size = std::int64_t, engine_list EngineType = engine_list::mt2203>
class engine_collection {
public:
    engine_collection(sycl::queue& queue, Size count, std::int64_t seed = 777)
            : count_(count),
              seed_(seed) {
        engines_.reserve(count_);
        for (Size i = 0; i < count_; ++i) {
            engines_.push_back(engine<EngineType>(queue, seed_));
        }
    }

    std::vector<engine<EngineType>> get_engines() const {
        return engines_;
    }

private:
    Size count_;
    std::int64_t seed_;
    std::vector<engine<EngineType>> engines_;
};

#endif
} // namespace oneapi::dal::backend::primitives
