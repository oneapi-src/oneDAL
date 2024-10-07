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

#include <daal/include/algorithms/engines/mt2203/mt2203.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>
#include "oneapi/dal/backend/primitives/rng/utils.hpp"

namespace oneapi::dal::backend::primitives {

namespace engine {
namespace v1 {

/// Tag-type that denotes the mt2203 engine.
struct mt2203 {};

/// Tag-type that denotes the mcg59 engine.
struct mcg59 {};

/// Tag-type that denotes the mt19937 engine.
struct mt19937 {};

/// Alias tag-type for the default engine (mt2203).
using by_default = mt2203;

} // namespace v1
} // namespace engine

template <engine_list EngineType = engine::v1::by_default>
class daal_engine {
public:
    explicit daal_engine(std::int64_t seed = 777)
            : daal_engine_(initialize_daal_engine(seed)),
              impl_(dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(
                  daal_engine_.get())) {
        if (!impl_) {
            throw std::domain_error("RNG engine is not supported");
        }
    }

    virtual ~daal_engine() = default;

    void* get_cpu_engine_state() const {
        return impl_->getState();
    }

    auto& get_cpu_engine() {
        return daal_engine_;
    }
private:
    daal::algorithms::engines::EnginePtr initialize_daal_engine(std::int64_t seed) {
        switch (EngineType) {
            case engine_list::mt2203:
                return daal::algorithms::engines::mt2203::Batch<>::create(seed);
            case engine_list::mcg59: return daal::algorithms::engines::mcg59::Batch<>::create(seed);
            case engine_list::mt19937:
                return daal::algorithms::engines::mt19937::Batch<>::create(seed);
            default: throw std::invalid_argument("Unsupported engine type");
        }
    }

    daal::algorithms::engines::EnginePtr daal_engine_;
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

template <typename Type, typename Size = std::int64_t>
class daal_rng {
public:
    daal_rng() = default;
    ~daal_rng() = default;

    template <engine_list EngineType>
    void uniform(Size count, Type* dst, daal_engine<EngineType>& engine_, Type a, Type b) {
        void* state = engine_.get_cpu_engine_state();
        uniform_dispatcher::uniform_by_cpu<Type>(count, dst, state, a, b);
    }

    template <engine_list EngineType>
    void uniform_without_replacement_cpu(Size count,
                                     Type* dst,
                                     Type* buffer,
                                     daal_engine<EngineType>& engine_,
                                     Type a,
                                     Type b) {
        void* state = engine_.get_cpu_engine_state();
        uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count,
                                                                     dst,
                                                                     buffer,
                                                                     state,
                                                                     a,
                                                                     b);
    }

    template <engine_list EngineType, typename T = Type, typename = std::enable_if_t<std::is_integral_v<T>>>
    void shuffle(Size count, Type* dst, daal_engine<EngineType>& engine_) {
        Type idx[2];

        void* state = engine_.get_cpu_engine_state();

        for (Size i = 0; i < count; ++i) {
            uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
            std::swap(dst[idx[0]], dst[idx[1]]);
        }
    }
};

} // namespace oneapi::dal::backend::primitives