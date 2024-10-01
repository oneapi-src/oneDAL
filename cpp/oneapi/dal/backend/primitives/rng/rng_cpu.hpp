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

#include "oneapi/dal/backend/primitives/rng/utils.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Type, typename Size = std::int64_t>
class rng {
public:
    rng() = default;
    ~rng() = default;

    void uniform(Size count, Type* dst, void* state, Type a, Type b) {
        uniform_dispatcher::uniform_by_cpu<Type>(count, dst, state, a, b);
    }

    void uniform_without_replacement(Size count,
                                     Type* dst,
                                     Type* buffer,
                                     void* state,
                                     Type a,
                                     Type b) {
        uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count,
                                                                     dst,
                                                                     buffer,
                                                                     state,
                                                                     a,
                                                                     b);
    }

    template <typename T = Type, typename = std::enable_if_t<std::is_integral_v<T>>>
    void shuffle(Size count, Type* dst, void* state) {
        Type idx[2];

        for (Size i = 0; i < count; ++i) {
            uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
            std::swap(dst[idx[0]], dst[idx[1]]);
        }
    }

private:
    daal::internal::RNGsInst<Type, DAAL_BASE_CPU> daal_rng_;
};

class engine {
public:
    explicit engine(std::int64_t seed = 777)
            : engine_(daal::algorithms::engines::mt2203::Batch<>::create(seed)) {
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(engine_.get());
        if (!impl_) {
            throw domain_error(dal::detail::error_messages::rng_engine_is_not_supported());
        }
    }

    explicit engine(const daal::algorithms::engines::EnginePtr& eng) : engine_(eng) {
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(eng.get());
        if (!impl_) {
            throw domain_error(dal::detail::error_messages::rng_engine_is_not_supported());
        }
    }

    virtual ~engine() = default;

    engine& operator=(const daal::algorithms::engines::EnginePtr& eng) {
        engine_ = eng;
        impl_ = dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(eng.get());
        if (!impl_) {
            throw domain_error(dal::detail::error_messages::rng_engine_is_not_supported());
        }

        return *this;
    }

    void* get_state() const {
        return impl_->getState();
    }

private:
    daal::algorithms::engines::EnginePtr engine_;
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

} // namespace oneapi::dal::backend::primitives