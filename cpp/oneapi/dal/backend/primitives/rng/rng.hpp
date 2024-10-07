/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include <oneapi/mkl.hpp>
#include <daal/include/algorithms/engines/mt2203/mt2203.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>
#include "oneapi/dal/backend/primitives/rng/utils.hpp"

namespace mkl = oneapi::mkl;
namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

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

// Helper to map engine types to corresponding oneAPI MKL engine types
template <typename EngineType>
struct select_onedal_engine;

template <>
struct select_onedal_engine<engine::v1::mt2203> {
    using type = oneapi::mkl::rng::mt2203;
};

template <>
struct select_onedal_engine<engine::v1::mcg59> {
    using type = oneapi::mkl::rng::mcg59;
};

template <>
struct select_onedal_engine<engine::v1::mt19937> {
    using type = oneapi::mkl::rng::mt19937;
};

template <typename EngineType = engine::v1::by_default>
class oneapi_engine {
public:
    using onedal_engine_t = typename select_onedal_engine<EngineType>::type;

    explicit oneapi_engine(sycl::queue& queue, std::int64_t seed = 777)
            : q(queue),
              daal_engine_(initialize_daal_engine(seed)),
              onedal_engine_(initialize_oneapi_engine(queue, seed)),
              impl_(dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(
                  daal_engine_.get())) {
        if (!impl_) {
            throw std::domain_error("RNG engine is not supported");
        }
    }

    virtual ~oneapi_engine() = default;

    void* get_cpu_engine_state() const {
        return impl_->getState();
    }

    auto& get_cpu_engine() {
        return daal_engine_;
    }

    auto& get_gpu_engine() {
        return onedal_engine_;
    }

    void skip_ahead_cpu(size_t nSkip) {
        daal_engine_->skipAhead(nSkip);
    }

    void skip_ahead_gpu(size_t nSkip) {
        if constexpr (std::is_same_v<EngineType, engine::v1::mt2203>) {
            // GPU-specific code for mt2203
        }
        else {
            skip_ahead(onedal_engine_, nSkip);
        }
    }

private:
    daal::algorithms::engines::EnginePtr initialize_daal_engine(std::int64_t seed) {
        if constexpr (std::is_same_v<EngineType, engine::v1::mt2203>) {
            return daal::algorithms::engines::mt2203::Batch<>::create(seed);
        }
        else if constexpr (std::is_same_v<EngineType, engine::v1::mcg59>) {
            return daal::algorithms::engines::mcg59::Batch<>::create(seed);
        }
        else if constexpr (std::is_same_v<EngineType, engine::v1::mt19937>) {
            return daal::algorithms::engines::mt19937::Batch<>::create(seed);
        }
        else {
            throw std::invalid_argument("Unsupported engine type. Supported types: mt2203, mcg59, mt19937");
        }
    }

    onedal_engine_t initialize_oneapi_engine(sycl::queue& queue, std::int64_t seed) {
        if constexpr (std::is_same_v<EngineType, engine::v1::mt2203>) {
            return onedal_engine_t(queue, seed, 0);  // Aligns CPU and GPU results for mt2203
        }
        else {
            return onedal_engine_t(queue, seed);
        }
    }

    sycl::queue q;
    daal::algorithms::engines::EnginePtr daal_engine_;
    onedal_engine_t onedal_engine_;
    daal::algorithms::engines::internal::BatchBaseImpl* impl_;
};

template <typename Type, typename Size = std::int64_t>
class oneapi_rng {
public:
    oneapi_rng() = default;
    ~oneapi_rng() = default;

    template <typename EngineType>
    void uniform(sycl::queue& queue,
                 Size count,
                 Type* dst,
                 oneapi_engine<EngineType>& engine_,
                 Type a,
                 Type b,
                 bool distr_mode = false,
                 const event_vector& deps = {});

    template <typename EngineType>
    void uniform_gpu(sycl::queue& queue,
                     Size count,
                     Type* dst,
                     oneapi_engine<EngineType>& engine_,
                     Type a,
                     Type b,
                     const event_vector& deps = {});

    template <typename EngineType>
    void uniform_cpu(Size count, Type* dst, oneapi_engine<EngineType>& engine_, Type a, Type b) {
        void* state = engine_.get_cpu_engine_state();
        engine_.skip_ahead_cpu(count);
        uniform_dispatcher::uniform_by_cpu<Type>(count, dst, state, a, b);
    }

    template <typename EngineType>
    void uniform_without_replacement(sycl::queue& queue,
                                     Size count,
                                     Type* dst,
                                     oneapi_engine<EngineType>& engine_,
                                     Type a,
                                     Type b,
                                     const event_vector& deps = {}) {
    }

    template <typename EngineType>
    void uniform_without_replacement_gpu(sycl::queue& queue,
                                         Size count,
                                         Type* dst,
                                         oneapi_engine<EngineType>& engine_,
                                         Type a,
                                         Type b,
                                         const event_vector& deps = {}) {
    }

    template <typename EngineType>
    void uniform_without_replacement_cpu(Size count,
                                         Type* dst,
                                         Type* buffer,
                                         oneapi_engine<EngineType>& engine_,
                                         Type a,
                                         Type b) {
        void* state = engine_.get_cpu_engine_state();
        engine_.skip_ahead_gpu(count);
        uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count, dst, buffer, state, a, b);
    }

    template <typename EngineType, typename T = Type, typename = std::enable_if_t<std::is_integral_v<T>>>
    void shuffle(Size count, Type* dst, oneapi_engine<EngineType>& engine_) {
        Type idx[2];

        void* state = engine_.get_cpu_engine_state();
        engine_.skip_ahead_gpu(count);

        for (Size i = 0; i < count; ++i) {
            uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
            std::swap(dst[idx[0]], dst[idx[1]]);
        }
    }

    template <typename EngineType, typename T = Type, typename = std::enable_if_t<std::is_integral_v<T>>>
    void shuffle_gpu(Size count, Type* dst, oneapi_engine<EngineType>& engine_) {
        Type idx[2];

        void* state = engine_.get_cpu_engine_state();
        engine_.skip_ahead_gpu(count);

        for (Size i = 0; i < count; ++i) {
            uniform_dispatcher::uniform_by_gpu<Type>(2, idx, engine_.get_gpu_engine(), 0, count);
            std::swap(dst[idx[0]], dst[idx[1]]);
        }
    }

    template <typename EngineType, typename T = Type, typename = std::enable_if_t<std::is_integral_v<T>>>
    void shuffle_cpu(Size count, Type* dst, oneapi_engine<EngineType>& engine_) {
        Type idx[2];

        void* state = engine_.get_cpu_engine_state();
        engine_.skip_ahead_gpu(count);

        for (Size i = 0; i < count; ++i) {
            uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
            std::swap(dst[idx[0]], dst[idx[1]]);
        }
    }
};

#endif
} // namespace oneapi::dal::backend::primitives
