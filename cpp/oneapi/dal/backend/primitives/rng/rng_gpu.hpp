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

#include <daal/include/algorithms/engines/mt2203/mt2203.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/include/algorithms/engines/mrg32k3a/mrg32k3a.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>
#include "oneapi/dal/backend/primitives/rng/utils.hpp"
#include <oneapi/mkl.hpp>
namespace mkl = oneapi::mkl;
namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

enum class engine_list { mt2203, mcg59, mt19937, mrg32k3a};

template <engine_list EngineType>
struct oneapi_engine_type;

template <>
struct oneapi_engine_type<engine_list::mt2203> {
    using type = oneapi::mkl::rng::mt2203;
};

template <>
struct oneapi_engine_type<engine_list::mcg59> {
    using type = oneapi::mkl::rng::mcg59;
};

template <>
struct oneapi_engine_type<engine_list::mt19937> {
    using type = oneapi::mkl::rng::mt19937;
};

template <>
struct oneapi_engine_type<engine_list::mrg32k3a> {
    using type = oneapi::mkl::rng::mrg32k3a;
};

template <engine_list EngineType = engine_list::mt2203>
class oneapi_engine {
public:
    using onedal_engine_t = typename oneapi_engine_type<EngineType>::type;

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
        if constexpr (EngineType == engine_list::mt2203) {
        }
        else {
            skip_ahead(onedal_engine_, nSkip);
        }
    }

private:
    daal::algorithms::engines::EnginePtr initialize_daal_engine(std::int64_t seed) {
        switch (EngineType) {
            case engine_list::mt2203:
                return daal::algorithms::engines::mt2203::Batch<>::create(seed);
            case engine_list::mcg59: return daal::algorithms::engines::mcg59::Batch<>::create(seed);
            case engine_list::mrg32k3a: return daal::algorithms::engines::mrg32k3a::Batch<>::create(seed);
            case engine_list::mt19937:
                return daal::algorithms::engines::mt19937::Batch<>::create(seed);
            default: throw std::invalid_argument("Unsupported engine type");
        }
    }

    onedal_engine_t initialize_oneapi_engine(sycl::queue& queue, std::int64_t seed) {
        if constexpr (EngineType == engine_list::mt2203) {
            return onedal_engine_t(queue, seed,
                                   0); // Aligns CPU and GPU results for mt2203
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

    template <engine_list EngineType>
    void uniform(sycl::queue& queue,
                 Size count,
                 Type* dst,
                 oneapi_engine<EngineType>& engine_,
                 Type a,
                 Type b,
                 bool distr_mode = false,
                 const event_vector& deps = {});

    template <engine_list EngineType>
    void uniform_gpu(sycl::queue& queue,
                     Size count,
                     Type* dst,
                     oneapi_engine<EngineType>& engine_,
                     Type a,
                     Type b,
                     const event_vector& deps = {});

    template <engine_list EngineType>
    void uniform_cpu(Size count, Type* dst, oneapi_engine<EngineType>& engine_, Type a, Type b);
    template <engine_list EngineType>
    void uniform_without_replacement(sycl::queue& queue,
                                     Size count,
                                     Type* dst,
                                     oneapi_engine<EngineType>& engine_,
                                     Type a,
                                     Type b,
                                     const event_vector& deps = {}) {}

    template <engine_list EngineType>
    void uniform_without_replacement_gpu(sycl::queue& queue,
                                         Size count,
                                         Type* dst,
                                         Type* buff,
                                         oneapi_engine<EngineType>& engine_,
                                         Type a,
                                         Type b,
                                         const event_vector& deps = {});

    template <engine_list EngineType>
    void uniform_without_replacement_cpu(Size count,
                                         Type* dst,
                                         Type* buffer,
                                         oneapi_engine<EngineType>& engine_,
                                         Type a,
                                         Type b) {
        void* state = engine_.get_cpu_engine_state();
        engine_.skip_ahead_gpu(count);
        uniform_dispatcher::uniform_without_replacement_by_cpu<Type>(count,
                                                                     dst,
                                                                     buffer,
                                                                     state,
                                                                     a,
                                                                     b);
    }

    template <engine_list EngineType,
              typename T = Type,
              typename = std::enable_if_t<std::is_integral_v<T>>>
    void shuffle(Size count, Type* dst, oneapi_engine<EngineType>& engine_) {
        Type idx[2];

        void* state = engine_.get_cpu_engine_state();
        engine_.skip_ahead_gpu(count);

        for (Size i = 0; i < count; ++i) {
            uniform_dispatcher::uniform_by_cpu<Type>(2, idx, state, 0, count);
            std::swap(dst[idx[0]], dst[idx[1]]);
        }
    }

    template <engine_list EngineType>
    void shuffle_gpu(sycl::queue& queue,
                     Size count,
                     Type* dst,
                     oneapi_engine<EngineType>& engine_,
                     const event_vector& deps);

    template <engine_list EngineType,
              typename T = Type,
              typename = std::enable_if_t<std::is_integral_v<T>>>
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
