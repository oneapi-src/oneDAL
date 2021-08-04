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
#include <daal/include/algorithms/engines/mt2203/mt2203.h>

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Type, typename... Args>
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

template <typename Type, typename... Args>
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

template <typename Type, typename Size = std::int64_t>
class rng {
public:
    rng() = default;
    ~rng() = default;

    void uniform(Size count, Type* dst, void* state, Type a, Type b) {
        uniform_by_cpu<Type>(count, dst, state, a, b);
    }

    void uniform_without_replacement(Size count,
                                     Type* dst,
                                     Type* buffer,
                                     void* state,
                                     Type a,
                                     Type b) {
        uniform_without_replacement_by_cpu<Type>(count, dst, buffer, state, a, b);
    }

    template <typename T = Type, typename = std::enable_if_t<std::is_integral_v<T>>>
    cl::sycl::event shuffle(Size count, Type* dst, void* state) {
        Type idx[2];

        for (Size i = 0; i < count; ++i) {
            uniform_by_cpu<Type>(2, idx, state, 0, count);
            std::swap(dst[idx[0]], dst[idx[1]]);
        }

        return cl::sycl::event{};
    }

private:
    sycl::queue queue_;
    daal::internal::RNGs<Type, daal::sse2> daal_rng_;
};

class engine {
public:
    engine() : engine_(daal::algorithms::engines::mt2203::Batch<>::create()) {
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

template <typename Size = std::int64_t>
class engine_collection {
public:
    engine_collection(Size count)
            : count_(count),
              engine_(daal::algorithms::engines::mt2203::Batch<>::create()),
              params_(count),
              technique_(daal::algorithms::engines::internal::family),
              engine_list_(count) {}

    template <typename Op>
    std::vector<engine> operator()(Op&& op) {
        daal::services::Status status;
        for (Size i = 0; i < count_; ++i) {
            op(i, params_.nSkip[i]);
        }
        select_parallelization_technique(technique_);
        daal::algorithms::engines::internal::EnginesCollection<daal::sse2> engine_collection(
            engine_,
            technique_,
            params_,
            engine_list_,
            &status);
        if (!status) {
            dal::backend::interop::status_to_exception(status);
        }

        std::vector<engine> arr_data(count_);
        for (Size i = 0; i < count_; ++i) {
            arr_data[i] = engine_list_[i];
        }

        //copy elision
        return arr_data;
    }

private:
    void select_parallelization_technique(
        daal::algorithms::engines::internal::ParallelizationTechnique& technique) {
        auto daal_engine_impl =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(engine_.get());

        daal::algorithms::engines::internal::ParallelizationTechnique techniques[] = {
            daal::algorithms::engines::internal::family,
            daal::algorithms::engines::internal::leapfrog,
            daal::algorithms::engines::internal::skipahead
        };

        for (auto& techn : techniques) {
            if (daal_engine_impl->hasSupport(techn)) {
                technique = techn;
                return;
            }
        }

        throw domain_error(
            dal::detail::error_messages::rng_engine_does_not_support_parallelization_techniques());
    }

private:
    Size count_;
    daal::algorithms::engines::EnginePtr engine_;
    daal::algorithms::engines::internal::Params<daal::sse2> params_;
    daal::algorithms::engines::internal::ParallelizationTechnique technique_;
    daal::services::internal::TArray<daal::algorithms::engines::EnginePtr, daal::sse2> engine_list_;
};

} // namespace oneapi::dal::backend::primitives
