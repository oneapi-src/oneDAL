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
#include "oneapi/dal/backend/dispatcher.hpp"

#include <daal/src/externals/service_rng.h>
#include <daal/include/algorithms/engines/mt19937/mt19937.h>
#include <daal/include/algorithms/engines/mcg59/mcg59.h>
#include <daal/src/algorithms/engines/engine_batch_impl.h>
#include "oneapi/dal/backend/interop/common.hpp"

#include "oneapi/dal/array.hpp"

using namespace oneapi::dal::backend;

namespace oneapi::dal::preview::connected_components::backend {

// Generates random sequence with required parameters
template <typename Cpu, typename Type>
class rnd_seq {
public:
    rnd_seq() = delete;
    /// @param[in]  count Number of elements in random sequence
    /// @param[in]  a     Minimal value in the sequence
    /// @param[in]  b     Maximal value in the sequence
    rnd_seq(std::int64_t count, std::int64_t a = 0, std::int64_t b = 1000) {
        ONEDAL_ASSERT(count > 0);
        seq_ = array<Type>::empty(count);
        generate(a, b);
    }
    std::int64_t get_count() {
        return seq_.get_count();
    }
    const Type* get_data() {
        return seq_.get_data();
    }

private:
    void generate(Type a, Type b) {
        auto engine = daal::algorithms::engines::mt19937::Batch<>::create(777);
        auto engine_impl =
            dynamic_cast<daal::algorithms::engines::internal::BatchBaseImpl*>(&(*engine));
        ONEDAL_ASSERT(engine_impl != nullptr);

        auto* values = this->seq_.get_mutable_data();
        auto count = this->seq_.get_count();
        const auto count_as_size_t = dal::detail::integral_cast<std::size_t>(count);

        auto number_array = array<size_t>::empty(count);
        daal::internal::RNGs<size_t, oneapi::dal::backend::interop::to_daal_cpu_type<Cpu>::value>
            rng;
        auto* number_ptr = number_array.get_mutable_data();

        ONEDAL_ASSERT(a >= 0);
        ONEDAL_ASSERT(b >= 0);
        ONEDAL_ASSERT(b >= a);
        rng.uniform(count_as_size_t, number_ptr, engine_impl->getState(), 0, count_as_size_t);
        std::transform(number_ptr, number_ptr + count, values, [=](size_t number) {
            return a + (b - a) * static_cast<Type>(number) / static_cast<Type>(count);
        });
    }
    array<Type> seq_;
};

void generate_uniformly(size_t* result_array, std::int64_t count, size_t a, size_t b);

} // namespace oneapi::dal::preview::connected_components::backend