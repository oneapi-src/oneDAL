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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

namespace oneapi::dal::test::engine {

class policy_fixture {
public:
    auto& get_policy() {
        return policy_;
    }

#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue& get_queue() {
        return policy_.get_queue();
    }
#endif

private:
    DECLARE_TEST_POLICY(policy_);
};

class algo_fixture : public policy_fixture {
public:
    template <typename... Args>
    auto base_train(Args&&... args) {
        return oneapi::dal::test::engine::train(get_policy(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto base_infer(Args&&... args) {
        return oneapi::dal::test::engine::infer(get_policy(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto base_compute(Args&&... args) {
        return oneapi::dal::test::engine::compute(get_policy(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto spmd_train(const dal::detail::spmd_communicator& comm, Args&&... args) {
        return oneapi::dal::test::engine::train(get_policy(), comm, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto spmd_infer(const dal::detail::spmd_communicator& comm, Args&&... args) {
        return oneapi::dal::test::engine::infer(get_policy(), comm, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto spmd_compute(const dal::detail::spmd_communicator& comm, Args&&... args) {
        return oneapi::dal::test::engine::compute(get_policy(), comm, std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto train(Args&&... args) {
        return base_train(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto infer(Args&&... args) {
        return base_infer(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto compute(Args&&... args) {
        return base_compute(std::forward<Args>(args)...);
    }
};

template <typename Float>
class float_algo_fixture : public algo_fixture {
public:
    using float_t = Float;

    constexpr bool is_float64() const {
        return std::is_same_v<float_t, double>;
    }

    bool not_float64_friendly() {
        return is_float64() && !this->get_policy().has_native_float64();
    }

    table_id get_homogen_table_id() const {
        return table_id::homogen<Float>();
    }
};

} // namespace oneapi::dal::test::engine
