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
#include "oneapi/dal/test/engine/spmd.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/thread_communicator.hpp"

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
    auto train(Args&&... args) {
        return oneapi::dal::test::engine::train(get_policy(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto infer(Args&&... args) {
        return oneapi::dal::test::engine::infer(get_policy(), std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto compute(Args&&... args) {
        return oneapi::dal::test::engine::compute(get_policy(), std::forward<Args>(args)...);
    }
};

template <typename Float>
class float_algo_fixture : public algo_fixture {
public:
    using float_t = Float;

    bool not_float64_friendly() {
        constexpr bool is_double = std::is_same_v<Float, double>;
        return is_double && !this->get_policy().has_native_float64();
    }

    table_id get_homogen_table_id() const {
        return table_id::homogen<Float>();
    }
};

template <typename TestType, typename Derived>
class crtp_algo_fixture : public float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using base_t = float_algo_fixture<std::tuple_element_t<0, TestType>>;
    using float_t = std::tuple_element_t<0, TestType>;
    using derived_t = Derived;

    template <typename... Args>
    auto train(Args&&... args) {
        return derived().train_override(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto infer(Args&&... args) {
        return derived().infer_override(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto compute(Args&&... args) {
        return derived().compute_override(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto split_train_input(Args&&... args) {
        return derived().split_train_input_override(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto merge_train_result(Args&&... args) {
        return derived().merge_train_result_override(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto split_compute_input(Args&&... args) {
        return derived().split_compute_input_override(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto merge_compute_result(Args&&... args) {
        return derived().merge_compute_result_override(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto train_override(Args&&... args) {
        return base_t::train(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto infer_override(Args&&... args) {
        return base_t::infer(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto compute_override(Args&&... args) {
        return base_t::compute(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto split_train_input_override(Args&&... args) {
        ONEDAL_ASSERT(!"This method must be overriden in the derived class");
    }

    template <typename... Args>
    auto merge_train_result_override(Args&&... args) {
        ONEDAL_ASSERT(!"This method must be overriden in the derived class");
    }

    template <typename... Args>
    auto split_compute_input_override(Args&&... args) {
        ONEDAL_ASSERT(!"This method must be overriden in the derived class");
    }

    template <typename... Args>
    auto merge_compute_result_override(Args&&... args) {
        ONEDAL_ASSERT(!"This method must be overriden in the derived class");
    }

    template <typename Descriptor, typename... Args>
    auto train_via_spmd_threads(std::int64_t thread_count, const Descriptor& desc, Args&&... args) {
        ONEDAL_ASSERT(thread_count > 0);

        CAPTURE(thread_count);

#ifdef ONEDAL_DATA_PARALLEL
        using comm_t = thread_communicator<spmd::device_memory_access::usm>;
        comm_t comm{ this->get_queue(), thread_count };
#else
        using comm_t = thread_communicator<spmd::device_memory_access::none>;
        comm_t comm{ thread_count };
#endif
        const auto input_per_rank =
            this->split_train_input(thread_count, std::forward<Args>(args)...);
        ONEDAL_ASSERT(input_per_rank.size() == std::size_t(thread_count));

        const auto results = comm.map([&](std::int64_t rank) {
            return dal::test::engine::spmd_train(this->get_policy(),
                                                 comm,
                                                 desc,
                                                 input_per_rank[rank]);
        });
        ONEDAL_ASSERT(results.size() == std::size_t(thread_count));

        return results;
    }

    template <typename Descriptor, typename... Args>
    auto train_via_spmd_threads_and_merge(std::int64_t thread_count,
                                          const Descriptor& desc,
                                          Args&&... args) {
        const auto results = this->train_via_spmd_threads( //
            thread_count,
            desc,
            std::forward<Args>(args)...);

        return this->merge_train_result(results);
    }

    template <typename Descriptor, typename... Args>
    auto compute_via_spmd_threads(std::int64_t thread_count,
                                  const Descriptor& desc,
                                  Args&&... args) {
        ONEDAL_ASSERT(thread_count > 0);

        CAPTURE(thread_count);
#ifdef ONEDAL_DATA_PARALLEL
        using comm_t = thread_communicator<spmd::device_memory_access::usm>;
        comm_t comm{ this->get_queue(), thread_count };
#else
        using comm_t = thread_communicator<spmd::device_memory_access::none>;
        comm_t comm{ thread_count };
#endif

        const auto input_per_rank =
            this->split_compute_input(thread_count, std::forward<Args>(args)...);
        ONEDAL_ASSERT(input_per_rank.size() == std::size_t(thread_count));

        const auto results = comm.map([&](std::int64_t rank) {
            return dal::test::engine::spmd_compute(this->get_policy(),
                                                   comm,
                                                   desc,
                                                   input_per_rank[rank]);
        });
        ONEDAL_ASSERT(results.size() == std::size_t(thread_count));

        return results;
    }

    template <typename Descriptor, typename... Args>
    auto compute_via_spmd_threads_and_merge(std::int64_t thread_count,
                                            const Descriptor& desc,
                                            Args&&... args) {
        const auto results = this->compute_via_spmd_threads( //
            thread_count,
            desc,
            std::forward<Args>(args)...);

        return this->merge_compute_result(results);
    }

private:
    Derived& derived() {
        return *(static_cast<Derived*>(this));
    }
};

} // namespace oneapi::dal::test::engine
