/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/backend/primitives/rng/rng.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

class mt2203 {};
class mcg59 {};
class mt19937 {};

template <typename engine_type>
struct engine_map {};

template <>
struct engine_map<mt2203> {
    constexpr static auto value = engine_list::mt2203;
};

template <>
struct engine_map<mcg59> {
    constexpr static auto value = engine_list::mcg59;
};

template <>
struct engine_map<mt19937> {
    constexpr static auto value = engine_list::mt19937;
};

template <typename engine_type>
constexpr auto engine_v = engine_map<engine_type>::value;

template <typename TestType>
class rng_test : public te::policy_fixture {
public:
    using Index = std::tuple_element_t<0, TestType>;
    using EngineType = std::tuple_element_t<1, TestType>;
    static constexpr auto engine_qq = engine_v<EngineType>;

    auto get_rng() const {
        rng<Index> rn_gen;
        return rn_gen;
    }

    auto get_engine(std::int64_t seed) {
        auto rng_engine = engine<engine_qq>(this->get_queue(), seed);
        return rng_engine;
    }

    auto allocate_arrays(std::int64_t elem_count) {
        auto& q = this->get_queue();
        auto val_gpu = ndarray<Index, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto val_host = ndarray<Index, 1>::empty({ elem_count });

        return std::make_tuple(val_gpu, val_host);
    }

    void check_results(const ndarray<Index, 1>& val_gpu, const ndarray<Index, 1>& val_host) {
        const Index* val_host_ptr = val_host.get_data();

        const auto val_gpu_host = val_gpu.to_host(this->get_queue());
        const Index* val_gpu_host_ptr = val_gpu_host.get_data();

        for (std::int64_t el = 0; el < val_host.get_count(); el++) {
            //necessary for debug
            //std::cout<<"index = "<<el<<"gpu="<<val_gpu_host_ptr[el]<<"cpu="<<val_host_ptr[el]<<std::endl;
            REQUIRE(val_gpu_host_ptr[el] == val_host_ptr[el]);
        }
    }
};

using rng_types = COMBINE_TYPES((float, double), (mt2203, mcg59, mt19937));

TEMPLATE_LIST_TEST_M(rng_test, "rng cpu vs gpu", "[rng]", rng_types) {
    SKIP_IF(this->get_policy().is_cpu());
    std::int64_t elem_count = GENERATE_COPY(10, 777, 10000);
    std::int64_t seed = GENERATE_COPY(1, 777, 999);

    auto [arr_gpu, arr_host] = this->allocate_arrays(elem_count);
    auto arr_gpu_ptr = arr_gpu.get_mutable_data();
    auto arr_host_ptr = arr_host.get_mutable_data();

    auto rn_gen = this->get_rng();
    auto rng_engine = this->get_engine(seed);

    rn_gen.uniform(elem_count, arr_host_ptr, rng_engine.get_state(), 0, elem_count);
    rn_gen.uniform(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine, 0, elem_count);

    this->check_results(arr_gpu, arr_host);
}

} // namespace oneapi::dal::backend::primitives::test
