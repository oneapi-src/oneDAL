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
#include "oneapi/dal/backend/primitives/rng/rng_engine_collection.hpp"
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
        oneapi_rng<Index> rn_gen;
        return rn_gen;
    }

    auto get_engine(std::int64_t seed) {
        auto rng_engine = oneapi_engine<engine_qq>(this->get_queue(), seed);
        return rng_engine;
    }

    auto allocate_arrays(std::int64_t elem_count) {
        auto& q = this->get_queue();
        auto val_gpu = ndarray<Index, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto val_host = ndarray<Index, 1>::empty({ elem_count });

        return std::make_tuple(val_gpu, val_host);
    }

    auto allocate_arrays_shared(std::int64_t elem_count) {
        auto& q = this->get_queue();
        auto val_gpu = ndarray<Index, 1>::empty(q, { elem_count }, sycl::usm::alloc::shared);
        auto val_host = ndarray<Index, 1>::empty(q, { elem_count }, sycl::usm::alloc::shared);

        return std::make_tuple(val_gpu, val_host);
    }

    auto allocate_arrays_device(std::int64_t elem_count) {
        auto& q = this->get_queue();
        auto val_gpu_1 = ndarray<Index, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto val_gpu_2 = ndarray<Index, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);

        return std::make_tuple(val_gpu_1, val_gpu_2);
    }

    auto allocate_arrays_host(std::int64_t elem_count) {
        auto val_host_1 = ndarray<Index, 1>::empty({ elem_count });
        auto val_host_2 = ndarray<Index, 1>::empty({ elem_count });

        return std::make_tuple(val_host_1, val_host_2);
    }

    void check_results_host(const ndarray<Index, 1>& val_host_1,
                            const ndarray<Index, 1>& val_host_2) {
        const Index* val_host_1_ptr = val_host_1.get_data();

        const Index* val_host_2_ptr = val_host_2.get_data();

        for (std::int64_t el = 0; el < val_host_1.get_count(); el++) {
            REQUIRE(val_host_1_ptr[el] == val_host_2_ptr[el]);
        }
    }

    void check_results_device(const ndarray<Index, 1>& val_gpu_1,
                              const ndarray<Index, 1>& val_gpu_2) {
        const auto val_gpu_host_1 = val_gpu_1.to_host(this->get_queue());
        const Index* val_gpu_host_1_ptr = val_gpu_host_1.get_data();

        const auto val_gpu_host_2 = val_gpu_2.to_host(this->get_queue());
        const Index* val_gpu_host_2_ptr = val_gpu_host_2.get_data();

        for (std::int64_t el = 0; el < val_gpu_2.get_count(); el++) {
            REQUIRE(val_gpu_host_2_ptr[el] == val_gpu_host_1_ptr[el]);
        }
    }

    void check_results(const ndarray<Index, 1>& val_gpu, const ndarray<Index, 1>& val_host) {
        const Index* val_host_ptr = val_host.get_data();

        const auto val_gpu_host = val_gpu.to_host(this->get_queue());
        const Index* val_gpu_host_ptr = val_gpu_host.get_data();

        for (std::int64_t el = 0; el < val_host.get_count(); el++) {
            REQUIRE(val_gpu_host_ptr[el] == val_host_ptr[el]);
        }
    }
};

using rng_types = COMBINE_TYPES((float, double), (mt2203, mt19937, mcg59));

TEMPLATE_LIST_TEST_M(rng_test, "rng cpu vs gpu", "[rng]", rng_types) {
    SKIP_IF(this->get_policy().is_cpu());
    std::int64_t elem_count = GENERATE_COPY(10, 777, 10000, 50000);
    std::int64_t seed = GENERATE_COPY(777, 999);

    auto [arr_gpu, arr_host] = this->allocate_arrays(elem_count);
    auto arr_gpu_ptr = arr_gpu.get_mutable_data();
    auto arr_host_ptr = arr_host.get_mutable_data();

    auto rn_gen = this->get_rng();
    auto rng_engine = this->get_engine(seed);
    auto rng_engine_ = this->get_engine(seed);

    rn_gen.uniform_cpu(elem_count, arr_host_ptr, rng_engine, 0, elem_count);
    rn_gen.uniform_gpu(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine_, 0, elem_count);

    this->check_results(arr_gpu, arr_host);
}

using rng_types_skip = COMBINE_TYPES((float), (mcg59));

// TEMPLATE_LIST_TEST_M(rng_test, "rng cpu vs gpu", "[rng]", rng_types_skip) {
//     SKIP_IF(this->get_policy().is_cpu());
//     std::int64_t elem_count =
//         GENERATE_COPY(10, 1000, 300000, 15000, 1000000, 100000000, 6100000000, 1LL * 64 * 1000000);
//     std::int64_t seed = GENERATE_COPY(777);

//     auto [arr_gpu, arr_host] = this->allocate_arrays(elem_count);
//     auto arr_gpu_ptr = arr_gpu.get_mutable_data();
//     auto arr_host_ptr = arr_host.get_mutable_data();

//     auto rn_gen = this->get_rng();
//     auto rng_engine = this->get_engine(seed);
//     auto rng_engine_ = this->get_engine(seed);

//     BENCHMARK("Uniform dispatcher HOST arr" + std::to_string(elem_count)) {
//         rn_gen.uniform(this->get_queue(), elem_count, arr_host_ptr, rng_engine, 0, elem_count);
//     };
//     BENCHMARK("Uniform dispatcher GPU arr" + std::to_string(elem_count)) {
//         rn_gen.uniform(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine_, 0, elem_count);
//     };

//     auto [arr_gpu_, arr_host_] = this->allocate_arrays(elem_count);
//     auto arr_gpu_ptr_ = arr_gpu_.get_mutable_data();
//     auto arr_host_ptr_ = arr_host_.get_mutable_data();

//     auto rn_gen_ = this->get_rng();
//     auto rng_engine_1 = this->get_engine(seed);
//     auto rng_engine_2 = this->get_engine(seed);
//     BENCHMARK("Uniform GPU arr" + std::to_string(elem_count)) {
//         rn_gen_.uniform_gpu(this->get_queue(),
//                                      elem_count,
//                                      arr_gpu_ptr_,
//                                      rng_engine_1,
//                                      0,
//                                      elem_count);
//     };

//     BENCHMARK("Uniform HOST arr" + std::to_string(elem_count)) {
//         rn_gen_.uniform(elem_count, arr_host_ptr_, rng_engine_2, 0, elem_count);
//     };
// }

TEMPLATE_LIST_TEST_M(rng_test, "mixed rng cpu skip", "[rng]", rng_types_skip) {
    SKIP_IF(this->get_policy().is_cpu());
    std::int64_t elem_count = GENERATE_COPY(10, 777, 10000, 100000);
    std::int64_t seed = GENERATE_COPY(777, 999);

    auto [arr_host_init_1, arr_host_init_2] = this->allocate_arrays_host(elem_count);
    auto [arr_gpu, arr_host] = this->allocate_arrays(elem_count);
    auto arr_host_init_1_ptr = arr_host_init_1.get_mutable_data();
    auto arr_host_init_2_ptr = arr_host_init_2.get_mutable_data();
    auto arr_gpu_ptr = arr_gpu.get_mutable_data();
    auto arr_host_ptr = arr_host.get_mutable_data();

    auto rn_gen = this->get_rng();
    auto rng_engine = this->get_engine(seed);
    auto rng_engine_2 = this->get_engine(seed);

    rn_gen.uniform_cpu(elem_count, arr_host_init_1_ptr, rng_engine, 0, elem_count);
    rn_gen.uniform_cpu(elem_count, arr_host_init_2_ptr, rng_engine_2, 0, elem_count);

    rn_gen.uniform_gpu(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine, 0, elem_count);
    rn_gen.uniform_cpu(elem_count, arr_host_ptr, rng_engine_2, 0, elem_count);

    this->check_results_host(arr_host_init_1, arr_host_init_2);
    this->check_results(arr_gpu, arr_host);
}

TEMPLATE_LIST_TEST_M(rng_test, "mixed rng gpu skip", "[rng]", rng_types_skip) {
    SKIP_IF(this->get_policy().is_cpu());
    std::int64_t elem_count = GENERATE_COPY(10, 100, 777, 10000);
    std::int64_t seed = GENERATE_COPY(1, 777, 999);

    auto [arr_device_init_1, arr_device_init_2] = this->allocate_arrays_device(elem_count);
    auto [arr_gpu, arr_host] = this->allocate_arrays(elem_count);
    auto arr_device_init_1_ptr = arr_device_init_1.get_mutable_data();
    auto arr_device_init_2_ptr = arr_device_init_2.get_mutable_data();
    auto arr_gpu_ptr = arr_gpu.get_mutable_data();
    auto arr_host_ptr = arr_host.get_mutable_data();

    auto rn_gen = this->get_rng();
    auto rng_engine = this->get_engine(seed);
    auto rng_engine_2 = this->get_engine(seed);

    rn_gen.uniform_gpu(this->get_queue(),
                       elem_count,
                       arr_device_init_1_ptr,
                       rng_engine,
                       0,
                       elem_count);
    rn_gen.uniform_gpu(this->get_queue(),
                       elem_count,
                       arr_device_init_2_ptr,
                       rng_engine_2,
                       0,
                       elem_count);

    rn_gen.uniform_gpu(this->get_queue(), elem_count, arr_gpu_ptr, rng_engine, 0, elem_count);
    rn_gen.uniform_cpu(elem_count, arr_host_ptr, rng_engine_2, 0, elem_count);

    this->check_results_device(arr_device_init_1, arr_device_init_2);
    this->check_results(arr_gpu, arr_host);
}

// TEMPLATE_LIST_TEST_M(rng_test, "mixed rng gpu skip collection", "[rng]", rng_types_skip) {
//     SKIP_IF(this->get_policy().is_cpu());
//     std::int64_t elem_count = GENERATE_COPY(10, 100, 777, 10000);
//     std::int64_t seed = GENERATE_COPY(1, 777, 999);

//     engine_collection<std::int64_t,engine_list::mcg59> collection(this->get_queue(), 2, seed);

//     auto engine_arr = collection.get_engines();

//     auto [arr_device_init_1, arr_device_init_2] = this->allocate_arrays_shared(elem_count);

//     auto arr_device_init_1_ptr = arr_device_init_1.get_mutable_data();
//     auto arr_device_init_2_ptr = arr_device_init_2.get_mutable_data();

//     auto rn_gen = this->get_rng();

//     rn_gen.uniform(this->get_queue(),
//                    elem_count,
//                    arr_device_init_1_ptr,
//                    engine_arr[0],
//                    0,
//                    elem_count);

//     rn_gen.uniform(this->get_queue(),
//                    elem_count,
//                    arr_device_init_2_ptr,
//                    engine_arr[1],
//                    0,
//                    elem_count);

//     // rn_gen.uniform(this->get_queue(), elem_count, arr_gpu_ptr, engine_arr[0], 0, elem_count);
//     // rn_gen.uniform(elem_count, arr_host_ptr, engine_arr[1], 0, elem_count);

//     //this->check_results_device(arr_device_init_1, arr_device_init_2);
//     this->check_results(arr_device_init_1, arr_device_init_2);
// }

} // namespace oneapi::dal::backend::primitives::test
