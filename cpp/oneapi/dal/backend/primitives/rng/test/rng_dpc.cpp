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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/io.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/rng/rng.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_engine_collection.hpp"
namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace de = dal::detail;

template <typename TestType>
class rng_test : public te::policy_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Index = std::tuple_element_t<1, TestType>;
    using rng_engine_t = engine;
    using rng_engine_list_t = std::vector<rng_engine_t>;

    auto allocate_arrays(Index elem_count) {
        auto& q = this->get_queue();
        auto val_gpu = ndarray<Index, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto val_host = ndarray<Index, 1>::empty({ elem_count });

        return std::make_tuple(val_gpu, val_host);
    }

    void check_results(const ndarray<Float, 1>& val_gpu, const ndarray<Float, 1>& val_host) {
        const Float* val_host_ptr = val_host.get_data();

        const auto val_gpu_host = val_gpu.to_host(this->get_queue());
        const Float* val_gpu_host_ptr = val_gpu_host.get_data();

        std::cout << std::endl << "Val_host:" << std::endl;
        for (Index el = 0; el < val_host.get_count(); el++) {
            std::cout << val_host_ptr[el] << " ";
        }
        std::cout << std::endl << "Val_gpu:" << std::endl;
        for (Index el = 0; el < val_host.get_count(); el++) {
            std::cout << val_gpu_host_ptr[el] << " ";
        }
    }
};

using rng_types = COMBINE_TYPES((int), (int));

// TEMPLATE_LIST_TEST_M(rng_test, "rng with states", "[rng]", rng_types) {
//     SKIP_IF(this->get_policy().is_cpu());

//     std::int64_t elem_count = GENERATE_COPY(2, 10);
//     std::int64_t batch_count = GENERATE_COPY(2, 4);
//     std::int64_t seed = GENERATE_COPY(777, 999);
//     engine_collection collection(batch_count, seed);
//     std::vector<std::uint8_t*> states(batch_count);

//     std::vector<engine> engine_arr = collection([&](std::size_t i, std::size_t& skip) {
//         skip = i * 1;
//         oneapi::mkl::rng::mrg32k3a engine(this->get_queue(), skip);
//         auto mem_size = oneapi::mkl::rng::get_state_size(engine);
//         std::uint8_t* mem_buf = new std::uint8_t[mem_size];
//         oneapi::mkl::rng::save_state(engine, mem_buf);
//         states[i] = mem_buf;
//     });
//     auto [arr_gpu, arr_host] = this->allocate_arrays(elem_count);
//     auto arr_gpu_ptr = arr_gpu.get_mutable_data();
//     rng<int> rn_gen;
//     for (int node_idx = 0; node_idx < batch_count; ++node_idx) {
//         rn_gen.uniform(this->get_queue(),
//                        std::int64_t(elem_count / batch_count),
//                        arr_gpu_ptr,
//                        states[node_idx],
//                        0,
//                        elem_count);
//     }
//     auto arr_host_ptr = arr_host.get_mutable_data();

//     for (int node_idx = 0; node_idx < batch_count; ++node_idx) {
//         rn_gen.uniform(elem_count / batch_count,
//                        arr_host_ptr,
//                        engine_arr[node_idx].get_state(),
//                        0,
//                        elem_count);

//     }
//     this->check_results(arr_gpu, arr_host);
// }

// TEMPLATE_LIST_TEST_M(rng_test, "rng without states", "[rng]", rng_types) {
//     SKIP_IF(this->get_policy().is_cpu());

//     std::int64_t elem_count = GENERATE_COPY(2, 10);
//     std::int64_t seed = GENERATE_COPY(777, 999);
//     engine_collection collection(1, seed);

//     std::int64_t real_seed = 0;
//     std::vector<engine> engine_arr = collection([&](std::size_t i, std::size_t& skip) {
//         skip = i * 1;
//         real_seed = skip;
//     });
//     auto [arr_gpu, arr_host] = this->allocate_arrays(elem_count);
//     auto arr_gpu_ptr = arr_gpu.get_mutable_data();
//     rng<int> rn_gen;

//     rn_gen.uniform_mt2203(this->get_queue(), elem_count, arr_gpu_ptr, real_seed, 0, elem_count);

//     auto arr_host_ptr = arr_host.get_mutable_data();

//     rn_gen.uniform(elem_count, arr_host_ptr, engine_arr[0].get_state(), 0, elem_count);

//     this->check_results(arr_gpu, arr_host);
// }

// TEMPLATE_LIST_TEST_M(rng_test, "rng without states", "[rng]", rng_types) {
//     SKIP_IF(this->get_policy().is_cpu());

//     std::int64_t elem_count = GENERATE_COPY(2, 10);
//     std::int64_t seed = GENERATE_COPY(777, 999);
//     engine_collection collection(1, seed);

//     std::int64_t real_seed = 0;
//     std::vector<engine> engine_arr = collection([&](std::size_t i, std::size_t& skip) {
//         skip = i * 1;
//         real_seed = skip;
//     });
//     auto [arr_gpu, arr_host] = this->allocate_arrays(elem_count);
//     auto arr_gpu_ptr = arr_gpu.get_mutable_data();
//     rng<int> rn_gen;

//     rn_gen.uniform_mt2203(this->get_queue(), elem_count, arr_gpu_ptr, real_seed, 0, elem_count);

//     auto arr_host_ptr = arr_host.get_mutable_data();

//     rn_gen.uniform(elem_count, arr_host_ptr, engine_arr[0].get_state(), 0, elem_count);

//     this->check_results(arr_gpu, arr_host);
// }

TEMPLATE_LIST_TEST_M(rng_test, "rng without states", "[rng]", rng_types) {
    SKIP_IF(this->get_policy().is_cpu());

    std::int64_t elem_count = GENERATE_COPY(2, 10);
    std::int64_t batch_count = GENERATE_COPY(1);
    std::int64_t seed = GENERATE_COPY(777, 999);
    engine_collection collection(batch_count, seed);
    std::vector<std::uint8_t*> states(batch_count);

    std::vector<engine> engine_arr = collection([&](std::size_t i, std::size_t& skip) {
        skip = i * 1;
        oneapi::mkl::rng::mrg32k3a engine(this->get_queue(), skip);
        auto mem_size = oneapi::mkl::rng::get_state_size(engine);
        std::uint8_t* mem_buf = new std::uint8_t[mem_size];
        oneapi::mkl::rng::save_state(engine, mem_buf);
        states[i] = mem_buf;
    });
    auto [arr_gpu, arr_host] = this->allocate_arrays(elem_count);
    auto arr_gpu_ptr = arr_gpu.get_mutable_data();
    rng<int> rn_gen;
    for (int node_idx = 0; node_idx < batch_count; ++node_idx) {
        rn_gen.uniform_without_replacement(this->get_queue(),
                                           std::int64_t(elem_count / batch_count),
                                           arr_gpu_ptr,
                                           states[node_idx],
                                           0,
                                           elem_count * 10);
    }
    auto arr_host_ptr = arr_host.get_mutable_data();

    for (int node_idx = 0; node_idx < batch_count; ++node_idx) {
        rn_gen.uniform_without_replacement(elem_count / batch_count,
                                           arr_host_ptr,
                                           arr_host_ptr + 1,
                                           engine_arr[node_idx].get_state(),
                                           0,
                                           elem_count * 10);
    }
    this->check_results(arr_gpu, arr_host);
}
} // namespace oneapi::dal::backend::primitives::test
