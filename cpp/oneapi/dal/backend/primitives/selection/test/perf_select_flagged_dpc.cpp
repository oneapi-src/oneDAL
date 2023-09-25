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
#include "oneapi/dal/backend/primitives/selection/select_flagged.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace de = dal::detail;

template <typename TestType>
class select_flagged_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Flag = std::tuple_element_t<1, TestType>;
    using Integer = std::uint32_t;

    auto allocate_arrays(Integer elem_count) {
        auto& q = this->get_queue();
        auto val = ndarray<Float, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto mask = ndarray<Flag, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);

        return std::make_tuple(val, mask);
    }

    auto allocate_vector_of_arrays(std::int64_t vector_count, Integer elem_count) {
        auto& q = this->get_queue();
        std::vector<ndarray<Float, 1>> val_vec(vector_count);
        std::vector<ndarray<Flag, 1>> mask_vec(vector_count);
        for (std::int64_t i = 0; i < vector_count; i++) {
            val_vec[i] = ndarray<Float, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
            mask_vec[i] = ndarray<Flag, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        }

        return std::make_tuple(val_vec, mask_vec);
    }

    auto init_vector_of_arrays(std::vector<ndarray<Float, 1>>& val_vec,
                               std::vector<ndarray<Flag, 1>>& mask_vec,
                               Float a,
                               Float b,
                               std::int64_t seed = 777) {
        const std::int64_t vector_count = val_vec.size();
        for (std::int64_t i = 0; i < vector_count; i++) {
            fill_uniform(val_vec[i], mask_vec[i], a, b, seed);
        }

        return std::make_tuple(val_vec, mask_vec);
    }

    void fill_uniform(ndarray<Float, 1>& val,
                      ndarray<Flag, 1>& mask,
                      Float a,
                      Float b,
                      std::int64_t seed = 777) {
        Integer elem_count = de::integral_cast<Integer>(val.get_count());
        std::mt19937 rng(seed);
        std::uniform_real_distribution<Float> distr(a, b);

        Float pivot = distr(rng);
        // move generation to device when rng is available there
        auto val_host = ndarray<Float, 1>::empty({ elem_count });
        auto mask_host = ndarray<Flag, 1>::empty({ elem_count });
        Float* val_ptr = val_host.get_mutable_data();
        Flag* mask_ptr = mask_host.get_mutable_data();

        for (Integer el = 0; el < elem_count; el++) {
            val_ptr[el] = distr(rng);
            mask_ptr[el] = val_ptr[el] < pivot ? 1 : 0;
        }

        auto& q = this->get_queue();

        val.assign(q, val_host).wait_and_throw();
        mask.assign(q, mask_host).wait_and_throw();
    }

    void run(ndarray<Float, 1>& in, ndarray<Flag, 1>& mask) {
        auto& q = this->get_queue();
        INFO("benchmark select flagged");
        std::int64_t elem_count = in.get_count();
        const auto name =
            fmt::format("Select flagged: val_type {}, flag_type {}, integer {}, elem_count {}",
                        te::type2str<Float>::name(),
                        te::type2str<Flag>::name(),
                        te::type2str<Integer>::name(),
                        elem_count);

        auto out = ndarray<Float, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        std::int64_t total_sum_res;

        this->get_queue().wait_and_throw();
        auto selector = select_flagged<Float, Flag>{ this->get_queue() };
        BENCHMARK(name.c_str()) {
            selector(mask, in, out, total_sum_res).wait_and_throw();
        };
    }

    void run(std::vector<ndarray<Float, 1>>& in_vec, std::vector<ndarray<Flag, 1>>& mask_vec) {
        auto& q = this->get_queue();
        INFO("benchmark select flagged");
        const std::int64_t vector_count = in_vec.size();
        ONEDAL_ASSERT(vector_count > 0);
        std::int64_t elem_count = in_vec[0].get_count();
        const auto name = fmt::format(
            "Select flagged: val_type {}, flag_type {}, integer {}, vector_count {}, elem_count {}",
            te::type2str<Float>::name(),
            te::type2str<Flag>::name(),
            te::type2str<Integer>::name(),
            vector_count,
            elem_count);

        auto out = ndarray<Float, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        std::int64_t total_sum_res;

        this->get_queue().wait_and_throw();
        auto selector = select_flagged<Float, Flag>{ this->get_queue() };
        BENCHMARK(name.c_str()) {
            for (std::int64_t i = 0; i < vector_count; i++) {
                selector(mask_vec[i], in_vec[i], out, total_sum_res).wait_and_throw();
            }
        };
    }
};

template <typename TestType>
class select_flagged_index_test : public te::float_algo_fixture<TestType> {
public:
    using Data = std::tuple_element_t<0, TestType>;
    using Flag = std::tuple_element_t<1, TestType>;
    using Integer = std::uint32_t;

    auto allocate_arrays(Integer elem_count) {
        auto& q = this->get_queue();
        auto val = ndarray<Data, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto mask = ndarray<Flag, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);

        return std::make_tuple(val, mask);
    }

    auto allocate_vector_of_arrays(std::int64_t vector_count, Integer elem_count) {
        auto& q = this->get_queue();
        std::vector<ndarray<Data, 1>> val_vec(vector_count);
        std::vector<ndarray<Flag, 1>> mask_vec(vector_count);
        for (std::int64_t i = 0; i < vector_count; i++) {
            val_vec[i] = ndarray<Data, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
            mask_vec[i] = ndarray<Flag, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        }

        return std::make_tuple(val_vec, mask_vec);
    }

    auto init_vector_of_arrays(std::vector<ndarray<Data, 1>>& val_vec,
                               std::vector<ndarray<Flag, 1>>& mask_vec,
                               std::int64_t seed = 777) {
        const std::int64_t vector_count = val_vec.size();
        for (std::int64_t i = 0; i < vector_count; i++) {
            fill_uniform(val_vec[i], mask_vec[i], seed);
        }

        return std::make_tuple(val_vec, mask_vec);
    }

    void fill_uniform(ndarray<Data, 1>& val, ndarray<Flag, 1>& mask, std::int64_t seed = 777) {
        Integer elem_count = de::integral_cast<Integer>(val.get_count());
        std::mt19937 rng(seed);
        std::uniform_int_distribution<Data> distr(0, val.get_count() - 1);

        // move generation to device when rng is available there
        auto val_host = ndarray<Data, 1>::empty({ elem_count });
        auto mask_host = ndarray<Flag, 1>::empty({ elem_count });
        Data* val_ptr = val_host.get_mutable_data();
        Flag* mask_ptr = mask_host.get_mutable_data();

        for (Integer el = 0; el < elem_count; el++) {
            val_ptr[el] = el;
            mask_ptr[el] = 0;
        }

        for (Integer el = 0; el < elem_count; el++) {
            Integer ind = distr(rng);
            mask_ptr[ind] = 1;
            Integer swap_ind = distr(rng);
            std::swap(val_ptr[el], val_ptr[swap_ind]);
        }

        auto& q = this->get_queue();

        val.assign(q, val_host).wait_and_throw();
        mask.assign(q, mask_host).wait_and_throw();
    }

    void run(ndarray<Data, 1>& in, ndarray<Flag, 1>& mask) {
        auto& q = this->get_queue();
        INFO("benchmark select flagged");
        std::int64_t elem_count = in.get_count();
        const auto name = fmt::format(
            "Select flagged index: val_type {}, flag_type {}, integer {}, elem_count {}",
            te::type2str<Data>::name,
            te::type2str<Flag>::name,
            te::type2str<Integer>::name,
            elem_count);

        auto out = ndarray<Data, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        std::int64_t total_sum_res;

        this->get_queue().wait_and_throw();
        auto selector = select_flagged_index<Data, Flag>{ this->get_queue() };
        BENCHMARK(name.c_str()) {
            selector(mask, in, out, total_sum_res).wait_and_throw();
        };
    }

    void run(std::vector<ndarray<Data, 1>>& in_vec, std::vector<ndarray<Flag, 1>>& mask_vec) {
        auto& q = this->get_queue();
        INFO("benchmark select flagged");
        const std::int64_t vector_count = in_vec.size();
        ONEDAL_ASSERT(vector_count > 0);
        std::int64_t elem_count = in_vec[0].get_count();
        const auto name = fmt::format(
            "Select flagged: val_type {}, flag_type {}, integer {}, vector_count {}, elem_count {}",
            te::type2str<Data>::name,
            te::type2str<Flag>::name,
            te::type2str<Integer>::name,
            vector_count,
            elem_count);

        auto out = ndarray<Data, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        std::int64_t total_sum_res;

        this->get_queue().wait_and_throw();
        auto selector = select_flagged_index<Data, Flag>{ this->get_queue() };
        BENCHMARK(name.c_str()) {
            for (std::int64_t i = 0; i < vector_count; i++) {
                selector(mask_vec[i], in_vec[i], out, total_sum_res).wait_and_throw();
            }
        };
    }
};

using select_flagged_types = COMBINE_TYPES((float, double), (std::uint8_t, std::uint32_t));
using select_flagged_index_types = COMBINE_TYPES((std::int32_t, std::uint32_t),
                                                 (std::uint8_t, std::uint32_t));

#define SELECT_FLAGGED_BENCH(name) \
    TEMPLATE_LIST_TEST_M(select_flagged_test, name, "[select flagged][perf]", select_flagged_types)

#define SELECT_FLAGGED_INDEX_BENCH(name)            \
    TEMPLATE_LIST_TEST_M(select_flagged_index_test, \
                         name,                      \
                         "[select flagged][perf]",  \
                         select_flagged_index_types)

SELECT_FLAGGED_BENCH("bench for select flagged") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    auto [val_vec_, mask_vec_] = this->allocate_vector_of_arrays(512, 32000);
    auto [val_vec, mask_vec] = this->init_vector_of_arrays(val_vec_, mask_vec_, -25., 25.);

    this->run(val_vec, mask_vec);
}

SELECT_FLAGGED_INDEX_BENCH("bench for select flagged index") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    auto [val_vec_, mask_vec_] = this->allocate_vector_of_arrays(512, 32000);
    auto [val_vec, mask_vec] = this->init_vector_of_arrays(val_vec_, mask_vec_);

    this->run(val_vec, mask_vec);
}

} // namespace oneapi::dal::backend::primitives::test
