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
#include "oneapi/dal/test/engine/config.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace de = dal::detail;

template <typename TestType>
class sort_with_indices_test : public te::policy_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Index = std::tuple_element_t<1, TestType>;

    auto allocate_arrays(Index elem_count) {
        auto& q = this->get_queue();
        auto val = ndarray<Float, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto ind = ndarray<Index, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);

        Index* ind_ptr = ind.get_mutable_data();
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::range<1>(elem_count), [=](sycl::item<1> item) {
                 Index ind = item.get_id()[0];
                 ind_ptr[ind] = ind;
             });
         }).wait_and_throw();

        return std::make_tuple(val, ind);
    }

    auto allocate_vector_of_arrays(std::int64_t vector_count, Index elem_count) {
        auto& q = this->get_queue();
        std::vector<ndarray<Float, 1>> val_vec(vector_count);
        std::vector<ndarray<Index, 1>> ind_vec(vector_count);
        for (std::int64_t i = 0; i < vector_count; i++) {
            val_vec[i] = ndarray<Float, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
            ind_vec[i] = ndarray<Index, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        }

        return std::make_tuple(val_vec, ind_vec);
    }

    auto init_vector_arrays(std::vector<ndarray<Float, 1>>& val_vec,
                            std::vector<ndarray<Index, 1>>& ind_vec,
                            Float a,
                            Float b,
                            std::int64_t seed = 777) {
        auto& q = this->get_queue();
        const std::int64_t vector_count = val_vec.size();
        for (std::int64_t i = 0; i < vector_count; i++) {
            Index* ind_ptr = ind_vec[i].get_mutable_data();
            auto event = q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(ind_vec[i].get_count()), [=](sycl::item<1> item) {
                    Index ind = item.get_id()[0];
                    ind_ptr[ind] = ind;
                });
            });

            fill_uniform(val_vec[i], a, b, seed);

            event.wait_and_throw();
        }

        return std::make_tuple(val_vec, ind_vec);
    }

    void fill_uniform(ndarray<Float, 1>& val, Float a, Float b, std::int64_t seed = 777) {
        Index elem_count = de::integral_cast<Index>(val.get_count());
        std::mt19937 rng(seed);
        std::uniform_real_distribution<Float> distr(a, b);

        // move generation to device when rng is available there
        Float* val_ptr = detail::host_allocator<Float>().allocate(val.get_count());
        for (Index el = 0; el < elem_count; el++) {
            val_ptr[el] = distr(rng);
        }
        val.assign(this->get_queue(), val_ptr, val.get_count()).wait_and_throw();
        detail::host_allocator<Float>().deallocate(val_ptr, val.get_count());
    }

    void run(ndarray<Float, 1>& val, ndarray<Index, 1>& ind) {
        INFO("benchmark sort with indices");
        const auto name =
            fmt::format("Basic sort with indices: val_type {}, indices_type {}, elem_count {}",
                        te::type2str<Float>::name(),
                        te::type2str<Index>::name(),
                        val.get_count());

        this->get_queue().wait_and_throw();
        auto sorter = radix_sort_indices_inplace<Float, Index>{ this->get_queue() };
        BENCHMARK(name.c_str()) {
            sorter(val, ind).wait_and_throw();
        };
    }

    void run(std::vector<ndarray<Float, 1>>& val_vec, std::vector<ndarray<Index, 1>>& ind_vec) {
        INFO("benchmark sort with indices");
        const std::int64_t vector_count = val_vec.size();
        ONEDAL_ASSERT(vector_count > 0);
        const auto name = fmt::format(
            "Basic sort with indices: val_type {}, indices_type {}, vector_count {}, elem_count {}",
            te::type2str<Float>::name,
            te::type2str<Index>::name,
            vector_count,
            val_vec[0].get_count());

        this->get_queue().wait_and_throw();
        auto sorter = radix_sort_indices_inplace<Float, Index>{ this->get_queue() };
        BENCHMARK(name.c_str()) {
            for (std::int64_t i = 0; i < vector_count; i++) {
                sorter(val_vec[i], ind_vec[i]).wait_and_throw();
            }
        };
    }
};

template <typename Integer>
class sort_test : public te::policy_fixture {
public:
    auto allocate_arrays(std::int64_t vector_count, std::int64_t elem_count) {
        auto& q = this->get_queue();
        auto val =
            ndarray<Integer, 2>::empty(q, { vector_count, elem_count }, sycl::usm::alloc::device);

        return val;
    }

    void fill_uniform(ndarray<Integer, 2>& val,
                      std::int64_t a,
                      std::int64_t b,
                      std::int64_t seed = 777) {
        std::uint32_t vector_count = de::integral_cast<std::uint32_t>(val.get_dimension(0));
        std::uint32_t elem_count = de::integral_cast<std::uint32_t>(val.get_dimension(1));

        std::mt19937 rng(seed);
        std::uniform_int_distribution<Integer> distr(a, b);

        // move generation to device when rng is available there
        Integer* val_ptr = detail::host_allocator<Integer>().allocate(val.get_count());
        for (std::uint32_t vec = 0; vec < vector_count; vec++) {
            for (std::uint32_t el = 0; el < elem_count; el++) {
                val_ptr[vec * elem_count + el] = distr(rng);
            }
        }

        val.assign(this->get_queue(), val_ptr, val.get_count()).wait_and_throw();
        detail::host_allocator<Integer>().deallocate(val_ptr, val.get_count());
    }

    void run(ndarray<Integer, 2>& val, std::int64_t sorted_elem_count) {
        auto& q = this->get_queue();
        std::uint32_t vector_count = de::integral_cast<std::uint32_t>(val.get_dimension(0));
        std::uint32_t elem_count = de::integral_cast<std::uint32_t>(val.get_dimension(1));

        const auto name =
            fmt::format("Basic sort: vector_count {} X elem_count {}, sorted_elem_count {}",
                        vector_count,
                        elem_count,
                        sorted_elem_count);

        q.wait_and_throw();

        INFO("allocate output buffer");
        auto val_out =
            ndarray<Integer, 2>::empty(q, { vector_count, elem_count }, sycl::usm::alloc::device);

        INFO("benchmark sort");
        auto sorter = radix_sort<Integer>{ this->get_queue() };
        BENCHMARK(name.c_str()) {
            sorter(val, val_out, sorted_elem_count).wait_and_throw();
        };
    }
};

using sort_indices_types = COMBINE_TYPES((float, double), (std::uint32_t));

#define SORT_WITH_INDICES_BENCH(name) \
    TEMPLATE_LIST_TEST_M(sort_with_indices_test, name, "[sort][perf]", sort_indices_types)

SORT_WITH_INDICES_BENCH("bench for sort with indices MNIST 784 x 60000") {
    SKIP_IF(this->get_policy().is_cpu());

    auto [val_vec_, ind_vec_] = this->allocate_vector_of_arrays(784, 60000);
    auto [val_vec, ind_vec] = this->init_vector_arrays(val_vec_, ind_vec_, -25., 25.);

    this->run(val_vec, ind_vec);
}

SORT_WITH_INDICES_BENCH("bench for sort with indices SUSY 18 x 4.5M") {
    SKIP_IF(this->get_policy().is_cpu());

    auto [val_vec_, ind_vec_] = this->allocate_vector_of_arrays(18, 4500000);
    auto [val_vec, ind_vec] = this->init_vector_arrays(val_vec_, ind_vec_, -25., 25.);

    this->run(val_vec, ind_vec);
}

SORT_WITH_INDICES_BENCH("bench for sort with indices HIGGS 28 x 1M") {
    SKIP_IF(this->get_policy().is_cpu());

    auto [val_vec_, ind_vec_] = this->allocate_vector_of_arrays(28, 1000000);
    auto [val_vec, ind_vec] = this->init_vector_arrays(val_vec_, ind_vec_, -25., 25.);

    this->run(val_vec, ind_vec);
}

SORT_WITH_INDICES_BENCH("bench for sort with indices HIGGS 28 x 10.5M") {
    SKIP_IF(this->get_policy().is_cpu());

    auto [val_vec_, ind_vec_] = this->allocate_vector_of_arrays(28, 10500000);
    auto [val_vec, ind_vec] = this->init_vector_arrays(val_vec_, ind_vec_, -25., 25.);

    this->run(val_vec, ind_vec);
}

SORT_WITH_INDICES_BENCH("bench for sort with indices YEAR 90 x 463715") {
    SKIP_IF(this->get_policy().is_cpu());

    auto [val_vec_, ind_vec_] = this->allocate_vector_of_arrays(90, 463715);
    auto [val_vec, ind_vec] = this->init_vector_arrays(val_vec_, ind_vec_, -25., 25.);

    this->run(val_vec, ind_vec);
}

SORT_WITH_INDICES_BENCH("bench for sort with indices HEPMASS 28 x 100K") {
    SKIP_IF(this->get_policy().is_cpu());

    auto [val_vec_, ind_vec_] = this->allocate_vector_of_arrays(28, 100000);
    auto [val_vec, ind_vec] = this->init_vector_arrays(val_vec_, ind_vec_, -25., 25.);

    this->run(val_vec, ind_vec);
}

TEMPLATE_TEST_M(sort_test, "benchmark for basic sort", "[sort][perf]", std::uint32_t) {
    SKIP_IF(this->get_policy().is_cpu());

    std::int64_t vector_count = GENERATE_COPY(128, 16384);
    std::int64_t elem_count = GENERATE_COPY(1024, 8192);

    std::int64_t sorted_elem_count = elem_count - (elem_count > 2 ? GENERATE_COPY(0, 12) : 0);

    auto val = this->allocate_arrays(vector_count, elem_count);
    this->fill_uniform(val, 0, 50);
    this->run(val, sorted_elem_count);
}

} // namespace oneapi::dal::backend::primitives::test
