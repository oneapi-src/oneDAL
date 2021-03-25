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

template <typename T>
struct type2str {
    static const char* name;
};

#define INSTANTIATE_TYPE_MAP(T) \
    template <>                 \
    const char* type2str<T>::name = #T;

INSTANTIATE_TYPE_MAP(float);
INSTANTIATE_TYPE_MAP(double);
INSTANTIATE_TYPE_MAP(std::uint8_t);
INSTANTIATE_TYPE_MAP(std::uint16_t);
INSTANTIATE_TYPE_MAP(std::uint32_t);
INSTANTIATE_TYPE_MAP(std::uint64_t);
INSTANTIATE_TYPE_MAP(std::int8_t);
INSTANTIATE_TYPE_MAP(std::int16_t);
INSTANTIATE_TYPE_MAP(std::int32_t);
INSTANTIATE_TYPE_MAP(std::int64_t);

template <typename TestType>
class sort_with_indices_test : public te::policy_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using IndexType = std::tuple_element_t<1, TestType>;

    auto allocate_arrays(IndexType elem_count) {
        auto& q = this->get_queue();
        auto val = ndarray<Float, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto ind = ndarray<IndexType, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);

        IndexType* ind_ptr = ind.get_mutable_data();
        q.submit([&](sycl::handler& cgh) {
             cgh.parallel_for(sycl::range<1>(elem_count), [=](sycl::item<1> item) {
                 IndexType ind = item.get_id()[0];
                 ind_ptr[ind] = ind;
             });
         }).wait_and_throw();

        return std::make_tuple(val, ind);
    }

    auto allocate_vector_arrays(std::int64_t vector_count, IndexType elem_count) {
        auto& q = this->get_queue();
        std::vector<ndarray<Float, 1>> val_vec(vector_count);
        std::vector<ndarray<IndexType, 1>> ind_vec(vector_count);
        for (std::int64_t i = 0; i < vector_count; i++) {
            val_vec[i] = ndarray<Float, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
            ind_vec[i] = ndarray<IndexType, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        }

        return std::make_tuple(val_vec, ind_vec);
    }

    auto init_vector_arrays(std::vector<ndarray<Float, 1>>& val_vec,
                            std::vector<ndarray<IndexType, 1>>& ind_vec,
                            Float a,
                            Float b,
                            std::int64_t seed = 777) {
        auto& q = this->get_queue();
        const std::int64_t vector_count = val_vec.size();
        for (std::int64_t i = 0; i < vector_count; i++) {
            IndexType* ind_ptr = ind_vec[i].get_mutable_data();
            auto event = q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<1>(ind_vec[i].get_count()), [=](sycl::item<1> item) {
                    IndexType ind = item.get_id()[0];
                    ind_ptr[ind] = ind;
                });
            });

            fill_uniform(val_vec[i], a, b, seed);

            event.wait_and_throw();
        }

        return std::make_tuple(val_vec, ind_vec);
    }

    void fill_uniform(ndarray<Float, 1>& val, Float a, Float b, std::int64_t seed = 777) {
        IndexType elem_count = de::integral_cast<IndexType>(val.get_count());
        std::mt19937 rng(seed);
        std::uniform_real_distribution<Float> distr(a, b);

        // move generation to device when rng is available there
        Float* val_ptr = detail::host_allocator<Float>().allocate(val.get_count());
        for (IndexType el = 0; el < elem_count; el++) {
            val_ptr[el] = distr(rng);
        }
        val.assign(this->get_queue(), val_ptr, val.get_count()).wait_and_throw();
        detail::host_allocator<Float>().deallocate(val_ptr, val.get_count());
    }

    void run(ndarray<Float, 1>& val, ndarray<IndexType, 1>& ind) {
        INFO("benchmark sort with indices");
        const auto name =
            fmt::format("Basic sort with indices: val_type {}, indices_type {}, elem_count {}",
                        type2str<Float>::name,
                        type2str<IndexType>::name,
                        val.get_count());

        this->get_queue().wait_and_throw();
        BENCHMARK(name.c_str()) {
            radix_sort_indices_inplace<Float, IndexType>{ this->get_queue() }(val, ind)
                .wait_and_throw();
        };
    }

    void run(std::vector<ndarray<Float, 1>>& val_vec, std::vector<ndarray<IndexType, 1>>& ind_vec) {
        INFO("benchmark sort with indices");
        const std::int64_t vector_count = val_vec.size();
        ONEDAL_ASSERT(vector_count > 0);
        const auto name = fmt::format(
            "Basic sort with indices: val_type {}, indices_type {}, vector_count {}, elem_count {}",
            type2str<Float>::name,
            type2str<IndexType>::name,
            vector_count,
            val_vec[0].get_count());

        this->get_queue().wait_and_throw();
        BENCHMARK(name.c_str()) {
            auto sorter = radix_sort_indices_inplace<Float, IndexType>{ this->get_queue() };
            for (std::int64_t i = 0; i < vector_count; i++) {
                //radix_sort_indices_inplace<Float, IndexType>{ this->get_queue() }(val_vec[i], ind_vec[i]).wait_and_throw();
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
        BENCHMARK(name.c_str()) {
            radix_sort<Integer>{ this->get_queue() }(val, val_out, sorted_elem_count)
                .wait_and_throw();
        };
    }
};

using sort_indices_types = COMBINE_TYPES((float, double), (std::uint32_t));

TEMPLATE_LIST_TEST_M(sort_with_indices_test,
                     "benchmark for array of vectors with indices",
                     "[sort][perf]",
                     sort_indices_types) {
    SKIP_IF(this->get_policy().is_cpu());

    // sizes used MNIST ds
    std::int64_t vector_count = GENERATE_COPY(784);
    std::int64_t elem_count = GENERATE_COPY(16384, 60000);

    auto [val_vec_, ind_vec_] = this->allocate_vector_arrays(vector_count, elem_count);
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
