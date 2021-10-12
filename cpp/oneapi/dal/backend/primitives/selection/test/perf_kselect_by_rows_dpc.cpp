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

#include <type_traits>
#include <tuple>

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/selection/kselect_by_rows.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename TestType>
class selection_test : public te::float_algo_fixture<TestType> {
public:
    using float_t = TestType;

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    auto allocate_matrices(std::int64_t k, std::int64_t row_count, std::int64_t col_count) {
        auto data = ndarray<float_t, 2>::empty(this->get_queue(),
                                               { row_count, col_count },
                                               sycl::usm::alloc::device);
        auto selection = ndarray<float_t, 2>::empty(this->get_queue(),
                                                    { row_count, k },
                                                    sycl::usm::alloc::device);
        auto indices = ndarray<std::int32_t, 2>::empty(this->get_queue(),
                                                       { row_count, k },
                                                       sycl::usm::alloc::device);
        return std::make_tuple(data, selection, indices);
    }

    void fill_constant(ndarray<float_t, 2>& data, float_t a) {
        auto count = data.get_count();
        float_t* data_ptr = data.get_mutable_data();

        auto event = this->get_queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(count), [=](sycl::item<1> item) {
                std::int32_t ind = item.get_id()[0];
                data_ptr[ind] = ind;
            });
        });
        event.wait_and_throw();
    }

    void run(ndarray<float_t, 2>& data,
             std::int64_t k,
             ndarray<float_t, 2>& selection,
             ndarray<std::int32_t, 2>& indices) {
        INFO("benchmark sort with indices");
        const auto name = fmt::format("Selection: val_type {}, k {}, elem_count {}",
                                      te::type2str<float_t>::name(),
                                      k,
                                      data.get_count());

        this->get_queue().wait_and_throw();
        kselect_by_rows<float_t> sel(this->get_queue(), data.get_shape(), k);
        BENCHMARK(name.c_str()) {
            sel(this->get_queue(), data, k, selection, indices).wait_and_throw();
        };
    }
};

using selection_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(selection_test,
                     "benchmark for selection (k <= 64)",
                     "[selection][perf]",
                     selection_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    std::int64_t k = GENERATE_COPY(32, 64);
    std::int64_t row_count = GENERATE_COPY(1024);
    std::int64_t col_count = GENERATE_COPY(64 * 1024);
    auto [data, selection, indices] = this->allocate_matrices(k, row_count, col_count);
    this->fill_constant(data, 1.0f);
    this->run(data, k, selection, indices);
}

TEMPLATE_LIST_TEST_M(selection_test,
                     "benchmark for selection (k <= 16)",
                     "[selection][perf]",
                     selection_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    std::int64_t k = GENERATE(8, 16);
    std::int64_t row_count = GENERATE(1024);
    std::int64_t col_count = GENERATE(16 * 1024);
    auto [data, selection, indices] = this->allocate_matrices(k, row_count, col_count);
    this->fill_constant(data, 1.0f);
    this->run(data, k, selection, indices);
}

TEMPLATE_LIST_TEST_M(selection_test,
                     "benchmark for selection (k == 1)",
                     "[selection][perf]",
                     selection_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    std::int64_t k = GENERATE(1);
    std::int64_t row_count = GENERATE(1024);
    std::int64_t col_count = GENERATE(16 * 1024);
    auto [data, selection, indices] = this->allocate_matrices(k, row_count, col_count);
    this->fill_constant(data, 1.0f);
    this->run(data, k, selection, indices);
}

} // namespace oneapi::dal::backend::primitives::test
