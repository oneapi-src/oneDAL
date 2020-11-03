/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gtest/gtest.h"
#include "oneapi/dal/algo/pca/train.hpp"
#include "oneapi/dal/algo/pca/infer.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi::dal;
namespace pca = oneapi::dal::pca;

using pca_methods = testing::Types<pca::method::cov, pca::method::svd>;

template <typename Tuple>
class pca_common_overflow_cpu_tests : public ::testing::Test {};

TYPED_TEST_SUITE_P(pca_common_overflow_cpu_tests);

struct homogen_table_dummy_impl {
    explicit homogen_table_dummy_impl(std::int64_t row_count, std::int64_t column_count)
            : _row_count(row_count),
              _column_count(column_count) {}

    std::int64_t get_column_count() const noexcept {
        return _column_count;
    }

    std::int64_t get_row_count() const noexcept {
        return _row_count;
    }

    template <typename Data>
    void pull_rows(array<Data>& block, const range&) const {
        block.reset();
    }

    template <typename Data>
    void pull_column(array<Data>& block, std::int64_t, const range&) const {
        block.reset();
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Data>
    void pull_rows(sycl::queue&, array<Data>& block, const range&, const sycl::usm::alloc&) const {
        block.reset();
    }

    template <typename Data>
    void pull_column(sycl::queue&,
                     array<Data>& block,
                     std::int64_t,
                     const range&,
                     const sycl::usm::alloc&) const {
        block.reset();
    }
#endif

    const void* get_data() const {
        return nullptr;
    }

    const table_metadata& get_metadata() const {
        return m;
    }

    data_layout get_data_layout() const {
        return data_layout::column_major;
    }

    table_metadata m;
    std::int64_t _row_count;
    std::int64_t _column_count;
};

TYPED_TEST_P(pca_common_overflow_cpu_tests, train_throws_if_component_count_leads_to_overflow) {
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t component_count = 0x7FFFFFFFFFFFFFFF;

    homogen_table data_table{ homogen_table_dummy_impl(row_count, component_count) };

    auto pca_desc =
        pca::descriptor<float, TypeParam, pca::task::dim_reduction>().set_component_count(
            component_count);

    ASSERT_THROW(train(pca_desc, data_table), range_error);
}

TYPED_TEST_P(pca_common_overflow_cpu_tests, infer_throws_if_component_count_leads_to_overflow) {
    constexpr std::int64_t row_count = 8;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t component_count = 0x7FFFFFFFFFFFFFFF;

    homogen_table data_infer_table{ homogen_table_dummy_impl(row_count, column_count) };
    homogen_table dummy_eigenvectors_table{ homogen_table_dummy_impl(component_count,
                                                                     column_count) };

    auto pca_desc =
        pca::descriptor<float, TypeParam, pca::task::dim_reduction>().set_component_count(
            component_count);

    pca::model m;
    m.set_eigenvectors(dummy_eigenvectors_table);

    ASSERT_THROW(infer(pca_desc, m, data_infer_table), range_error);
}

REGISTER_TYPED_TEST_SUITE_P(pca_common_overflow_cpu_tests,
                            infer_throws_if_component_count_leads_to_overflow,
                            train_throws_if_component_count_leads_to_overflow);

INSTANTIATE_TYPED_TEST_SUITE_P(run_pca_common_overflow_cpu_tests,
                               pca_common_overflow_cpu_tests,
                               pca_methods);
