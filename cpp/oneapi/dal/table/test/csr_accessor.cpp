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
#include "oneapi/dal/table/csr_accessor.hpp"

namespace oneapi::dal {

namespace te = dal::test::engine;

template <typename TestType>
class csr_accessor_test : public te::policy_fixture {
public:
    using table_data_t = std::tuple_element_t<0, TestType>;
    using accessor_data_t = std::tuple_element_t<1, TestType>;
    using array_d = dal::array<accessor_data_t>;
    using array_i = dal::array<std::int64_t>;

    static constexpr std::int64_t row_count = 4;
    static constexpr std::int64_t column_count = 4;
    static constexpr std::int64_t element_count = 7;

    bool not_float64_friendly() {
        constexpr bool is_double =
            std::is_same_v<table_data_t, double> || std::is_same_v<accessor_data_t, double>;
        return is_double && !this->get_policy().has_native_float64();
    }

    void initialize_indices() {
        if (table_indexing_ == sparse_indexing::zero_based) {
            column_indices = column_indices_zero_based_.data();
            row_offsets = row_offsets_zero_based_.data();
        }
        else {
            column_indices = column_indices_one_based_.data();
            row_offsets = row_offsets_one_based_.data();
        }
    }

    void pull_checks(std::int64_t start_idx, std::int64_t end_idx) {
        ONEDAL_ASSERT(0 <= start_idx && start_idx <= row_count);
        ONEDAL_ASSERT(end_idx == -1 || end_idx <= row_count);

        initialize_indices();

        csr_table t = csr_table::wrap(data_.data(),
                                      column_indices,
                                      row_offsets,
                                      row_count,
                                      column_count,
                                      table_indexing_);

        const auto [data_array, cidx_array, ridx_array] =
            csr_accessor<const accessor_data_t>(t).pull({ start_idx, end_idx }, accessor_indexing_);

        check_pull_results(start_idx, end_idx, data_array, cidx_array, ridx_array);
    }

    void check_pull_results(std::int64_t start_idx,
                            std::int64_t end_idx,
                            const array_d& data_array,
                            const array_i& cidx_array,
                            const array_i& ridx_array) {
        if (end_idx == -1)
            end_idx = row_count;

        REQUIRE(data_array.get_count() == row_offsets[end_idx] - row_offsets[start_idx]);
        REQUIRE(data_array.get_count() == cidx_array.get_count());
        REQUIRE(ridx_array.get_count() == end_idx - start_idx + 1);

        const table_data_t* const data = data_.data();
        const std::int64_t data_shift = row_offsets[start_idx] - row_offsets[0];

        if (std::is_same_v<table_data_t, accessor_data_t>)
            REQUIRE(reinterpret_cast<const table_data_t*>(data) + data_shift ==
                    reinterpret_cast<const table_data_t*>(data_array.get_data()));

        for (std::int64_t i = 0; i < data_array.get_count(); i++) {
            REQUIRE(data_array[i] == data[data_shift + i]);
        }

        if (table_indexing_ == accessor_indexing_) {
            REQUIRE(column_indices + data_shift == cidx_array.get_data());
            if (start_idx == 0) {
                REQUIRE(row_offsets == ridx_array.get_data());
            }
            for (std::int64_t i = 0; i < data_array.get_count(); i++) {
                REQUIRE(cidx_array[i] == column_indices[data_shift + i]);
            }

            for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
                REQUIRE(ridx_array[i] == row_offsets[start_idx + i] - data_shift);
            }
        }
        else if (table_indexing_ ==
                 sparse_indexing::zero_based /* && accessor_indexing == one_based */) {
            for (std::int64_t i = 0; i < data_array.get_count(); i++) {
                REQUIRE(cidx_array[i] - 1 == column_indices[data_shift + i]);
            }

            for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
                REQUIRE(ridx_array[i] - 1 == row_offsets[start_idx + i] - data_shift);
            }
        }
        else /* table_indexing == sparse_indexing::one_based && accessor_indexing == zero_based */ {
            for (std::int64_t i = 0; i < data_array.get_count(); i++) {
                REQUIRE(cidx_array[i] + 1 == column_indices[data_shift + i]);
            }

            for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
                REQUIRE(ridx_array[i] + 1 == row_offsets[start_idx + i] - data_shift);
            }
        }
    }

protected:
    sparse_indexing table_indexing_;
    sparse_indexing accessor_indexing_;

private:
    static constexpr std::array<table_data_t, element_count> data_ = { 1, 2, 3, 4, 1, 11, 8 };

    static constexpr std::array<std::int64_t, element_count> column_indices_one_based_ = {
        1, 2, 4, 3, 2, 4, 2
    };
    static constexpr std::array<std::int64_t, element_count> column_indices_zero_based_ = {
        0, 1, 3, 2, 1, 3, 1
    };
    const std::int64_t* column_indices;

    static constexpr std::array<std::int64_t, row_count + 1> row_offsets_one_based_ = {
        1, 4, 5, 7, 8
    };
    static constexpr std::array<std::int64_t, row_count + 1> row_offsets_zero_based_ = {
        0, 3, 4, 6, 7
    };
    const std::int64_t* row_offsets;
};

using csr_accessor_types = COMBINE_TYPES((int, std::int64_t, float, double), (int, float, double));

TEMPLATE_LIST_TEST_M(csr_accessor_test,
                     "CSR accessor can read the whole table",
                     "[csr_accessor][integration]",
                     csr_accessor_types) {
    SKIP_IF(this->not_float64_friendly());

    this->table_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    this->accessor_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);

    this->pull_checks(0, -1);
}

TEMPLATE_LIST_TEST_M(csr_accessor_test,
                     "CSR accessor can read the part of the table",
                     "[csr_accessor][integration]",
                     csr_accessor_types) {
    SKIP_IF(this->not_float64_friendly());

    this->table_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);
    this->accessor_indexing_ = GENERATE(sparse_indexing::zero_based, sparse_indexing::one_based);

    this->pull_checks(1, 3);
}

} // namespace oneapi::dal
