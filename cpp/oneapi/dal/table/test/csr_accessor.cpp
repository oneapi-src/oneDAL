/*******************************************************************************
* Copyright 2023 Intel Corporation
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

/// Tests dal::csr_accessor class on a fixed data table:
///     | 1,  2,  0,  3 |
/// A = | 0,  0,  4,  0 |
///     | 0,  1,  0, 11 |
///     | 0,  8,  0,  0 |
/// which is stored in CSR format, zero-based or one-based
///
/// @tparam TestType    The tuple of two elements.
///                     The first element in the tuple has the data type of the elements
///                     in the table A.
///                     The second elements in the tuple has the data type of the elements
///                     in the data block pulled from the table by csr_accessor.
template <typename TestType>
class csr_accessor_test : public te::policy_fixture {
public:
    using table_data_t = std::tuple_element_t<0, TestType>;
    using accessor_data_t = std::tuple_element_t<1, TestType>;
    using array_d = dal::array<accessor_data_t>;
    using array_i = dal::array<std::int64_t>;

    static constexpr std::int64_t row_count = 4; // number of rows in the tested table
    static constexpr std::int64_t column_count = 4; // number of columns in the tested table

    // number of non-zero elements in the tested table
    static constexpr std::int64_t element_count = 7;

    /// Check that the test with the particular `TestType` cannot be run on the system
    /// that does not support 64-bit floating point operations.
    ///
    /// @return True if this test cannot be run on the system.
    ///         False, otherwise.
    bool not_float64_friendly() {
        constexpr bool is_double =
            std::is_same_v<table_data_t, double> || std::is_same_v<accessor_data_t, double>;
        return is_double && !this->get_policy().has_native_float64();
    }

    /// Initialize column indices and row offsets for the tested data table in CSR format
    /// based on the table's indexing scheme.
    void initialize_indices() {
        if (table_indexing_ == sparse_indexing::zero_based) {
            column_indices_ = column_indices_zero_based_.data();
            row_offsets_ = row_offsets_zero_based_.data();
        }
        else {
            column_indices_ = column_indices_one_based_.data();
            row_offsets_ = row_offsets_one_based_.data();
        }
    }

    /// Check that `pull` method of `csr_accessor` class works correctly.
    ///
    /// @param[in] start_idx    Zero-based index of the first row of the block of data
    ///                         to be pulled from the table.
    /// @param[in] end_idx      Either zero-based index of the row that goes after the last row
    ///                         of the block of data to be pulled from the table;
    ///                         or -1, if the last row to be pulled is equal to the last row
    ///                         of the whole table.
    void pull_checks(std::int64_t start_idx, std::int64_t end_idx) {
        ONEDAL_ASSERT(0 <= start_idx && start_idx <= row_count);
        ONEDAL_ASSERT(end_idx == -1 || end_idx <= row_count);

        initialize_indices();

        csr_table t = csr_table::wrap(data_.data(),
                                      column_indices_,
                                      row_offsets_,
                                      row_count,
                                      column_count,
                                      table_indexing_);

        const auto [data_array, cidx_array, ridx_array] =
            csr_accessor<const accessor_data_t>(t).pull({ start_idx, end_idx }, accessor_indexing_);

        check_pull_results(start_idx, end_idx, data_array, cidx_array, ridx_array);
    }

    /// Check that the block of rows in CSR format pulled from the table is correct.
    ///
    /// @param[in] start_idx    Zero-based index of the first row of the block of data
    ///                         pulled from the table.
    /// @param[in] end_idx      Either zero-based index of the row that goes after the last row
    ///                         of the block of data pulled from the table,
    ///                         or -1, if the last pulled row is equal to the last row
    ///                         of the whole table.
    /// @param[in] data_array   The block of values pulled from the table in the CSR layout.
    /// @param[in] cidx_array   The block of column indicies pulled from the table
    ///                         in the CSR layout.
    /// @param[in] ridx_array   The block of row offsets pulled from the table
    ///                         in the CSR layout.
    void check_pull_results(std::int64_t start_idx,
                            std::int64_t end_idx,
                            const array_d& data_array,
                            const array_i& cidx_array,
                            const array_i& ridx_array) {
        if (end_idx == -1)
            end_idx = row_count;

        // check that the sizes of the pulled data blocks are correct
        REQUIRE(data_array.get_count() == row_offsets_[end_idx] - row_offsets_[start_idx]);
        REQUIRE(data_array.get_count() == cidx_array.get_count());
        REQUIRE(ridx_array.get_count() == end_idx - start_idx + 1);

        const table_data_t* const data = data_.data();
        const std::int64_t data_shift = row_offsets_[start_idx] - row_offsets_[0];

        // check that no data copying happened when possible
        if (std::is_same_v<table_data_t, accessor_data_t>)
            REQUIRE(reinterpret_cast<const table_data_t*>(data) + data_shift ==
                    reinterpret_cast<const table_data_t*>(data_array.get_data()));

        // check that the data values in the pulled data block are correct
        for (std::int64_t i = 0; i < data_array.get_count(); i++) {
            REQUIRE(data_array[i] == data[data_shift + i]);
        }

        // check column indices and row offsets depending on the indexing schemes
        // ot the tested table and the accessor

        if (table_indexing_ == accessor_indexing_) {
            REQUIRE(column_indices_ + data_shift == cidx_array.get_data());
            if (start_idx == 0) {
                REQUIRE(row_offsets_ == ridx_array.get_data());
            }
            for (std::int64_t i = 0; i < data_array.get_count(); i++) {
                REQUIRE(cidx_array[i] == column_indices_[data_shift + i]);
            }

            for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
                REQUIRE(ridx_array[i] == row_offsets_[start_idx + i] - data_shift);
            }
        }
        else if (table_indexing_ ==
                 sparse_indexing::zero_based /* && accessor_indexing == one_based */) {
            for (std::int64_t i = 0; i < data_array.get_count(); i++) {
                REQUIRE(cidx_array[i] - 1 == column_indices_[data_shift + i]);
            }

            for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
                REQUIRE(ridx_array[i] - 1 == row_offsets_[start_idx + i] - data_shift);
            }
        }
        else /* table_indexing == sparse_indexing::one_based && accessor_indexing == zero_based */ {
            for (std::int64_t i = 0; i < data_array.get_count(); i++) {
                REQUIRE(cidx_array[i] + 1 == column_indices_[data_shift + i]);
            }

            for (std::int64_t i = 0; i < ridx_array.get_count(); i++) {
                REQUIRE(ridx_array[i] + 1 == row_offsets_[start_idx + i] - data_shift);
            }
        }
    }

protected:
    // Indexing schemes in the tested table and in the accessor.
    // Can be sparse_indexing::zero_based or sparse_indexing::one_based.
    sparse_indexing table_indexing_;
    sparse_indexing accessor_indexing_;

private:
    static constexpr std::array<table_data_t, element_count> data_ = { 1, 2, 3, 4, 1, 11, 8 };

    static constexpr std::array<std::int64_t, element_count> column_indices_one_based_ = { 1, 2, 4,
                                                                                           3, 2, 4,
                                                                                           2 };
    static constexpr std::array<std::int64_t, element_count> column_indices_zero_based_ = { 0, 1, 3,
                                                                                            2, 1, 3,
                                                                                            1 };
    const std::int64_t* column_indices_;

    static constexpr std::array<std::int64_t, row_count + 1> row_offsets_one_based_ = { 1,
                                                                                        4,
                                                                                        5,
                                                                                        7,
                                                                                        8 };
    static constexpr std::array<std::int64_t, row_count + 1> row_offsets_zero_based_ = { 0,
                                                                                         3,
                                                                                         4,
                                                                                         6,
                                                                                         7 };
    const std::int64_t* row_offsets_;
};

// Generate a sequense of `TestType` template parameters for the `csr_accessor_test`.
// The first elements in the tuple will be taken from the first list of types;
// the second elements in the tuple will be taken from the second list of type.
// As the results, 12 variants of `TestType` parameter are created.
using csr_accessor_types = COMBINE_TYPES((std::int32_t, std::int64_t, float, double),
                                         (std::int32_t, float, double));

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
