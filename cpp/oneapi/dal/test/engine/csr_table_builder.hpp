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

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/test/engine/math.hpp"
// TODO Remove debug prints
#include <iostream>

namespace oneapi::dal::test::engine {

namespace pr = dal::backend::primitives;

/**
* Generates random CSR table based on inputs
*/
template <typename Float>
struct csr_table_builder {
    std::int32_t row_count_, column_count_;
    float nonzero_fraction_;
    sparse_indexing indexing_;
    pr::ndarray<Float, 1> data_;
    pr::ndarray<std::int64_t, 1> column_indices_;
    pr::ndarray<std::int64_t, 1> row_offsets_;

    csr_table_builder(std::int32_t row_count,
                      std::int32_t column_count,
                      float nnz_fraction = 0.2,
                      sparse_indexing indexing = sparse_indexing::zero_based)
            : row_count_(row_count),
              column_count_(column_count),
              nonzero_fraction_(nnz_fraction),
              indexing_(indexing)
    {
        std::int32_t total_count = row_count_ * column_count_;
        std::int32_t nonzero_count = total_count * nnz_fraction;
        std::int32_t indexing_shift = bool(indexing == sparse_indexing::one_based);

        std::uint32_t seed = 42;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<Float> uniform_data(-3.0f, 3.0f);
        std::uniform_int_distribution<std::int64_t> uniform_indices(0, column_count_ - 1);
        std::uniform_int_distribution<std::int64_t> uniform_ind_count(0, column_count_ - 1);

        pr::ndarray<Float, 1> data = pr::ndarray<Float, 1>::empty(nonzero_count);
        auto col_indices = pr::ndarray<std::int64_t, 1>::empty(nonzero_count);
        auto row_offsets = pr::ndarray<std::int64_t, 1>::empty(row_count_ + 1);

        auto data_ptr = data.get_mutable_data();
        auto col_indices_ptr = col_indices.get_mutable_data();
        auto row_offsets_ptr = row_offsets.get_mutable_data();
        // Generate data
        for (std::int32_t i = 0; i < nonzero_count; ++i) {
            data_ptr[i] = uniform_data(rng);
        }
        // Generate column indices and fill row offsets
        std::int32_t row_idx = 0;
        std::int32_t fill_count = 0;
        row_offsets_ptr[0] = indexing_shift;
        while (fill_count < nonzero_count && row_idx < row_count_) {
            // Generate the number of non-zero columns for current row
            int nnz_col_count = uniform_ind_count(rng);
            nnz_col_count = std::min(nnz_col_count, nonzero_count - fill_count);
            for (std::int32_t i = 0; i < nnz_col_count; ++i) {
                std::int32_t col_idx = uniform_indices(rng) + indexing_shift;
                col_indices_ptr[fill_count + i] = col_idx;
            }
            std::sort(col_indices_ptr + fill_count, col_indices_ptr + fill_count + nnz_col_count);
            // Remove duplications
            std::int32_t dup_count = 0;
            for (std::int32_t i = 1; i < nnz_col_count; ++i) {
                auto cur_ptr = col_indices_ptr + (fill_count - dup_count);
                if (cur_ptr[i] == cur_ptr[i - 1]) {
                    ++dup_count;
                    // Shift the tail if there is duplication
                    for (std::int32_t j = i + 1; j < nnz_col_count; ++j) {
                        cur_ptr[j - 1] = cur_ptr[j];
                    }
                }
            }
            fill_count += (nnz_col_count - dup_count);
            // Update row offsets
            row_offsets_ptr[row_idx + 1] = fill_count + indexing_shift;
            row_idx++;
        }
        if (row_idx < row_count_) {
            for (std::int32_t i = row_idx; i <= row_count_; ++i) {
                row_offsets_ptr[i] = nonzero_count;
            }
        }
        std::cout << "DATA nnz=" << nonzero_count << ", row_count=" << row_count_
                  << ", column_indices_count=" << nonzero_count << ":\n";
        for (int i = 0; i < nonzero_count; ++i) {
            std::cout << data_ptr[i] << " ";
        }
        std::cout << std::endl << "row_offsets:" << std::endl;
        for (int i = 0; i < row_count_ + 1; ++i) {
            std::cout << row_offsets_ptr[i] << " ";
        }
        std::cout << std::endl << "column_indices:" << std::endl;
        for (int i = 0; i < nonzero_count; ++i) {
            std::cout << col_indices_ptr[i] << " ";
        }
        std::cout << std::endl << "--------------------------" << std::endl;
        // Assign to class attributes
        data_ = data;
        column_indices_ = col_indices;
        row_offsets_ = row_offsets;

        // Create table
        //return csr_table::wrap(data, col_indices, row_offsets, column_count_, indexing);
    }

    csr_table build_random_csr_table(device_test_policy& policy) {
        auto queue = policy.get_queue();
        auto device_data = data_.to_device(queue);
        auto device_column_indices = column_indices_.to_device(queue);
        auto device_row_offsets = row_offsets_.to_device(queue);
        
        return csr_table::wrap(queue,
                               device_data.get_data(),
                               device_column_indices.get_data(),
                               device_row_offsets.get_data(),
                               row_count_,
                               column_count_,
                               indexing_);
    }

    csr_table build_random_csr_table(host_test_policy& policy) {
        return csr_table::wrap(data_.get_data(),
                               column_indices_.get_data(),
                               row_offsets_.get_data(),
                               row_count_,
                               column_count_,
                               indexing_);
    }
};

} //namespace oneapi::dal::test::engine