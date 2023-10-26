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

namespace oneapi::dal::test::engine {

namespace pr = dal::backend::primitives;

/**
* Generates random CSR table based on inputs
*/
struct csr_table_builder {
    using Float = float;
    std::int64_t row_count_, column_count_;
    float nonzero_fraction_;
    sparse_indexing indexing_;
    const dal::array<Float> data_;
    const dal::array<std::int64_t> column_indices_;
    const dal::array<std::int64_t> row_offsets_;

    csr_table_builder(std::int64_t row_count,
                      std::int64_t column_count,
                      float nnz_fraction = 0.3,
                      sparse_indexing indexing = sparse_indexing::one_based)
            : row_count_(row_count),
              column_count_(column_count),
              nonzero_fraction_(nnz_fraction),
              indexing_(indexing),
              data_(dal::array<Float>::empty(nnz_fraction * row_count * column_count)),
              column_indices_(
                  dal::array<std::int64_t>::empty(nnz_fraction * row_count * column_count)),
              row_offsets_(dal::array<std::int64_t>::empty(row_count + 1)) {
        std::int64_t total_count = row_count_ * column_count_;
        std::int64_t nonzero_count = total_count * nnz_fraction;
        std::int64_t indexing_shift = bool(indexing == sparse_indexing::one_based);

        std::uint32_t seed = 42;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<Float> uniform_data(-10.0f, 10.0f);
        std::uniform_int_distribution<std::int64_t> uniform_indices(
            0,
            column_count_ - 1 - indexing_shift);
        std::uniform_int_distribution<std::int64_t> uniform_ind_count(1, column_count_ - 2);

        auto data_ptr = data_.get_mutable_data();
        auto col_indices_ptr = column_indices_.get_mutable_data();
        auto row_offsets_ptr = row_offsets_.get_mutable_data();
        // Generate data
        for (std::int32_t i = 0; i < nonzero_count; ++i) {
            data_ptr[i] = uniform_data(rng);
        }
        // Generate column indices and fill row offsets
        std::int64_t row_idx = 0;
        std::int64_t fill_count = 0;
        row_offsets_ptr[0] = indexing_shift;
        while (fill_count < nonzero_count && row_idx < row_count_) {
            // Generate the number of non-zero columns for current row
            std::int64_t nnz_col_count = uniform_ind_count(rng);
            nnz_col_count = std::min(nnz_col_count, nonzero_count - fill_count);
            for (std::int32_t i = 0; i < nnz_col_count; ++i) {
                std::int64_t col_idx = uniform_indices(rng) + indexing_shift;
                col_indices_ptr[fill_count + i] = col_idx + indexing_shift;
            }
            std::sort(col_indices_ptr + fill_count, col_indices_ptr + fill_count + nnz_col_count);
            // Remove duplications
            std::int64_t dup_count = 0;
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
                row_offsets_ptr[i] = nonzero_count + indexing_shift;
            }
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    csr_table build_csr_table(device_test_policy& policy) const {
        auto queue = policy.get_queue();
        auto nnz_count = data_.get_count();
        const auto copied_data =
            dal::array<Float>::empty(queue, nnz_count, sycl::usm::alloc::device);
        const auto copied_col_indices =
            dal::array<std::int64_t>::empty(queue, nnz_count, sycl::usm::alloc::device);
        const auto copied_row_offsets =
            dal::array<std::int64_t>::empty(queue, row_count_ + 1, sycl::usm::alloc::device);
        auto data_event = dal::backend::copy_host2usm(queue,
                                                      copied_data.get_mutable_data(),
                                                      data_.get_data(),
                                                      nnz_count);

        auto col_indices_event = dal::backend::copy_host2usm(queue,
                                                             copied_col_indices.get_mutable_data(),
                                                             column_indices_.get_data(),
                                                             nnz_count);
        auto row_offsets_event = dal::backend::copy_host2usm(queue,
                                                             copied_row_offsets.get_mutable_data(),
                                                             row_offsets_.get_data(),
                                                             row_count_ + 1);
        sycl::event::wait_and_throw({ data_event, col_indices_event, row_offsets_event });
        return csr_table::wrap(queue,
                               copied_data.get_data(),
                               copied_col_indices.get_data(),
                               copied_row_offsets.get_data(),
                               row_count_,
                               column_count_,
                               indexing_);
    }
#endif // ONEDAL_DATA_PARALLEL

    csr_table build_csr_table(host_test_policy& policy) const {
        auto nnz_count = data_.get_count();
        const auto copied_data = dal::array<Float>::empty(nnz_count);
        const auto copied_col_indices = dal::array<std::int64_t>::empty(nnz_count);
        const auto copied_row_offsets = dal::array<std::int64_t>::empty(row_count_ + 1);
        dal::backend::copy(copied_data.get_mutable_data(), data_.get_data(), nnz_count);
        dal::backend::copy(copied_col_indices.get_mutable_data(),
                           column_indices_.get_data(),
                           nnz_count);
        dal::backend::copy(copied_row_offsets.get_mutable_data(),
                           row_offsets_.get_data(),
                           row_count_ + 1);
        return csr_table::wrap(copied_data,
                               copied_col_indices,
                               copied_row_offsets,
                               column_count_,
                               indexing_);
    }

    table build_dense_table() const {
        const dal::array<Float> dense_data = dal::array<Float>::zeros(row_count_ * column_count_);
        std::int64_t indexing_shift = bool(indexing_ == sparse_indexing::one_based);
        auto data_ptr = dense_data.get_mutable_data();
        auto sparse_data_ptr = data_.get_data();
        auto row_offs_ptr = row_offsets_.get_data();
        auto col_indices_ptr = column_indices_.get_data();
        for (std::int32_t row_idx = 0; row_idx < row_count_; ++row_idx) {
            for (std::int32_t data_idx = row_offs_ptr[row_idx] - indexing_shift;
                 data_idx < row_offs_ptr[row_idx + 1] - indexing_shift;
                 ++data_idx) {
                data_ptr[row_idx * column_count_ + col_indices_ptr[data_idx] - indexing_shift] =
                    sparse_data_ptr[data_idx];
            }
        }
        return homogen_table::wrap(dense_data, row_count_, column_count_);
    }
};

} //namespace oneapi::dal::test::engine
