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

#include "oneapi/dal/algo/svm.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
namespace pr = oneapi::dal::backend::primitives;
using oneapi::dal::detail::empty_delete;

const dal::csr_table convert_to_csr(const dal::table& data) {
    std::int64_t non_zero_count = 0;
    auto data_ndarray = pr::table2ndarray_1d<float>(data);
    auto data_ptr = data_ndarray.get_data();
    std::int64_t column_count = data.get_column_count();
    std::int64_t row_count = data.get_row_count();

    for (std::int64_t i = 0; i < data_ndarray.get_count(); i++) {
        if (std::fabs(data_ptr[i]) > std::numeric_limits<float>::epsilon()) {
            non_zero_count++;
        }
    }
    pr::ndarray<float, 1> compressed_data = pr::ndarray<float, 1>::empty({ non_zero_count });
    pr::ndarray<std::int64_t, 1> row_offsets = pr::ndarray<std::int64_t, 1>::empty({ non_zero_count });
    pr::ndarray<std::int64_t, 1> col_indices = pr::ndarray<std::int64_t, 1>::empty({ row_count + 1 });
    float* comp_data_ptr = compressed_data.get_mutable_data();
    std::int64_t* row_offsets_ptr = row_offsets.get_mutable_data();
    std::int64_t* col_indices_ptr = col_indices.get_mutable_data();

    std::uint64_t compressed_idx = 0;
    row_offsets_ptr[0] = 0; // zero-based indexing

    for (std::int64_t i = 0; i < data_ndarray.get_count(); i++) {
        if (std::fabs(data_ptr[i]) > std::numeric_limits<float>::epsilon()) {
            comp_data_ptr[compressed_idx] = data_ptr[i];
            const std::int64_t row_idx = i / column_count;
            const std::int64_t col_idx = i % column_count;
            row_offsets_ptr[row_idx + 1] = compressed_idx;
            row_offsets_ptr[std::min(row_count, row_idx + 2)] = compressed_idx; // in case when the whole row is zeros
            col_indices_ptr[compressed_idx] = col_idx;
            compressed_idx++;
        }
    }

    std::cout << "row_offsets:" << std::endl;
    for (int i = 0; i < row_count + 1; i++) {
        std::cout << row_offsets_ptr[i] << " ";
    }
    std::cout << std::endl;

    dal::csr_table t{ compressed_data.get_data(),
                              col_indices.get_data(),
                              row_offsets.get_data(),
                              row_count,
                              column_count,
                              empty_delete<const float>(),
                              empty_delete<const std::int64_t>(),
                              empty_delete<const std::int64_t>(),
                              oneapi::dal::sparse_indexing::zero_based};
    return t;
}

int main(int argc, char const *argv[]) {
    const auto train_data_file_name = get_data_path("svm_multi_class_train_dense_data.csv");
    const auto train_response_file_name = get_data_path("svm_multi_class_train_dense_label.csv");
    const auto test_data_file_name = get_data_path("svm_multi_class_test_dense_data.csv");
    const auto test_response_file_name = get_data_path("svm_multi_class_test_dense_label.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    const auto y_train = dal::read<dal::table>(dal::csv::data_source{ train_response_file_name });
    const auto x_train_csr = convert_to_csr(x_train);
    const auto y_train_csr = convert_to_csr(y_train);

    const auto kernel_desc = dal::linear_kernel::descriptor{}.set_scale(1.0).set_shift(0.0);
    const auto svm_desc = dal::svm::descriptor{ kernel_desc }.set_class_count(5).set_c(1.0);
    const auto result_train = dal::train(svm_desc, x_train_csr, y_train_csr);

    std::cout << "Biases:\n" << result_train.get_biases() << std::endl;
    std::cout << "Coeffs indices:\n" << result_train.get_coeffs() << std::endl;

    const auto x_test = dal::read<dal::table>(dal::csv::data_source{ test_data_file_name });
    const auto y_true = dal::read<dal::table>(dal::csv::data_source{ test_response_file_name });

    const auto result_test = dal::infer(svm_desc, result_train.get_model(), x_test);

    std::cout << "Decision function result:\n" << result_test.get_decision_function() << std::endl;
    std::cout << "Responses result:\n" << result_test.get_responses() << std::endl;
    std::cout << "Responses true:\n" << y_true << std::endl;

    return 0;
}
