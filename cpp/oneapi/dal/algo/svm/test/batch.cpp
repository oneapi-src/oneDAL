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

#include "oneapi/dal/algo/svm/infer.hpp"
#include "oneapi/dal/algo/svm/train.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::svm::test {

namespace te = dal::test::engine;
namespace rbf = oneapi::dal::rbf_kernel;
namespace linear = oneapi::dal::linear_kernel;


template <typename TestType>
class svm_batch_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using KernelTypeLinear = typename linear::descriptor<Float, linear::method::dense>;
    using KernelTypeRBF = typename rbf::descriptor<Float, rbf::method::dense>;

    bool not_available_on_device() {
        constexpr bool is_smo = std::is_same_v<Method, svm::method::smo>;
        return get_policy().is_gpu() && is_smo;
    }

    void check_linear_kernel(const table& train_data,
                        const table& train_labels,
                        const svm::descriptor<Float, Method, svm::task::classification, KernelTypeLinear>& desc,
                        const std::int64_t support_vector_count,
                        const table& support_indices) {
        CAPTURE(support_vector_count);

        INFO("run training");
        auto train_result = this->train(desc, train_data, train_labels);
        const auto model = train_result.get_model();
        check_train_result(train_data, train_result, support_vector_count, support_indices);

        INFO("run inference");
        const auto infer_result = infer(desc, model, data);
        check_infer_result(train_data, infer_result);
    }

    void check_train_result(const table& train_data, const svm::train_result<>& result, 
        const std::int64_t support_vector_count,
        const table& support_indices) {
        check_shapes(train_data, result, support_vector_count);
        check_nans(result);
        
        SECTION("support_indices values is expected") {
            check_tables_values_match(support_indices, result.get_support_indices());
        }
    }

    void check_infer_result(const table& infer_data, const svm::infer_result<>& result) {
        chech_shapes(infer_data, result);
        check_nans(result);
    }

    void check_shapes(const table& train_data, const svm::train_result<>& result, 
        const std::int64_t support_vector_count) {
        const auto [support_vectors, support_indices, coeffs] = unpack_result(result);

        SECTION("support_vector_count is expected") {
            REQUIRE(result.get_support_vector_count() == support_vector_count);
        }asf
        }

        SECTION("support_indices shape is expected") {
            REQUIRE(support_indices.get_row_count() == suppas
    }

    void check_nans(const svm::train_result<>& result) {
        const auto [support_vectors, support_indices, coeffs] = unpack_result(result);

        SECTION("there is no NaN in support_vectors") {
            REQUIRE(te::has_no_nans(support_vectors));
        }

        SECTION("there is no NaN in support_indices") {
            REQUIRE(te::has_no_nans(support_indices));
        }

        SECTION("there is no NaN in coeffs") {
            REQUIRE(te::has_no_nans(coeffs));
        }
    }

    void check_tables_values_match(const table& left, const table& right) {
        SECTION("tables values shape is expected") {
            REQUIRE(left.get_row_count() == right.get_row_count());
            REQUIRE(left.get_column_count() == right.get_column_count());
            REQUIRE(left.get_column_count() == 1);
        }
        SECTION("tables values match is expected") {
            const auto left_rows = row_accessor<const Float>(left).pull({ 0, -1 });
            const auto right_rows = row_accessor<const Float>(right).pull({ 0, -1 });
            for (std::int64_t i = 0; i < left_rows.get_count(); i++) {
                const Float l = left_rows[i];
                const Float r = right_rows[i];
                if (l != r) {
                    CAPTURE(l, r);
                    FAIL("tables values mismatch");
                }
            }
        }
    }

private:
    static auto unpack_result(const svm::train_result<>& result) {
        const auto support_vectors = result.get_support_vectors();
        const auto support_indices = result.get_support_indices();
        const auto coeffs = result.get_coeffs();
        return std::make_tuple(support_vectors, support_indices, coeffs);
    }

};

using svm_types = COMBINE_TYPES((float, double), (svm::method::thunder, svm::method::smo));

TEMPLATE_LIST_TEST_M(svm_batch_test,
                     "svm can classify linear separable surface",
                     "[svm][integration][batch]",
                     svm_types) {
    SKIP_IF(this->not_available_on_device());

    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using KernelTypeLinear = typename linear::descriptor<Float, linear::method::dense>;

    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t element_count_train = row_count_train * column_count;

    constexpr std::array<Float, element_count_train> x_data = 
        { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f, };
    const auto x = homogen_table::wrap(x_data.data(), row_count_train, column_count);

    constexpr std::array<Float, row_count_train> y_data = { -1.f, -1.f, -1.f, +1.f, +1.f, +1.f, };
    const auto y = homogen_table::wrap(y_data.data(), row_count_train, 1);

    const double scale = GENERATE_COPY(0.1, 1.0);
    const double c = GENERATE_COPY(1.0, 10.0);

    const auto kernel_desc = KernelTypeLinear{}.set_scale(scale);
    const auto svm_desc = svm::descriptor<Float, Method, svm::task::classification, KernelTypeLinear>{}.set_c(c);

    constexpr std::int64_t support_vector_count = 2;

    constexpr std::array<Float, support_vector_count> support_indices_data = { 1, 3 };
    const auto support_indices = homogen_table::wrap(support_indices_data.data(), support_vector_count, 1);

    // constexpr std::array<Float, support_vector_count> support_indices_data = { -1.0, +1.0 };
    // const auto support_indices = homogen_table::wrap(support_indices_data.data(), support_vector_count, 1);

    this->check_linear_kernel(x, y, svm_desc, support_vector_count, support_indices);
}

}
