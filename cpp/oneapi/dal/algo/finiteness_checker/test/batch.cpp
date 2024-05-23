/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/finiteness_checker/compute.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::finiteness_checker::test {

namespace te = dal::test::engine;

template <typename TestType>
class finite_checker_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    void general_checks(const te::dataframe& x_data,
                        bool allowNaN,
                        const te::table_id& x_data_table_id) {
        CAPTURE(allowNaN);
        const table x = x_data.get_table(this->get_policy(), x_data_table_id);

        INFO("create descriptor");
        const auto finiteness_desc =
            finiteness_checker::descriptor<Float, Method>{}.set_allow_NaN(allowNaN);

        INFO("run compute");
        const auto compute_result = this->compute(finiteness_desc, x);
        check_compute_result(sigma, x, y, compute_result);
    }

    void check_compute_result(bool allowNaN, const table& x_data, const bool& result) {
        INFO("check if there is no NaN in result values table");
        REQUIRE(te::has_no_nans(x) == (result && allowNaN));

        INFO("check if result values are expected");
        check_result_values(sigma, x_data, y_data, result_values);
    }

    void check_result_values(double sigma,
                             const table& x_data,
                             const table& y_data,
                             const table& result_values) {
        const auto reference = compute_reference(sigma, x_data, y_data);

        const auto col_count = reference.get_column_count();
        const auto row_count = reference.get_row_count();
        REQUIRE(row_count == result_values.get_row_count());
        REQUIRE(col_count == result_values.get_column_count());

        row_accessor<const Float> acc{ result_values };
        for (std::int64_t row = 0; row < row_count; ++row) {
            auto row_arr = acc.pull({ row, row + 1 });
            for (std::int64_t col = 0; col < col_count; ++col) {
                const auto res = row_arr[col];
                const auto gtr = reference.get(row, col);
                const auto rerr = std::abs(res - gtr) /
                                  std::max<double>({ double(1), std::abs(res), std::abs(gtr) });
                CAPTURE(row_count, col_count, x_data.get_column_count(), row, col, res, gtr, rerr);
                if (rerr > 1e-4)
                    FAIL();
            }
        }
    }

    la::matrix<double> compute_reference(double sigma, const table& x_data, const table& y_data) {
        const auto x_data_matrix = la::matrix<double>::wrap(x_data);
        const auto y_data_matrix = la::matrix<double>::wrap(y_data);
        const auto row_count_x = x_data_matrix.get_row_count();
        const auto row_count_y = y_data_matrix.get_row_count();
        const auto column_count = x_data_matrix.get_column_count();
        auto reference = la::matrix<double>::full({ row_count_x, row_count_y }, 0.0);

        const double inv_sigma = 1.0 / (sigma * sigma);
        for (std::int64_t i = 0; i < row_count_x; i++)
            for (std::int64_t j = 0; j < row_count_y; j++) {
                for (std::int64_t k = 0; k < column_count; k++) {
                    double diff = x_data_matrix.get(i, k) - y_data_matrix.get(j, k);
                    reference.set(i, j) += diff * diff;
                }
                reference.set(i, j) = std::exp(-0.5 * inv_sigma * reference.get(i, j));
            }
        return reference;
    }
};

using finiteness_types = COMBINE_TYPES((float, double), (finiteness_checker::method::dense));

TEMPLATE_LIST_TEST_M(finiteness_batch_test,
                     "finiteness checker typical",
                     "[finiteness_checker][integration][batch]",
                     finiteness_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe x_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 50, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 100, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 250, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 1100, 50 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto x_data_table_id = this->get_homogen_table_id();
    const te::dataframe y_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 50, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 100, 50 }.fill_normal(0, 1, 8888),
                           te::dataframe_builder{ 200, 50 }.fill_normal(0, 1, 8888),
                           te::dataframe_builder{ 1000, 50 }.fill_normal(0, 1, 8888));

    const double sigma = GENERATE_COPY(0.8, 1.0, 5.0);

    this->general_checks(x_data, y_data, sigma, x_data_table_id, y_data_table_id);
}

TEMPLATE_LIST_TEST_M(rbf_kernel_batch_test,
                     "rbf_kernel compute one element matrix",
                     "[rbf_kernel][integration][batch]",
                     rbf_kernel_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe x_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto x_data_table_id = this->get_homogen_table_id();

    const te::dataframe y_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 8888));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto y_data_table_id = this->get_homogen_table_id();

    const double sigma = GENERATE_COPY(0.8, 1.0, 5.0);

    this->general_checks(x_data, y_data, sigma, x_data_table_id, y_data_table_id);
}

} // namespace oneapi::dal::finiteness_checker::test
