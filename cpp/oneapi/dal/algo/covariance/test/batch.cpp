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

#include "oneapi/dal/algo/covariance/compute.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::covariance::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename TestType>
class covariance_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor() const {
        return covariance::descriptor<Float, Method>, covariance::task::classification>()
            .set_result_options(covaraince::result_options::cov_matrix | covariance::result_options::cor_matrix |
                                covariance::result_options::means);
    }
    void general_checks(const te::dataframe& input,
                        const te::table_id& input_table_id) {
        const table data = input.get_table(this->get_policy(), input_table_id);

        INFO("create descriptor")
        const auto cov_desc = get_descriptor();

        INFO("run compute");
        const auto compute_result = this->compute(cov_desc, data);
        check_compute_result(data, compute_result);
    }

    void check_compute_result(const table& data,
                              const covariance::compute_result<>& result) {
        const auto cov_matrix = result.get_cov();
        const auto cor_matrix = result.get_cor();
        const auto means = result.get_means();

        INFO("check if cov matrix table shape is expected")
        REQUIRE(cov_matrix.get_row_count() == data.get_row_count());
        REQUIRE(cov_matrix.get_column_count() == data.get_column_count());

        INFO("check if cor matrix table shape is expected")
        REQUIRE(cor_matrix.get_row_count() == data.get_row_count());
        REQUIRE(cor_matrix.get_column_count() == data.get_column_count());

        INFO("check if means table shape is expected")
        REQUIRE(means.get_row_count() == 1);
        REQUIRE(means.get_column_count() == data.get_column_count());

        INFO("check if there is no NaN in cov matrix table")
        REQUIRE(te::has_no_nans(cov_matrix));
        INFO("check if there is no NaN in cor matrix table")
        REQUIRE(te::has_no_nans(cor_matrix));
        INFO("check if there is no NaN in means table")
        REQUIRE(te::has_no_nans(means));

        INFO("check if cov matrix values are expected")
        check_cov_matrix_values(data, cov_matrix);
        INFO("check if cor matrix values are expected")
        check_cor_matrix_values(data, cor_matrix);
        INFO("check if means values are expected")
        check_means_values(data, means);
    }

    void check_cov_matrix_values(const table& data,
                                 const table& cov_matrix) {
        const auto reference_cov = compute_reference_cov(data);
        const double tol = te::get_tolerance<Float>(1e-4, 1e-9);
        const double diff = te::abs_error(reference_cov, cov_matrix);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference_cov(const table& data) {
        const auto data_matrix = la::matrix<double>::wrap(data);
        const auto row_count_data = x_data_matrix.get_row_count();
        const auto column_count_data = x_data_matrix.get_column_count();
        auto reference_means = la::matrix<double>::full({ 1, column_count_data }, 0.0);
        auto reference_cov = la::matrix<double>::full({ row_count_data, column_count_data }, 0.0);
        for (std::int64_t i = 0; i < column_count_data; i++)
            for (std::int64_t j = 0; j < row_count_data; j++) {
                    double sum += data_matrix.get(j, i);
                }
                reference_means.set(0, i) = sum / column_count_data;
            }
        
        return reference;
    }
    void check_cor_matrix_values(const table& data,
                                 const table& cor_matrix) {
        const auto reference = compute_reference_cov(sigma, x_data, y_data);
        const double tol = te::get_tolerance<Float>(1e-4, 1e-9);
        const double diff = te::abs_error(reference, result_values);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference_cor(double sigma, const table& x_data, const table& y_data) {
        const auto data_matrix = la::matrix<double>::wrap(data);
        const auto row_count_data = x_data_matrix.get_row_count();
        const auto column_count_data = x_data_matrix.get_column_count();
        auto reference_means = la::matrix<double>::full({ 1, column_count_data }, 0.0);
        auto reference_cor = la::matrix<double>::full({ row_count_data, column_count_data }, 0.0);
        for (std::int64_t i = 0; i < column_count_data; i++)
            for (std::int64_t j = 0; j < row_count_data; j++) {
                    double sum += data_matrix.get(j, i);
                }
                reference_means.set(0, i) = sum / column_count_data;
            }
        
        return reference;
    }
    void check_means_values(const table& data,
                            const table& means) {
        const auto reference_means = compute_reference_means(data);
        const double tol = te::get_tolerance<Float>(1e-4, 1e-9);
        const double diff = te::abs_error(reference_means, means);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference_means(const table& data) {
        const auto data_matrix = la::matrix<double>::wrap(data);
        const auto row_count_data = x_data_matrix.get_row_count();
        const auto column_count_data = x_data_matrix.get_column_count();
        auto reference_means = la::matrix<double>::full({ 1, column_count_data }, 0.0);

        for (std::int64_t i = 0; i < column_count_data; i++)
            for (std::int64_t j = 0; j < row_count_data; j++) {
                    double sum += data_matrix.get(j, i);
                }
                reference_means.set(0, i) = sum / column_count_data;
            }
        return reference_means;
    }
};

using covariance_types = COMBINE_TYPES((float, double), (covariance::method::dense));

// TEMPLATE_LIST_TEST_M(covariance_batch_test,
//                      "covariance common flow",
//                      "[covariance][integration][batch]",
//                      covariance_types) {
//     SKIP_IF(this->not_float64_friendly());

//     const te::dataframe x_data =
//         GENERATE_DATAFRAME(te::dataframe_builder{ 50, 50 }.fill_normal(0, 1, 7777),
//                            te::dataframe_builder{ 100, 50 }.fill_normal(0, 1, 7777),
//                            te::dataframe_builder{ 250, 50 }.fill_normal(0, 1, 7777),
//                            te::dataframe_builder{ 1100, 50 }.fill_normal(0, 1, 7777));

//     // Homogen floating point type is the same as algorithm's floating point type
//     const auto x_data_table_id = this->get_homogen_table_id();

//     const te::dataframe y_data =
//         GENERATE_DATAFRAME(te::dataframe_builder{ 50, 50 }.fill_normal(0, 1, 7777),
//                            te::dataframe_builder{ 100, 50 }.fill_normal(0, 1, 8888),
//                            te::dataframe_builder{ 200, 50 }.fill_normal(0, 1, 8888),
//                            te::dataframe_builder{ 1000, 50 }.fill_normal(0, 1, 8888));

//     // Homogen floating point type is the same as algorithm's floating point type
//     const auto y_data_table_id = this->get_homogen_table_id();

//     const double sigma = GENERATE_COPY(0.8, 1.0, 5.0);

//     this->general_checks(x_data, y_data, sigma, x_data_table_id, y_data_table_id);
// }

// TEMPLATE_LIST_TEST_M(covariance_batch_test,
//                      "covariance compute one element matrix",
//                      "[covariance][integration][batch]",
//                      covariance_types) {
//     SKIP_IF(this->not_float64_friendly());

//     const te::dataframe x_data =
//         GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 7777));

//     // Homogen floating point type is the same as algorithm's floating point type
//     const auto x_data_table_id = this->get_homogen_table_id();

//     const te::dataframe y_data =
//         GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 8888));

//     // Homogen floating point type is the same as algorithm's floating point type
//     const auto y_data_table_id = this->get_homogen_table_id();

//     const double sigma = GENERATE_COPY(0.8, 1.0, 5.0);

//     this->general_checks(x_data, y_data, sigma, x_data_table_id, y_data_table_id);
}

} // namespace oneapi::dal::covariance::test
