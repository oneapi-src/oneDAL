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

#include <iostream>
#include "oneapi/dal/io/csv.hpp"

#include "oneapi/dal/algo/covariance/compute.hpp"
#include "oneapi/dal/algo/covariance/compute_types.hpp"
#include "oneapi/dal/algo/covariance/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/test/engine/linalg/matrix.hpp"
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

    void general_checks(const te::dataframe& input, const te::table_id& input_table_id) {
        const table data = input.get_table(this->get_policy(), input_table_id);

        INFO("create descriptor cov cor means")
        auto cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix | covariance::result_options::cor_matrix |
                covariance::result_options::means);
        INFO("run compute optional: cov cor means");
        auto compute_result = this->compute(cov_desc, data);
        check_compute_result(data, compute_result);

        INFO("create descriptor cov")
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix);
        INFO("run compute optional: cov");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(data, compute_result);

        INFO("create descriptor cor")
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cor_matrix);
        INFO("run compute optional: cor");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(data, compute_result);

        INFO("create descriptor means")
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::means);
        INFO("run compute optional: means");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(data, compute_result);

        INFO("create descriptor cov cor")
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix | covariance::result_options::cor_matrix);
        INFO("run compute optional: cov cor");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(data, compute_result);

        INFO("create descriptor cov means")
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cov_matrix | covariance::result_options::means);
        INFO("run compute optional: cov means");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(data, compute_result);

        INFO("create descriptor cor means")
        cov_desc =
            covariance::descriptor<Float, Method, covariance::task::compute>().set_result_options(
                covariance::result_options::cor_matrix | covariance::result_options::means);
        INFO("run compute optional: cor means");
        compute_result = this->compute(cov_desc, data);
        check_compute_result(data, compute_result);
    }

    void check_compute_result(const table& data, const covariance::compute_result<>& result) {
        if (result.get_result_options().test(result_options::cov_matrix)) {
            const auto cov_matrix = result.get_cov_matrix();
            INFO("check if cov matrix table shape is expected")
            REQUIRE(cov_matrix.get_row_count() == data.get_column_count());
            REQUIRE(cov_matrix.get_column_count() == data.get_column_count());
            INFO("check if there is no NaN in cov matrix table")
            REQUIRE(te::has_no_nans(cov_matrix));
            INFO("check if cov matrix values are expected")
            check_cov_matrix_values(data, cov_matrix);
        }
        if (result.get_result_options().test(result_options::cor_matrix)) {
            const auto cor_matrix = result.get_cor_matrix();
            INFO("check if cor matrix table shape is expected")
            REQUIRE(cor_matrix.get_row_count() == data.get_column_count());
            REQUIRE(cor_matrix.get_column_count() == data.get_column_count());
            INFO("check if there is no NaN in cor matrix table")
            REQUIRE(te::has_no_nans(cor_matrix));
            INFO("check if cor matrix values are expected")
            check_cor_matrix_values(data, cor_matrix);
        }
        if (result.get_result_options().test(result_options::means)) {
            const auto means = result.get_means();
            INFO("check if means table shape is expected")
            REQUIRE(means.get_row_count() == 1);
            REQUIRE(means.get_column_count() == data.get_column_count());
            INFO("check if there is no NaN in means table")
            REQUIRE(te::has_no_nans(means));
            INFO("check if means values are expected")
            check_means_values(data, means);
        }
    }

    void check_means_values(const table& data, const table& means) {
        const auto reference_means = compute_reference_means(data);
        const double tol = te::get_tolerance<Float>(1e-4, 1e-9);
        const double diff = te::abs_error(reference_means, means);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference_means(const table& data) {
        const auto data_matrix = la::matrix<double>::wrap(data);
        const auto row_count_data = data_matrix.get_row_count();
        const auto column_count_data = data_matrix.get_column_count();
        auto reference_means = la::matrix<double>::full({ 1, column_count_data }, 0.0);

        for (std::int64_t i = 0; i < column_count_data; i++) {
            double sum = 0;
            for (std::int64_t j = 0; j < row_count_data; j++) {
                sum += data_matrix.get(j, i);
            }
            reference_means.set(0, i) = sum / row_count_data;
        }
        return reference_means;
    }

    void check_cov_matrix_values(const table& data, const table& cov_matrix) {
        const auto reference_cov = compute_reference_cov(data);
        const double tol = te::get_tolerance<Float>(1e-2, 1e-9);
        const double diff = te::abs_error(reference_cov, cov_matrix);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference_cov(const table& data) {
        const auto data_matrix = la::matrix<double>::wrap(data);
        const auto row_count_data = data_matrix.get_row_count();
        const auto column_count_data = data_matrix.get_column_count();
        auto reference_means = compute_reference_means(data);
        auto reference_cov =
            la::matrix<double>::full({ column_count_data, column_count_data }, 0.0);
        for (std::int64_t i = 0; i < column_count_data; i++) {
            for (std::int64_t j = 0; j < column_count_data; j++) {
                double elem = 0;
                for (std::int64_t k = 0; k < row_count_data; k++) {
                    elem += (data_matrix.get(k, i) - reference_means.get(0, i)) *
                            (data_matrix.get(k, j) - reference_means.get(0, j));
                }
                reference_cov.set(i, j) = elem * (1 / static_cast<double>(row_count_data - 1));
            }
        }
        return reference_cov;
    }
    void check_cor_matrix_values(const table& data, const table& cor_matrix) {
        const auto reference_cor = compute_reference_cor(data);
        const double tol = te::get_tolerance<Float>(1e-4, 1e-9);
        const double diff = te::abs_error(reference_cor, cor_matrix);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference_cor(const table& data) {
        const auto data_matrix = la::matrix<double>::wrap(data);
        const auto column_count_data = data_matrix.get_column_count();
        auto reference_means = compute_reference_means(data);
        auto reference_cov = compute_reference_cov(data);
        auto reference_cor =
            la::matrix<double>::full({ column_count_data, column_count_data }, 0.0);
        for (std::int64_t i = 0; i < column_count_data; i++) {
            for (std::int64_t j = 0; j < column_count_data; j++) {
                reference_cor.set(i, j) =
                    reference_cov.get(i, j) /
                    std::sqrt(reference_cov.get(i, i) * reference_cov.get(j, j));
            }
        }

        return reference_cor;
    }
};

using covariance_types = COMBINE_TYPES((float, double), (covariance::method::dense));

TEMPLATE_LIST_TEST_M(covariance_batch_test,
                     "covariance  fill_normal common flow",
                     "[covariance][integration][batch]",
                     covariance_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 4, 4 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 100, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 250, 250 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 500, 100 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();
    this->general_checks(input, input_data_table_id);
}

TEMPLATE_LIST_TEST_M(covariance_batch_test,
                     "covariance compute one element matrix",
                     "[covariance][integration][batch]",
                     covariance_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();
    this->general_checks(input, input_data_table_id);
}

TEMPLATE_LIST_TEST_M(covariance_batch_test,
                     "covariance fill_uniform common flow",
                     "[covariance][integration][batch]",
                     covariance_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1000, 20 }.fill_uniform(-30, 30, 7777),
                           te::dataframe_builder{ 16, 16 }.fill_uniform(-5, 5, 7777),
                           te::dataframe_builder{ 100, 10 }.fill_uniform(0, 1, 7777),
                           te::dataframe_builder{ 100, 10 }.fill_uniform(-10, 10, 7777),
                           te::dataframe_builder{ 500, 40 }.fill_uniform(-100, 100, 7777),
                           te::dataframe_builder{ 500, 250 }.fill_uniform(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();
    this->general_checks(input, input_data_table_id);
}

TEMPLATE_LIST_TEST_M(covariance_batch_test,
                     "covariance fill_uniform nightly common flow",
                     "[covariance][integration][batch][nightly]",
                     covariance_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 5000, 20 }.fill_uniform(-30, 30, 7777),
                           te::dataframe_builder{ 10000, 200 }.fill_uniform(-30, 30, 7777),
                           te::dataframe_builder{ 1000000, 20 }.fill_uniform(-0.5, 0.5, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();
    this->general_checks(input, input_data_table_id);
}

} // namespace oneapi::dal::covariance::test
