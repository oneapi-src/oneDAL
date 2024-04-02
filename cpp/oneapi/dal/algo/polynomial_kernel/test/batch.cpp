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

#include "oneapi/dal/algo/polynomial_kernel/compute.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::polynomial_kernel::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class polynomial_kernel_batch_test
        : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor(double scale, double shift, std::int64_t degree) const {
        return polynomial_kernel::descriptor<Float, Method>{}
            .set_scale(scale)
            .set_shift(shift)
            .set_degree(degree);
    }

    bool not_available_on_device() {
        return this->get_policy().is_gpu();
    }

    void general_checks(const te::dataframe& x_data,
                        const te::dataframe& y_data,
                        double scale,
                        double shift,
                        std::int64_t degree,
                        const te::table_id& x_data_table_id,
                        const te::table_id& y_data_table_id) {
        CAPTURE(scale);
        CAPTURE(shift);
        CAPTURE(degree);
        const table x = x_data.get_table(this->get_policy(), x_data_table_id);
        const table y = y_data.get_table(this->get_policy(), y_data_table_id);

        INFO("create descriptor");
        const auto polynomial_kernel_desc = get_descriptor(scale, shift, degree);

        INFO("run compute");
        const auto compute_result = this->compute(polynomial_kernel_desc, x, y);
        check_compute_result(scale, shift, degree, x, y, compute_result);
    }

    void check_compute_result(double scale,
                              double shift,
                              std::int64_t degree,
                              const table& x_data,
                              const table& y_data,
                              const polynomial_kernel::compute_result<>& result) {
        const auto result_values = result.get_values();

        INFO("check if result values table shape is expected");
        REQUIRE(result_values.get_row_count() == x_data.get_row_count());
        REQUIRE(result_values.get_column_count() == y_data.get_row_count());

        INFO("check if there is no NaN in result values table");
        REQUIRE(te::has_no_nans(result_values));

        INFO("check if result values are expected");
        check_result_values(scale, shift, degree, x_data, y_data, result_values);
    }

    void check_result_values(double scale,
                             double shift,
                             std::int64_t degree,
                             const table& x_data,
                             const table& y_data,
                             const table& result_values) {
        const auto reference = compute_reference(scale, shift, degree, x_data, y_data);
        const double tol = te::get_tolerance<Float>(1e-2, 1e-9);
        const double diff = te::abs_error(reference, result_values);
        CHECK(diff < tol);
    }

    la::matrix<double> compute_reference(double scale,
                                         double shift,
                                         std::int64_t degree,
                                         const table& x_data,
                                         const table& y_data) {
        const auto x_data_matrix = la::matrix<double>::wrap(x_data);
        const auto y_data_matrix = la::matrix<double>::wrap(y_data);

        auto reference = la::dot(x_data_matrix, y_data_matrix.t(), scale);

        la::enumerate_linear(reference, [&](std::int64_t i, double) {
            reference.set(i) += shift;
            reference.set(i) = std::pow(reference.set(i), degree);
        });

        return reference;
    }
};

using polynomial_kernel_types = COMBINE_TYPES((float, double), (polynomial_kernel::method::dense));

TEMPLATE_LIST_TEST_M(polynomial_kernel_batch_test,
                     "polynomial_kernel common flow",
                     "[polynomial_kernel][integration][batch]",
                     polynomial_kernel_types) {
    SKIP_IF(this->not_available_on_device());
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

    // Homogen floating point type is the same as algorithm's floating point type
    const auto y_data_table_id = this->get_homogen_table_id();

    const double scale = GENERATE_COPY(1.0, 2.0);
    const double shift = GENERATE_COPY(0.0, 1.0);
    const std::int64_t degree = GENERATE_COPY(1.0, 2.0);

    this->general_checks(x_data, y_data, scale, shift, degree, x_data_table_id, y_data_table_id);
}

TEMPLATE_LIST_TEST_M(polynomial_kernel_batch_test,
                     "polynomial_kernel compute one element matrix",
                     "[polynomial_kernel][integration][batch]",
                     polynomial_kernel_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe x_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto x_data_table_id = this->get_homogen_table_id();

    const te::dataframe y_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 8888));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto y_data_table_id = this->get_homogen_table_id();

    const double scale = GENERATE_COPY(1.0, 2.0);
    const double shift = GENERATE_COPY(0.0, 1.0);
    const std::int64_t degree = GENERATE_COPY(1.0, 2.0);

    this->general_checks(x_data, y_data, scale, shift, degree, x_data_table_id, y_data_table_id);
}

} // namespace oneapi::dal::polynomial_kernel::test
