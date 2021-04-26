/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/algo/pca/train.hpp"
#include "oneapi/dal/algo/pca/infer.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class pca_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    bool not_available_on_device() {
        constexpr bool is_svd = std::is_same_v<Method, pca::method::svd>;
        return this->get_policy().is_gpu() && is_svd;
    }

    auto get_descriptor(std::int64_t component_count, bool deterministic = false) const {
        return pca::descriptor<Float, Method>{}
            .set_component_count(component_count)
            .set_deterministic(deterministic);
    }

    table get_gold_data() {
        const std::int64_t row_count = 10;
        const std::int64_t column_count = 5;
        static const Float data[] = {
            4.59,  0.81,  -1.37, -0.04, -0.75, //
            4.87,  0.34,  -0.98, 4.1,   -0.12, //
            4.44,  0.11,  -0.4,  3.27,  4.82, //
            0.59,  0.98,  -1.88, -0.64, 2.54, //
            -1.98, 2.57,  4.11,  -1.3,  -0.66, //
            3.26,  2.8,   2.65,  0.83,  2.12, //
            0.21,  4.23,  2.71,  2.2,   3.85, //
            1.27,  -1.15, 2.84,  1.11,  -1.12, //
            0.25,  1.61,  1.69,  4.51,  0.09, //
            -0.01, 0.58,  0.83,  2.73,  -1.33, //
        };
        return homogen_table::wrap(data, row_count, column_count);
    }

    table get_gold_eigenvectors() {
        const std::int64_t component_count = 5;
        static const Float data[] = {
            0.58693744,  -0.46709857, -0.57460003, 0.32053585,  0.0664451, //
            0.18207761,  0.53331191,  -0.04798588, 0.19145584,  0.80216467, //
            -0.11054327, -0.01167475, 0.38203369,  0.90339519,  -0.15991021, //
            0.77818759,  0.2819281,   0.44987364,  -0.14493743, -0.30256812, //
            -0.06750144, 0.64635716,  -0.56497445, 0.15320478,  -0.48476607, //
        };
        return homogen_table::wrap(data, component_count, component_count);
    }

    table get_gold_eigenvalues() {
        const std::int64_t component_count = 5;
        static const Float data[] = {
            2.07061706, 1.32805591, 0.88554629, 0.38030539, 0.33547535,
        };
        return homogen_table::wrap(data, 1, component_count);
    }

    void general_checks(const te::dataframe& data,
                        std::int64_t component_count,
                        const te::table_id& data_table_id) {
        CAPTURE(component_count);
        const table x = data.get_table(this->get_policy(), data_table_id);

        INFO("create descriptor")
        const auto pca_desc = get_descriptor(component_count);

        INFO("run training");
        const auto train_result = this->train(pca_desc, x);
        const auto model = train_result.get_model();
        check_train_result(pca_desc, data, train_result);

        INFO("run inference");
        const auto infer_result = this->infer(pca_desc, model, x);
        check_infer_result(pca_desc, data, infer_result);
    }

    void check_train_result(const pca::descriptor<Float, Method>& desc,
                            const te::dataframe& data,
                            const pca::train_result<>& result) {
        const auto [means, variances, eigenvalues, eigenvectors] = unpack_result(result);

        check_shapes(desc, data, result);
        check_nans(result);

        INFO("check if eigenvectors order is descending")
        this->check_eigenvalues_order(eigenvalues);

        INFO("check if eigenvectors matrix is orthogonal")
        check_eigenvectors_orthogonality(eigenvectors);

        const auto bs = te::compute_basic_statistics<double>(data);

        INFO("check if means are expected")
        check_means(bs, means);

        INFO("check if variances are expected")
        check_variances(bs, variances);
    }

    void check_infer_result(const pca::descriptor<Float, Method>& desc,
                            const te::dataframe& data,
                            const pca::infer_result<>& result) {}

    void check_shapes(const pca::descriptor<Float, Method>& desc,
                      const te::dataframe& data,
                      const pca::train_result<>& result) {
        const auto [means, variances, eigenvalues, eigenvectors] = unpack_result(result);

        const std::int64_t expected_component_count =
            (desc.get_component_count() > 0) ? desc.get_component_count() : data.get_column_count();

        INFO("check if eigenvalues shape is expected")
        REQUIRE(eigenvalues.get_row_count() == 1);
        REQUIRE(eigenvalues.get_column_count() == expected_component_count);

        INFO("check if eigenvectors shape is expected")
        REQUIRE(eigenvectors.get_row_count() == expected_component_count);
        REQUIRE(eigenvectors.get_column_count() == data.get_column_count());

        INFO("check if means shape is expected")
        REQUIRE(means.get_row_count() == 1);
        REQUIRE(means.get_column_count() == data.get_column_count());

        INFO("check if variances shape is expected")
        REQUIRE(variances.get_row_count() == 1);
        REQUIRE(variances.get_column_count() == data.get_column_count());
    }

    void check_nans(const pca::train_result<>& result) {
        const auto [means, variances, eigenvalues, eigenvectors] = unpack_result(result);

        INFO("check if there is no NaN in eigenvalues")
        REQUIRE(te::has_no_nans(eigenvalues));

        INFO("check if there is no NaN in eigenvectors")
        REQUIRE(te::has_no_nans(eigenvectors));

        INFO("check if there is no NaN in means")
        REQUIRE(te::has_no_nans(means));

        INFO("check if there is no NaN in variances")
        REQUIRE(te::has_no_nans(variances));
    }

    void check_eigenvalues_order(const table& eigenvalues) const {
        const auto W = la::matrix<double>::wrap(eigenvalues);
        bool is_descending = true;
        la::enumerate_linear(W, [&](std::int64_t i, double) {
            if (i > 0) {
                CAPTURE(i, W.get(i - 1), W.get(i));
                is_descending = is_descending && (W.get(i - 1) >= W.get(i));
            }
        });
        CHECK(is_descending);
    }

    void check_eigenvectors_orthogonality(const table& eigenvectors) {
        const auto V = la::matrix<double>::wrap(eigenvectors);
        const auto E = la::matrix<double>::eye(V.get_row_count());
        const auto VxVT = la::dot(V, V.t());
        const double diff = la::abs_error(VxVT, E);
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        CHECK(diff < tol);
    }

    void check_means(const te::basic_statistics<double>& reference, const table& means) {
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        const double diff = te::rel_error(reference.get_means(), means, tol);
        CHECK(diff < tol);
    }

    void check_variances(const te::basic_statistics<double>& reference, const table& variances) {
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        const double diff = te::rel_error(reference.get_variances(), variances, tol);
        CHECK(diff < tol);
    }

    void check_eigenvalues(const table& reference, const table& eigenvalues) {
        const auto v_ref = la::matrix<double>::wrap(reference);
        const auto v_actual = la::matrix<double>::wrap(eigenvalues);
        const double tol = te::get_tolerance<Float>(1e-4, 1e-8);
        const double diff = te::rel_error(v_ref, v_actual, tol);
        CHECK(diff < tol);
    }

    void check_eigenvectors(const table& reference, const table& eigenvectors) {
        const auto v_ref = la::matrix<double>::wrap(reference);
        const auto v_actual = la::matrix<double>::wrap(eigenvectors);
        const double tol = te::get_tolerance<Float>(1e-3, 1e-6);
        const double diff = te::rel_error(v_ref, v_actual, tol);
        CHECK(diff < tol);
    }

private:
    static auto unpack_result(const pca::train_result<>& result) {
        const auto means = result.get_means();
        const auto variances = result.get_variances();
        const auto eigenvalues = result.get_eigenvalues();
        const auto eigenvectors = result.get_eigenvectors();
        return std::make_tuple(means, variances, eigenvalues, eigenvectors);
    }
};

using pca_types = COMBINE_TYPES((float, double), (pca::method::cov, pca::method::svd));

TEMPLATE_LIST_TEST_M(pca_batch_test, "pca common flow", "[pca][integration][batch]", pca_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_uniform(0.2, 0.5),
                           te::dataframe_builder{ 100000, 10 }.fill_uniform(-0.2, 1.5));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto data_table_id = this->get_homogen_table_id();

    const std::int64_t component_count = GENERATE_COPY(0,
                                                       1,
                                                       data.get_column_count(),
                                                       data.get_column_count() - 1,
                                                       data.get_column_count() / 2);

    this->general_checks(data, component_count, data_table_id);
}

TEMPLATE_LIST_TEST_M(pca_batch_test, "pca on gold data", "[pca][integration][batch]", pca_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t component_count = 0;
    const bool deterministic = true;
    const auto pca_desc = this->get_descriptor(component_count, deterministic);
    const auto gold_data = this->get_gold_data();

    const auto pca_result = te::train(this->get_policy(), pca_desc, gold_data);
    const auto eigenvalues = pca_result.get_eigenvalues();
    const auto eigenvectors = pca_result.get_eigenvectors();

    INFO("check eigenvalues") {
        const auto gold_eigenvalues = this->get_gold_eigenvalues();
        this->check_eigenvalues(gold_eigenvalues, eigenvalues);
    }

    INFO("check eigenvectors") {
        const auto gold_eigenvectors = this->get_gold_eigenvectors();
        this->check_eigenvectors(gold_eigenvectors, eigenvectors);
    }
}

TEMPLATE_LIST_TEST_M(pca_batch_test,
                     "pca common flow higgs",
                     "[external-dataset][pca][integration][batch]",
                     pca_types) {
    SKIP_IF(this->not_available_on_device());
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t component_count = 0;
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ "workloads/higgs/dataset/higgs_100t_train.csv" });

    const auto data_table_id = this->get_homogen_table_id();

    this->general_checks(data, component_count, data_table_id);
}

} // namespace oneapi::dal::pca::test
