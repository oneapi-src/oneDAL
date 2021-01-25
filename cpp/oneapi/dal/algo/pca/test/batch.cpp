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
#include "oneapi/dal/test/engine/linalg.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class pca_batch_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor(std::int64_t component_count) const {
        return pca::descriptor<Float, Method>{}
            .set_component_count(component_count)
            .set_deterministic(false);
    }

    table get_input_data(const te::dataframe& df, const std::string& table_type) {
        return df.get_table<Float>(this->get_policy(), table_type);
    }

    void general_checks(const te::dataframe& data,
                        std::int64_t component_count,
                        const std::string& table_type) {
        CAPTURE(component_count, table_type);
        const table x = get_input_data(data, table_type);

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

        SECTION("eigenvectors order is descending") {
            this->check_eigenvalues_order(eigenvalues);
        }

        SECTION("eigenvectors matrix is orthogonal") {
            check_eigenvectors_orthogonality(eigenvectors);
        }

        const auto bs = te::compute_basic_statistics<double>(data);

        SECTION("means are expected") {
            check_means(bs, means);
        }

        SECTION("variances are expected") {
            check_variances(bs, variances);
        }
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

        SECTION("eigenvalues shape is expected") {
            REQUIRE(eigenvalues.get_row_count() == 1);
            REQUIRE(eigenvalues.get_column_count() == expected_component_count);
        }

        SECTION("eigenvectors shape is expected") {
            REQUIRE(eigenvectors.get_row_count() == expected_component_count);
            REQUIRE(eigenvectors.get_column_count() == data.get_column_count());
        }

        SECTION("means shape is expected") {
            REQUIRE(means.get_row_count() == 1);
            REQUIRE(means.get_column_count() == data.get_column_count());
        }

        SECTION("variances shape is expected") {
            REQUIRE(variances.get_row_count() == 1);
            REQUIRE(variances.get_column_count() == data.get_column_count());
        }
    }

    void check_nans(const pca::train_result<>& result) {
        const auto [means, variances, eigenvalues, eigenvectors] = unpack_result(result);

        SECTION("there is no NaN in eigenvalues") {
            REQUIRE(te::has_no_nans(eigenvalues));
        }

        SECTION("there is no NaN in eigenvectors") {
            REQUIRE(te::has_no_nans(eigenvectors));
        }

        SECTION("there is no NaN in means") {
            REQUIRE(te::has_no_nans(means));
        }

        SECTION("there is no NaN in variances") {
            REQUIRE(te::has_no_nans(variances));
        }
    }

    void check_eigenvalues_order(const table& eigenvalues) const {
        const auto W = la::matrix<double>::wrap(eigenvalues);
        bool is_descinding = true;
        la::enumerate_linear(W, [&](std::int64_t i, double) {
            if (i > 0) {
                CAPTURE(i, W.get(i - 1), W.get(i));
                is_descinding = is_descinding && (W.get(i - 1) >= W.get(i));
            }
        });
        CHECK(is_descinding);
    }

    void check_eigenvectors_orthogonality(const table& eigenvectors) {
        const auto V = la::matrix<double>::wrap(eigenvectors);
        const auto E = la::matrix<double>::eye(V.get_row_count());
        const auto VxVT = la::dot(V, V.t());
        const double diff = la::l_inf_norm(VxVT, E);
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        CHECK(diff < tol);
    }

    void check_means(const te::basic_statistics<double>& reference, const table& means) {
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        const double diff = te::l_inf_norm(reference.get_means(), means);
        CHECK(diff < tol);
    }

    void check_variances(const te::basic_statistics<double>& reference, const table& variances) {
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        const double diff = te::l_inf_norm(reference.get_variances(), variances);
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
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 10 }.fill_uniform(0.2, 0.5),
                           te::dataframe_builder{ 100000, 10 }.fill_uniform(-0.2, 1.5));

    const std::int64_t component_count = GENERATE_COPY(0,
                                                       1,
                                                       data.get_column_count(),
                                                       data.get_column_count() - 1,
                                                       data.get_column_count() / 2);

    const std::string table_type = GENERATE("homogen");

    this->general_checks(data, component_count, table_type);
}

} // namespace oneapi::dal::pca::test
