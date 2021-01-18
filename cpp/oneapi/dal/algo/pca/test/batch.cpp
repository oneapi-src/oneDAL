/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include "oneapi/dal/backend/linalg.hpp"
#include "oneapi/dal/test/engine/dataframes.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;
namespace la = backend::linalg;

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

    table get_table(const te::dataframe& df, const std::string& table_type) {
        return df.get_table<Float>(this->get_policy(), table_type);
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
        const double diff = la::max(la::abs(la::difference(VxVT, E)));
        const double tol = te::get_tolerance<Float>(1e-4, 1e-10);
        CHECK(diff < tol);
    }
};

using pca_types = COMBINE_TYPES((float, double), (pca::method::cov, pca::method::svd));

TEMPLATE_LIST_TEST_M(pca_batch_test, "common flow",
                     "[pca][integration][batch]", pca_types) {
    const te::dataframe data =
        GENERATE_DATAFRAME(te::dataframe_builder{1000, 100}.fill_uniform(-2, 5),
                           te::dataframe_builder{100, 1000}.fill_uniform(-0.2, 1.5));

    const std::int64_t component_count =
        GENERATE_COPY(0, 1, data.get_column_count(),
                      data.get_column_count() - 1,
                      data.get_column_count() / 2);

    const std::string table_type = GENERATE("homogen");

    SECTION("training") {
        const table x = this->get_table(data, table_type);
        const auto pca_desc = this->get_descriptor(component_count);

        CAPTURE(component_count, table_type);
        const auto result = this->train(pca_desc, x);
        const auto eigenvalues = result.get_eigenvalues();
        const auto eigenvectors = result.get_eigenvectors();

        const std::int64_t expected_component_count =
            (component_count > 0) ? component_count : x.get_column_count();

        SECTION("eigenvalues shape is expected") {
            CHECK(eigenvalues.get_row_count() == 1);
            CHECK(eigenvalues.get_column_count() == expected_component_count);
        }

        SECTION("eigenvectors shape is expected") {
            CHECK(eigenvectors.get_row_count() == expected_component_count);
            CHECK(eigenvectors.get_column_count() == x.get_column_count());
        }

        SECTION("there is no NaN in eigenvalues") {}
        SECTION("there is no NaN in eigenvectors") {}

        SECTION("eigenvectors order is descending") {
            this->check_eigenvalues_order(eigenvalues);
        }

        SECTION("eigenvectors matrix is orthogonal") {
            this->check_eigenvectors_orthogonality(eigenvectors);
        }
    }
}

} // namespace oneapi::dal::pca::test
