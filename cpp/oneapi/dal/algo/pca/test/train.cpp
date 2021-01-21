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

#include "oneapi/dal/test/datasets.hpp"
#include "oneapi/dal/algo/pca/train.hpp"
#include "oneapi/dal/backend/linalg.hpp"

namespace oneapi::dal::pca::test {

namespace la = backend::linalg;

ALGO_TEST_CASE("PCA", (float, double), (method::cov)) {
    DECLARE_TEST_POLICY(policy);

    const dal::test::dataset data =
        GENERATE_DATASET(dal::test::random_dataset(1000, 100).uniform(-2, 5),
                         dal::test::random_dataset(100, 10).uniform(-0.2, 1.5));
    const std::string table_type = GENERATE("homogen");
    const std::int64_t component_count = GENERATE(2, 3, 5);

    SECTION("training") {
        const table x = data.get_table<Float>(policy, table_type);

        const auto pca_desc = descriptor<Float, Method>{}
                                  .set_component_count(component_count)
                                  .set_deterministic(false);

        const auto result = dal::test::train(policy, pca_desc, x);
        const auto eigenvalues = result.get_eigenvalues();
        const auto eigenvectors = result.get_eigenvectors();

        SECTION("eigenvalues shape is expected") {
            CHECK(eigenvalues.get_row_count() == 1);
            CHECK(eigenvalues.get_column_count() == component_count);
        }

        SECTION("eigenvectors shape is expected") {
            CHECK(eigenvectors.get_row_count() == component_count);
            CHECK(eigenvectors.get_column_count() == x.get_column_count());
        }

        SECTION("there is no NaN in eigenvalues") {}

        SECTION("eigenvectors order is descending") {
            const auto W = la::matrix<double>::wrap(eigenvalues);
            W.enumerate_linear([&](std::int64_t j, double w) {
                if (j > 0) {
                    CHECK(W.get(0, j - 1) > W.get(0, j));
                }
            });
        }

        SECTION("eigenvectors matrix is orthogonal") {
            const auto V = la::matrix<double>::wrap(eigenvectors);
            const auto E = la::matrix<double>::eye(V.get_row_count());
            const auto VxVT = la::dot(V, V.T());
            const double diff = (VxVT - E).abs().max();
            CHECK(diff < dal::test::get_tolerance<Float>(1e-10, 1e-4));
        }
    }
}

} // namespace oneapi::dal::pca::test
