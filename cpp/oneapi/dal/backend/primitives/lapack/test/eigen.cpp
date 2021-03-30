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

#include "oneapi/dal/backend/primitives/lapack/eigen.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename Float>
static la::matrix<Float> generate_symmetric_positive(std::int64_t dim, int seed) {
    const auto a = la::generate_uniform<Float>({ dim, dim }, -1.0, 1.0, seed);
    const auto at = la::transpose(a);
    const auto c = la::multiply(Float(0.5), la::add(a, at));
    return la::add(c, la::matrix<Float>::diag(dim, dim));
}

template <typename Float>
static auto eigval(const la::matrix<Float>& symmetric_matrix) {
    ONEDAL_ASSERT(symmetric_matrix.get_row_count() == symmetric_matrix.get_column_count());

    const std::int64_t dim = symmetric_matrix.get_row_count();
    const auto s_copy_flat = symmetric_matrix.copy().get_array();

    auto data_or_eigenvectors_nd = ndarray<Float, 2>::wrap_mutable(s_copy_flat, { dim, dim });
    auto eigenvalues_nd = ndarray<Float, 1>::empty(dim);
    sym_eigval(data_or_eigenvectors_nd, eigenvalues_nd);

    const auto eigenvectors = la::matrix<Float>::wrap_nd(data_or_eigenvectors_nd);
    const auto eigenvalues = la::matrix<Float>::wrap_nd(eigenvalues_nd);
    return std::make_tuple(eigenvectors, eigenvalues);
}

#define SYM_EIGVAL_TEST(name) TEMPLATE_TEST(name, "[sym_eigval]", float, double)

SYM_EIGVAL_TEST("check eigenvectors on symmetric positive-definite matrix") {
    using float_t = TestType;
    const std::int64_t dim = GENERATE(3, 28, 125, 256);
    const auto s = generate_symmetric_positive<float_t>(dim, 7777);

    const auto [eigenvectors, eigenvalues] = eigval(s);

    INFO("check eigenvectors") {
        const auto s_f64 = la::astype<double>(s);
        const auto eigenvalues_f64 = la::astype<double>(eigenvalues);
        const auto eigenvectors_f64 = la::astype<double>(eigenvectors);

        for (std::int64_t i = 0; i < dim; i++) {
            const auto v = la::transpose(eigenvectors_f64.get_row(i));
            const double w = eigenvalues_f64.get(i);
            CAPTURE(i, w);

            const double tol = te::get_tolerance<float>(1e-4, 1e-10) * w;

            // Check condition: $S \times v_i = w_i \dot v_i$
            const double err = la::rel_error(la::dot(s_f64, v), la::multiply(w, v), tol);
            REQUIRE(err < tol);
        }
    }
}

} // namespace oneapi::dal::backend::primitives::test
