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
class sym_eigvals_test {
public:
    std::int64_t generate_dim() const {
        return GENERATE(3, 28, 125, 256);
    }

    la::matrix<Float> generate_symmetric_positive() {
        const std::int64_t dim = this->generate_dim();
        return la::generate_symmetric_positive_matrix<Float>(dim, -1, 1, seed_);
    }

    auto call_sym_eigvals_inplace(const la::matrix<Float>& symmetric_matrix) {
        constexpr bool is_ascending = true;
        return call_sym_eigvals_inplace_generic(symmetric_matrix, is_ascending);
    }

    auto call_sym_eigvals_inplace_descending(const la::matrix<Float>& symmetric_matrix) {
        constexpr bool is_ascending = false;
        return call_sym_eigvals_inplace_generic(symmetric_matrix, is_ascending);
    }

    auto call_sym_eigvals_descending(const la::matrix<Float>& symmetric_matrix,
                                     std::int64_t eigval_count) {
        ONEDAL_ASSERT(symmetric_matrix.get_row_count() == symmetric_matrix.get_column_count());

        const std::int64_t dim = symmetric_matrix.get_row_count();
        const auto s_copy_flat = symmetric_matrix.copy().get_array();

        auto data_or_scratchpad_nd = ndarray<Float, 2>::wrap_mutable(s_copy_flat, { dim, dim });
        auto eigvecs_nd = ndarray<Float, 2>::empty({ eigval_count, dim });
        auto eigvals_nd = ndarray<Float, 1>::empty(eigval_count);
        sym_eigvals_descending(data_or_scratchpad_nd, eigval_count, eigvecs_nd, eigvals_nd);

        const auto eigvecs = la::matrix<Float>::wrap_nd(eigvecs_nd);
        const auto eigvals = la::matrix<Float>::wrap_nd(eigvals_nd);
        return std::make_tuple(eigvecs, eigvals);
    }

    auto call_sym_eigvals_inplace_generic(const la::matrix<Float>& symmetric_matrix,
                                          bool is_ascending) {
        ONEDAL_ASSERT(symmetric_matrix.get_row_count() == symmetric_matrix.get_column_count());

        const std::int64_t dim = symmetric_matrix.get_row_count();
        const auto s_copy_flat = symmetric_matrix.copy().get_array();

        auto data_or_eigenvectors_nd = ndarray<Float, 2>::wrap_mutable(s_copy_flat, { dim, dim });
        auto eigenvalues_nd = ndarray<Float, 1>::empty(dim);
        if (is_ascending) {
            sym_eigvals(data_or_eigenvectors_nd, eigenvalues_nd);
        }
        else {
            sym_eigvals_descending(data_or_eigenvectors_nd, eigenvalues_nd);
        }

        const auto eigenvectors = la::matrix<Float>::wrap_nd(data_or_eigenvectors_nd);
        const auto eigenvalues = la::matrix<Float>::wrap_nd(eigenvalues_nd);
        return std::make_tuple(eigenvectors, eigenvalues);
    }

    void check_eigvals_definition(const la::matrix<Float>& s,
                                  const la::matrix<Float>& eigvecs,
                                  const la::matrix<Float>& eigvals) const {
        INFO("convert results to float64");
        const auto s_f64 = la::astype<double>(s);
        const auto eigvals_f64 = la::astype<double>(eigvals);
        const auto eigvecs_f64 = la::astype<double>(eigvecs);

        INFO("check eigenvectors and eigenvalues definition");
        for (std::int64_t i = 0; i < eigvecs.get_row_count(); i++) {
            const auto v = la::transpose(eigvecs_f64.get_row(i));
            const double w = eigvals_f64.get(i);
            CAPTURE(i, w);

            // Input matrix is positive-definite, so all eigenvalues must be positive
            REQUIRE(w > 0);

            const double tol = te::get_tolerance<float>(1e-4, 1e-10) * w;

            // Check condition: $S \times v_i = w_i \dot v_i$
            const double err = la::rel_error(la::dot(s_f64, v), la::multiply(w, v), tol);
            REQUIRE(err < tol);
        }
    }

    void check_eigvals_are_ascending(const la::matrix<Float>& eigvals) const {
        INFO("check eigenvalues order is ascending");
        la::enumerate_linear(eigvals, [&](std::int64_t i, Float x) {
            if (i > 0) {
                REQUIRE(eigvals.get(i - 1) <= x);
            }
        });
    }

    void check_eigvals_are_descending(const la::matrix<Float>& eigvals) const {
        INFO("check eigenvalues order is descending");
        la::enumerate_linear(eigvals, [&](std::int64_t i, Float x) {
            if (i > 0) {
                REQUIRE(eigvals.get(i - 1) >= x);
            }
        });
    }

private:
    static constexpr int seed_ = 7777;
};

using eigen_types = COMBINE_TYPES((float, double));

#define SYM_EIGVALS_TEST(name) \
    TEMPLATE_LIST_TEST_M(sym_eigvals_test, name, "[sym_eigvals]", eigen_types)

SYM_EIGVALS_TEST("check inplace sym_eigvals on symmetric positive-definite matrix") {
    const auto s = this->generate_symmetric_positive();

    const auto [eigenvectors, eigenvalues] = this->call_sym_eigvals_inplace(s);

    this->check_eigvals_definition(s, eigenvectors, eigenvalues);
    this->check_eigvals_are_ascending(eigenvalues);
}

SYM_EIGVALS_TEST("check inplace sym_eigvals_descending on symmetric positive-definite matrix") {
    const auto s = this->generate_symmetric_positive();

    const auto [eigenvectors, eigenvalues] = this->call_sym_eigvals_inplace_descending(s);

    this->check_eigvals_definition(s, eigenvectors, eigenvalues);
    this->check_eigvals_are_descending(eigenvalues);
}

SYM_EIGVALS_TEST("check sym_eigvals_descending on symmetric positive-definite matrix") {
    const auto s = this->generate_symmetric_positive();
    const std::int64_t eigvals_count = GENERATE_COPY(1, s.get_row_count() / 2, s.get_row_count());

    const auto [eigenvectors, eigenvalues] = this->call_sym_eigvals_descending(s, eigvals_count);

    REQUIRE(eigenvectors.get_row_count() == eigvals_count);
    REQUIRE(eigenvalues.get_count() == eigvals_count);
    this->check_eigvals_definition(s, eigenvectors, eigenvalues);
    this->check_eigvals_are_descending(eigenvalues);
}

} // namespace oneapi::dal::backend::primitives::test
