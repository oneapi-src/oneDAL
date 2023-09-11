/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <type_traits>

#include "oneapi/dal/backend/primitives/blas/syrk.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <ndorder order>
struct order_tag {
    static constexpr auto value = order;
};

using c_order = order_tag<ndorder::c>;
using f_order = order_tag<ndorder::f>;

template <mkl::uplo uplo>
struct uplo_tag {
    static constexpr auto value = uplo;
};

using upper = uplo_tag<mkl::uplo::upper>;
using lower = uplo_tag<mkl::uplo::lower>;

template <typename Param>
class syrk_test : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    static constexpr auto co = ndorder::c;
    using float_t = std::tuple_element_t<0, Param>;
    static constexpr auto ao = std::tuple_element_t<1, Param>::value;
    static constexpr auto ul = std::tuple_element_t<2, Param>::value;

    void generate_small_dimensions() {
        n_ = GENERATE(4, 5, 6);
        k_ = GENERATE(3, 7, 9);
        CAPTURE(n_, k_);
    }

    void generate_medium_dimensions() {
        n_ = GENERATE(400, 500);
        k_ = GENERATE(500, 600);
        CAPTURE(n_, k_);
    }

    auto A() {
        check_if_initialized();
        return ndarray<float_t, 2, ao>::ones(this->get_queue(), { n_, k_ });
    }

    auto B() {
        check_if_initialized();
        return ndarray<float_t, 2, ao>::ones(this->get_queue(), { k_, n_ });
    }

    auto C() {
        check_if_initialized();
        return ndarray<float_t, 2, co>::zeros(this->get_queue(), { n_, n_ });
    }

    void test_asyrk() {
        auto [c, c_e] = C();
        auto [a, a_e] = A();

        auto& q = this->get_queue();

        SECTION("At x At") {
            const auto b = a.t();
            REQUIRE(b.get_shape() == ndshape<2>{ k_, n_ });
            syrk<ul>(q, b, c, { a_e, c_e }).wait_and_throw();
            check_ones_matrix(c);
        }
    }

    void test_bsyrk() {
        auto [c, c_e] = C();
        auto [b, b_e] = B();

        auto& q = this->get_queue();

        SECTION("B x B") {
            REQUIRE(b.get_shape() == ndshape<2>{ k_, n_ });
            syrk<ul>(q, b, c, { b_e, c_e }).wait_and_throw();
            check_ones_matrix(c);
        }
    }

    void check_ones_matrix(const ndview<float_t, 2, co>& mat, float_t tol = 1e-7) {
        check_if_initialized();
        REQUIRE(mat.get_shape() == ndshape<2>{ n_, n_ });

        constexpr bool is_upper = (ul == mkl::uplo::upper);
        constexpr bool is_lower = (ul == mkl::uplo::lower);

        for (std::int64_t r = 0; r < n_; ++r) {
            for (std::int64_t c = r; is_lower && c < n_; ++c) {
                const auto gtr = float_t(k_);
                const auto val = mat.at(r, c);
                const auto err = std::abs(val - gtr) / gtr;
                if (err > tol) {
                    CAPTURE(r, c, val, gtr, err);
                    CAPTURE(tol, ao, co, ul);
                    FAIL();
                }
            }
            for (std::int64_t c = 0; is_upper && c <= r; ++c) {
                const auto gtr = float_t(k_);
                const auto val = mat.at(r, c);
                const auto err = std::abs(val - gtr) / gtr;
                if (err > tol) {
                    CAPTURE(r, c, val, gtr, err);
                    CAPTURE(tol, ao, co, ul);
                    FAIL();
                }
            }
        }

        SUCCEED();
    }

    bool is_initialized() const {
        return n_ > 0 && k_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "syrk test is not initialized" };
        }
    }

private:
    std::int64_t n_ = 0;
    std::int64_t k_ = 0;
};

using syrk_types = COMBINE_TYPES((float, double), (c_order, f_order), (upper, lower));

TEMPLATE_LIST_TEST_M(syrk_test,
                     "ones matrix AxA syrk"
                     "on small sizes",
                     "[syrk][small]",
                     syrk_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->generate_small_dimensions();
    this->test_asyrk();
}

TEMPLATE_LIST_TEST_M(syrk_test,
                     "ones matrix BxB syrk"
                     "on small sizes",
                     "[syrk][small]",
                     syrk_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->generate_small_dimensions();
    this->test_bsyrk();
}

TEMPLATE_LIST_TEST_M(syrk_test,
                     "ones matrix AxA"
                     "syrk on medium sizes",
                     "[syrk][medium]",
                     syrk_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->generate_medium_dimensions();
    this->test_asyrk();
}

TEMPLATE_LIST_TEST_M(syrk_test,
                     "ones matrix BxB"
                     "syrk on medium sizes",
                     "[syrk][medium]",
                     syrk_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->generate_medium_dimensions();
    this->test_bsyrk();
}

} // namespace oneapi::dal::backend::primitives::test
