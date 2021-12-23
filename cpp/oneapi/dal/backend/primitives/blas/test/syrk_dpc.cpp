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
    using float_t = std::tuple_element_t<0, Param>;
    static constexpr auto ao = std::tuple_element_t<1, Param>::value;
    static constexpr auto co = std::tuple_element_t<2, Param>::value;
    static constexpr auto ul = std::tuple_element_t<3, Param>::value;

    void generate_small_dimensions() {
        n_ = GENERATE(4, 5, 6);
        k_ = GENERATE(5, 6, 7);
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

    auto At() {
        check_if_initialized();
        return ndarray<float_t, 2, ao>::ones(this->get_queue(), { n_, k_ });
    }

    auto C() {
        check_if_initialized();
        return ndarray<float_t, 2, co>::empty(this->get_queue(), { n_, n_ });
    }

    void test_syrk() {
        auto c = C();
        auto [a, a_e] = A();
        auto [at, at_e] = At();

        auto& q = this->get_queue();

        SECTION("A x A") {
            syrk<ul>(q, a, c, { a_e }).wait_and_throw();
            check_ones_matrix(c);
        }

        SECTION("At x At") {
            syrk<ul>(q, at.t(), c, { at_e }).wait_and_throw();
            check_ones_matrix(c);
        }
    }

    void check_ones_matrix(const ndview<float_t, 2, co>& mat) {
        check_if_initialized();
        REQUIRE(mat.get_shape() == ndshape<2>{ n_, n_ });
        SUCCEED();
    }

    bool is_initialized() const {
        return  n_ > 0 && k_ > 0;
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

using syrk_types = COMBINE_TYPES((float, double),
                                 (c_order, f_order),
                                 (c_order, f_order),
                                 (upper, lower));

TEMPLATE_LIST_TEST_M(syrk_test, "ones matrix syrk on small sizes", "[syrk][small]", syrk_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->generate_small_dimensions();
    this->test_syrk();
}

TEMPLATE_LIST_TEST_M(syrk_test, "ones matrix syrk on medium sizes", "[syrk][medium]", syrk_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    this->generate_medium_dimensions();
    this->test_syrk();
}

} // namespace oneapi::dal::backend::primitives::test
