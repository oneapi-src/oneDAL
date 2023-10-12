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

#include "oneapi/dal/backend/primitives/blas/gemm.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <ndorder order>
struct order_tag {
    static constexpr ndorder value = order;
};

using c_order = order_tag<ndorder::c>;
using f_order = order_tag<ndorder::f>;

template <typename Param>
class gemm_test : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    static constexpr ndorder ao = std::tuple_element_t<1, Param>::value;
    static constexpr ndorder bo = std::tuple_element_t<2, Param>::value;
    static constexpr ndorder co = std::tuple_element_t<3, Param>::value;

    gemm_test() {
        m_ = 0;
        n_ = 0;
        k_ = 0;
    }

    void generate_small_dimensions() {
        m_ = GENERATE(3, 4, 5);
        n_ = GENERATE(4, 5, 6);
        k_ = GENERATE(5, 6, 7);
        CAPTURE(m_, n_, k_);
    }

    void generate_medium_dimensions() {
        m_ = GENERATE(300, 400);
        n_ = GENERATE(400, 500);
        k_ = GENERATE(500, 600);
        CAPTURE(m_, n_, k_);
    }

    auto A() {
        check_if_initialized();
        return ndarray<float_t, 2, ao>::ones(this->get_queue(), { m_, k_ });
    }

    auto At() {
        check_if_initialized();
        return ndarray<float_t, 2, ao>::ones(this->get_queue(), { k_, m_ });
    }

    auto B() {
        check_if_initialized();
        return ndarray<float_t, 2, bo>::ones(this->get_queue(), { k_, n_ });
    }

    auto Bt() {
        check_if_initialized();
        return ndarray<float_t, 2, bo>::ones(this->get_queue(), { n_, k_ });
    }

    auto C() {
        check_if_initialized();
        return ndarray<float_t, 2, co>::empty(this->get_queue(), { m_, n_ });
    }

    void test_onedal_gemm() {
        auto c = C();
        auto [a, a_e] = A();
        auto [b, b_e] = B();
        auto [at, at_e] = At();
        auto [bt, bt_e] = Bt();
        //sycl::event axb_event;
        //sycl::event axbt_event;
        SECTION("A x B") {
            gemm(this->get_queue(), a, b, c, { a_e, b_e }).wait_and_throw();
            check_ones_matrix(c);
        }

        SECTION("A x Bt") {
            gemm(this->get_queue(), a, bt.t(), c, { a_e, bt_e }).wait_and_throw();
            check_ones_matrix(c);
        }

        SECTION("At x B") {
            gemm(this->get_queue(), at.t(), b, c, { at_e, b_e }).wait_and_throw();
            check_ones_matrix(c);
        }

        SECTION("At x Bt") {
            gemm(this->get_queue(), at.t(), bt.t(), c, { at_e, bt_e }).wait_and_throw();
            check_ones_matrix(c);
        }
    }

    void test_mkl_gemm() {
        auto c = C();
        auto [a, a_e] = A();
        auto [b, b_e] = B();
        auto [at, at_e] = At();
        auto [bt, bt_e] = Bt();
        constexpr bool is_c_trans = (c.get_order() == ndorder::c);
        sycl::event axb_event, axbt_event;
        SECTION("A x B") {
            if constexpr (is_c_trans) {
                axb_event = mkl::blas::gemm(this->get_queue(),
                                            f_order_as_transposed(b.get_order()),
                                            f_order_as_transposed(a.get_order()),
                                            c.get_dimension(1),
                                            c.get_dimension(0),
                                            a.get_dimension(1),
                                            1,
                                            b.get_data(),
                                            b.get_leading_stride(),
                                            a.get_data(),
                                            a.get_leading_stride(),
                                            0,
                                            c.get_mutable_data(),
                                            c.get_leading_stride(),
                                            { a_e, b_e });
            }
            else {
                axb_event = mkl::blas::gemm(this->get_queue(),
                                            c_order_as_transposed(a.get_order()),
                                            c_order_as_transposed(b.get_order()),
                                            c.get_dimension(0),
                                            c.get_dimension(1),
                                            a.get_dimension(1),
                                            1,
                                            a.get_data(),
                                            a.get_leading_stride(),
                                            b.get_data(),
                                            b.get_leading_stride(),
                                            0,
                                            c.get_mutable_data(),
                                            c.get_leading_stride(),
                                            { a_e, b_e });
            }
            check_ones_matrix(c);
        }
        //TODO::create a ticket for MKL with this reproducer
        SECTION("A x Bt") {
            if constexpr (is_c_trans) {
                axbt_event = mkl::blas::gemm(this->get_queue(),
                                             f_order_as_transposed(bt.t().get_order()),
                                             f_order_as_transposed(a.get_order()),
                                             c.get_dimension(1),
                                             c.get_dimension(0),
                                             a.get_dimension(1),
                                             1,
                                             bt.get_data(),
                                             bt.get_leading_stride(),
                                             a.get_data(),
                                             a.get_leading_stride(),
                                             0,
                                             c.get_mutable_data(),
                                             c.get_leading_stride(),
                                             { a_e, bt_e, axb_event });
            }
            else {
                axbt_event = mkl::blas::gemm(this->get_queue(),
                                             c_order_as_transposed(a.get_order()),
                                             c_order_as_transposed(bt.t().get_order()),
                                             c.get_dimension(0),
                                             c.get_dimension(1),
                                             a.get_dimension(1),
                                             1,
                                             a.get_data(),
                                             a.get_leading_stride(),
                                             bt.get_data(),
                                             bt.get_leading_stride(),
                                             0,
                                             c.get_mutable_data(),
                                             c.get_leading_stride(),
                                             { a_e, bt_e, axb_event });
            }
            check_ones_matrix(c);
        }

        // SECTION("At x B") {
        //     gemm_event gemm(this->get_queue(), at.t(), b, c, { at_e, b_e }, {gemm_event});
        //     check_ones_matrix(c);
        // }

        // SECTION("At x Bt") {
        //     gemm_event = gemm(this->get_queue(), at.t(), bt.t(), c, { at_e, bt_e }, {gemm_event});
        //     check_ones_matrix(c);
        // }
    }

    void check_ones_matrix(const ndarray<float_t, 2, co>& mat) {
        check_if_initialized();
        REQUIRE(mat.get_shape() == ndshape<2>{ m_, n_ });

        const float_t* mat_ptr = mat.get_data();
        for (std::int64_t i = 0; i < mat.get_count(); i++) {
            if (std::int64_t(mat_ptr[i]) != k_) {
                CAPTURE(i, mat_ptr[i]);
                FAIL();
            }
        }
        SUCCEED();
    }

    bool is_initialized() const {
        return m_ > 0 && n_ > 0 && k_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "gemm test is not initialized" };
        }
    }

private:
    std::int64_t m_;
    std::int64_t n_;
    std::int64_t k_;
};

using gemm_types = COMBINE_TYPES((float, double),
                                 (c_order, f_order),
                                 (c_order, f_order),
                                 (c_order, f_order));

TEMPLATE_LIST_TEST_M(gemm_test, "ones matrix gemm on small sizes", "[gemm][small]", gemm_types) {
    // DPC++ GEMM from micro MKL libs is not supported on CPU
    SKIP_IF(this->get_policy().is_cpu());

    // Test takes too long time if HW emulates float64
    SKIP_IF(this->not_float64_friendly());

    this->generate_small_dimensions();
    this->test_onedal_gemm();
}

TEMPLATE_LIST_TEST_M(gemm_test,
                     "ones matrix mkl gemm on small sizes",
                     "[gemm][small]",
                     gemm_types) {
    // DPC++ GEMM from micro MKL libs is not supported on CPU
    SKIP_IF(this->get_policy().is_cpu());

    // Test takes too long time if HW emulates float64
    SKIP_IF(this->not_float64_friendly());

    this->generate_small_dimensions();
    this->test_mkl_gemm();
}

// TEMPLATE_LIST_TEST_M(gemm_test, "ones matrix gemm on medium sizes", "[gemm][medium]", gemm_types) {
//     // DPC++ GEMM from micro MKL libs is not supported on CPU
//     SKIP_IF(this->get_policy().is_cpu());

//     // Test takes too long time if HW emulates float64
//     SKIP_IF(this->not_float64_friendly());

//     this->generate_medium_dimensions();
//     this->test_onedal_gemm();
// }

} // namespace oneapi::dal::backend::primitives::test
