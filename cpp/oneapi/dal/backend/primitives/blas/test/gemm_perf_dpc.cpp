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

    void generate_specific_dimensions() {
        beta_ = true;
        k_ = GENERATE(28, 768);
        m_ = GENERATE(1024, 4096);
        n_ = GENERATE(1024, 4096);
        CAPTURE(m_, n_, k_);
    }

    void generate_medium_dimensions() {
        beta_ = GENERATE(0, 1);
        k_ = GENERATE(128, 1024);
        m_ = GENERATE(256, 4096);
        n_ = GENERATE(256, 4096);
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
        return ndarray<float_t, 2, co>::ones(this->get_queue(), { m_, n_ });
    }

    auto fpt_desc() const {
        constexpr auto is_f32 = std::is_same_v<float_t, float>;
        constexpr auto is_f64 = std::is_same_v<float_t, double>;
        static_assert(is_f32 || is_f64);
        if constexpr (is_f32)
            return "float";
        else
            return "double";
    }

    auto order_desc(ndorder ord) const {
        const auto is_c_order = ord == ndorder::c;
        const auto is_f_order = ord == ndorder::f;
        ONEDAL_ASSERT(is_c_order || is_f_order);
        if (is_c_order)
            return "C-order";
        else
            return "F-order";
    }

    auto layout_desc() const {
        return fmt::format("A matrix layout: {}, B matrix layout: {}, C matix layout: {}",
                           order_desc(ao),
                           order_desc(bo),
                           order_desc(co));
    }

    auto type_desc() const {
        return fmt::format("Floating Point Type: {}", fpt_desc());
    }

    auto size_desc() const {
        return fmt::format("M: {}, N: {}, K: {}, Beta: {}",
                           std::to_string(m_),
                           std::to_string(n_),
                           std::to_string(k_),
                           std::to_string(beta_));
    }

    template <typename Arg>
    auto full_desc(const Arg& inp) const {
        auto fmt_str = std::string(inp) + std::string(" : {}, {}, {}");
        return fmt::format(fmt_str, type_desc(), size_desc(), layout_desc());
    }

    void test_gemm() {
        auto c_pair = C();
        auto a_pair = A();
        auto b_pair = B();
        auto at_pair = At();
        auto bt_pair = Bt();

        sycl::event::wait_and_throw({ std::get<1>(a_pair),
                                      std::get<1>(b_pair),
                                      std::get<1>(c_pair),
                                      std::get<1>(at_pair),
                                      std::get<1>(bt_pair) });

        std::string axb_name = this->full_desc("A x B");
        BENCHMARK(axb_name.c_str()) {
            gemm(this->get_queue(),
                 std::get<0>(a_pair),
                 std::get<0>(b_pair),
                 std::get<0>(c_pair),
                 float_t(1.0),
                 float_t(beta_))
                .wait_and_throw();
        };

        std::string axbt_name = this->full_desc("A x Bt");
        BENCHMARK(axbt_name.c_str()) {
            gemm(this->get_queue(),
                 std::get<0>(a_pair),
                 std::get<0>(bt_pair).t(),
                 std::get<0>(c_pair),
                 float_t(1.0),
                 float_t(beta_))
                .wait_and_throw();
        };

        std::string atxb_name = this->full_desc("At x B");
        BENCHMARK(atxb_name.c_str()) {
            gemm(this->get_queue(),
                 std::get<0>(at_pair).t(),
                 std::get<0>(b_pair),
                 std::get<0>(c_pair),
                 float_t(1.0),
                 float_t(beta_))
                .wait_and_throw();
        };

        std::string atxbt_name = this->full_desc("At x Bt");
        BENCHMARK(atxbt_name.c_str()) {
            gemm(this->get_queue(),
                 std::get<0>(at_pair).t(),
                 std::get<0>(bt_pair).t(),
                 std::get<0>(c_pair),
                 float_t(1.0),
                 float_t(beta_))
                .wait_and_throw();
        };
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
    bool beta_;
    std::int64_t m_;
    std::int64_t n_;
    std::int64_t k_;
};

using gemm_types = COMBINE_TYPES((float, double),
                                 (c_order, f_order),
                                 (c_order, f_order),
                                 (c_order, f_order));

TEMPLATE_LIST_TEST_M(gemm_test, "ones matrix gemm on medium sizes", "[gemm][medium]", gemm_types) {
    // DPC++ GEMM from micro MKL libs is not supported on GPU
    SKIP_IF(this->get_policy().is_cpu());

    this->generate_medium_dimensions();
    this->test_gemm();
}

TEMPLATE_LIST_TEST_M(gemm_test,
                     "ones matrix gemm on specific sizes",
                     "[gemm][specific]",
                     gemm_types) {
    // DPC++ GEMM from micro MKL libs is not supported on GPU
    SKIP_IF(this->get_policy().is_cpu());

    this->generate_medium_dimensions();
    this->test_gemm();
}

} // namespace oneapi::dal::backend::primitives::test
