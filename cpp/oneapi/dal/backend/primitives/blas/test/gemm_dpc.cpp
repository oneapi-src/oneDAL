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
class gemm_test : public te::policy_fixture {
public:
    using float_t = std::tuple_element_t<0, Param>;
    static constexpr ndorder ao = std::tuple_element_t<1, Param>::value;
    static constexpr ndorder bo = std::tuple_element_t<2, Param>::value;
    static constexpr ndorder co = std::tuple_element_t<3, Param>::value;

    gemm_test() {
        queue_ = get_policy().get_queue();
        m_ = GENERATE(3, 4, 5);
        n_ = GENERATE(4, 5, 6);
        k_ = GENERATE(5, 6, 7);
        CAPTURE(m_, n_, k_);
    }

    sycl::queue& get_queue() {
        return queue_;
    }

    auto A() {
        auto [x, event] = ndarray<float_t, 2, ao>::ones(queue_, { m_, k_ });
        event.wait_and_throw();
        return x;
    }

    auto At() {
        auto [x, event] = ndarray<float_t, 2, ao>::ones(queue_, { k_, m_ });
        event.wait_and_throw();
        return x.t();
    }

    auto B() {
        auto [x, event] = ndarray<float_t, 2, bo>::ones(queue_, { k_, n_ });
        event.wait_and_throw();
        return x;
    }

    auto Bt() {
        auto [x, event] = ndarray<float_t, 2, bo>::ones(queue_, { n_, k_ });
        event.wait_and_throw();
        return x.t();
    }

    auto C() {
        return ndarray<float_t, 2, co>::empty(queue_, { m_, n_ });
    }

    void check_ones_matrix(const ndarray<float_t, 2, co>& mat) {
        REQUIRE(mat.get_shape() == ndshape<2>{ m_, n_ });

        float_t* mat_ptr = mat.get_data();
        for (std::int64_t i = 0; i < mat.get_count(); i++) {
            REQUIRE(std::int64_t(mat_ptr[i]) == k_);
        }
    }

private:
    std::int64_t m_;
    std::int64_t n_;
    std::int64_t k_;
    sycl::queue queue_;
};

using gemm_types = COMBINE_TYPES((float, double),
                                 (c_order, f_order),
                                 (c_order, f_order),
                                 (c_order, f_order));

TEMPLATE_LIST_TEST_M(gemm_test, "ones matrix gemm", "[gemm]", gemm_types) {
    auto C = this->C();

    SECTION("A x B") {
        gemm(this->get_queue(), this->A(), this->B(), C).wait_and_throw();
        this->check_ones_matrix(C);
    }
}

// TEST("dot orthogonal matrix", "[linalg][dot]") {
//     const std::int64_t row_count = 5;
//     const std::int64_t column_count = 5;
//     const std::int64_t element_count = row_count * column_count;
//     const double X_ptr[element_count] = {
//         0.5728966506,  0.5677902077,  -0.4104886344, 0.0993844187,  0.4135523258,
//         -0.4590520326, 0.2834513205,  -0.6214677550, -0.5114715156, -0.2471867686,
//         0.0506571111,  -0.3713048334, 0.0645569868,  -0.6484125926, 0.6595150363,
//         0.3318135734,  0.4178413295,  0.5362188809,  -0.5381061517, -0.3717787744,
//         -0.5902493276, 0.5336768432,  0.3918910789,  0.1360974115,  0.4412410170
//     };

//     const auto X = matrix<double>::wrap(array<double>::wrap(X_ptr, element_count),
//                                         { row_count, column_count });

//     const auto C = dot(X, X.t());

//     SECTION("result is ones matrix") {
//         enumerate(C, [&](std::int64_t i, std::int64_t j, double x) {
//             CAPTURE(i, j);

//             if (i == j) {
//                 REQUIRE(std::abs(x - 1.0) < 1e-9);
//             }
//             else {
//                 REQUIRE(std::abs(x) < 1e-9);
//             }
//         });
//     }
// }

} // namespace oneapi::dal::backend::primitives::test
