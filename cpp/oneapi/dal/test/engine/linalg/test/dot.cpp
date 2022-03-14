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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/linalg/dot.hpp"
#include "oneapi/dal/test/engine/linalg/loops.hpp"

namespace oneapi::dal::test::engine::linalg::test {

template <layout lyt_a, layout lyt_b, layout lyt_c>
class dot_test {
public:
    dot_test() {
        m_ = GENERATE(3, 4, 5);
        n_ = GENERATE(4, 5, 6);
        k_ = GENERATE(5, 6, 7);
        CAPTURE(m_, n_, k_);
    }

    auto A() const {
        return matrix<float, lyt_a>::ones({ m_, k_ });
    }

    auto At() const {
        return matrix<float, lyt_a>::ones({ k_, m_ }).t();
    }

    auto B() const {
        return matrix<float, lyt_b>::ones({ k_, n_ });
    }

    auto Bt() const {
        return matrix<float, lyt_b>::ones({ n_, k_ }).t();
    }

    auto C() const {
        return matrix<float, lyt_c>::empty({ m_, n_ });
    }

    void check_ones_matrix_dot(const matrix<float, lyt_c>& mat) const {
        REQUIRE(mat.get_shape() == shape{ m_, n_ });
        for_each(mat, [&](float x) {
            REQUIRE(std::int64_t(x) == k_);
        });
    }

private:
    std::int64_t m_;
    std::int64_t n_;
    std::int64_t k_;
};

#define LAYOUT_VEC2(x, y)    (layout::x, layout::y)
#define LAYOUT_VEC3(x, y, z) (layout::x, layout::y, layout::z)

#define LAYOUTS_AB                                                                                 \
    LAYOUT_VEC3(row_major, row_major, row_major), LAYOUT_VEC3(row_major, column_major, row_major), \
        LAYOUT_VEC3(column_major, row_major, row_major),                                           \
        LAYOUT_VEC3(column_major, column_major, row_major)

#define LAYOUTS_ABC                                                                                \
    LAYOUT_VEC3(row_major, row_major, row_major), LAYOUT_VEC3(row_major, row_major, column_major), \
        LAYOUT_VEC3(row_major, column_major, row_major),                                           \
        LAYOUT_VEC3(row_major, column_major, column_major),                                        \
        LAYOUT_VEC3(column_major, row_major, row_major),                                           \
        LAYOUT_VEC3(column_major, row_major, column_major),                                        \
        LAYOUT_VEC3(column_major, column_major, row_major),                                        \
        LAYOUT_VEC3(column_major, column_major, column_major)

#define DOT_TEST_TEMPLATE_SIG ((layout lyt_a, layout lyt_b, layout lyt_c), lyt_a, lyt_b, lyt_c)

#define DOT_TEST(name, tags, layouts) \
    TEMPLATE_SIG_TEST_M(dot_test, name, tags, DOT_TEST_TEMPLATE_SIG, layouts)

DOT_TEST("dot simple", "[linalg][dot]", LAYOUTS_AB) {
    SECTION("A x B") {
        const auto C = dot(this->A(), this->B());
        this->check_ones_matrix_dot(C);
    }

    SECTION("A x Bt") {
        const auto C = dot(this->A(), this->Bt());
        this->check_ones_matrix_dot(C);
    }

    SECTION("At x B") {
        const auto C = dot(this->At(), this->B());
        this->check_ones_matrix_dot(C);
    }

    SECTION("At x Bt") {
        const auto C = dot(this->At(), this->Bt());
        this->check_ones_matrix_dot(C);
    }
}

DOT_TEST("dot in-place", "[linalg][dot]", LAYOUTS_ABC) {
    auto C = this->C();

    SECTION("A x B") {
        dot(this->A(), this->B(), C);
        this->check_ones_matrix_dot(C);
    }

    SECTION("A x Bt") {
        dot(this->A(), this->Bt(), C);
        this->check_ones_matrix_dot(C);
    }

    SECTION("At x B") {
        dot(this->At(), this->B(), C);
        this->check_ones_matrix_dot(C);
    }

    SECTION("At x Bt") {
        dot(this->At(), this->Bt(), C);
        this->check_ones_matrix_dot(C);
    }
}

TEST("dot orthogonal matrix", "[linalg][dot]") {
    const std::int64_t row_count = 5;
    const std::int64_t column_count = 5;
    const std::int64_t element_count = row_count * column_count;
    const double X_ptr[element_count] = {
        0.5728966506,  0.5677902077,  -0.4104886344, 0.0993844187,  0.4135523258,
        -0.4590520326, 0.2834513205,  -0.6214677550, -0.5114715156, -0.2471867686,
        0.0506571111,  -0.3713048334, 0.0645569868,  -0.6484125926, 0.6595150363,
        0.3318135734,  0.4178413295,  0.5362188809,  -0.5381061517, -0.3717787744,
        -0.5902493276, 0.5336768432,  0.3918910789,  0.1360974115,  0.4412410170
    };

    const auto X = matrix<double>::wrap(array<double>::wrap(X_ptr, element_count),
                                        { row_count, column_count });

    const auto C = dot(X, X.t());

    SECTION("result is ones matrix") {
        enumerate(C, [&](std::int64_t i, std::int64_t j, double x) {
            CAPTURE(i, j);

            if (i == j) {
                REQUIRE(std::abs(x - 1.0) < 1e-9);
            }
            else {
                REQUIRE(std::abs(x) < 1e-9);
            }
        });
    }
}

} // namespace oneapi::dal::test::engine::linalg::test
