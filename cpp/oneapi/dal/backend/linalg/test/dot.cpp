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

#include <iostream>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/backend/linalg/dot.hpp"
#include "oneapi/dal/backend/linalg/io.hpp"

namespace oneapi::dal::backend::linalg::test {

using test_param_t = std::tuple<
    std::int64_t, std::int64_t, std::int64_t,
    layout, layout, layout>;

class dot_test : public ::testing::TestWithParam<test_param_t> {
public:
    std::int64_t get_m() const {
        return  std::get<0>(GetParam());
    }

    std::int64_t get_n() const {
        return  std::get<1>(GetParam());
    }

    std::int64_t get_k() const {
        return  std::get<2>(GetParam());
    }

    layout get_A_layout() const {
        return std::get<3>(GetParam());
    }

    layout get_B_layout() const {
        return std::get<4>(GetParam());
    }

    layout get_C_layout() const {
        return std::get<5>(GetParam());
    }

    matrix<float> get_A() const {
        return matrix<float>::ones({ get_m(), get_k() }, get_A_layout());
    }

    matrix<float> get_At() const {
        return matrix<float>::ones({ get_k(), get_m() }, get_A_layout()).t();
    }

    matrix<float> get_B() const {
        return matrix<float>::ones({ get_k(), get_n() }, get_B_layout());
    }

    matrix<float> get_Bt() const {
        return matrix<float>::ones({ get_n(), get_k() }, get_B_layout()).t();
    }

    matrix<float> get_C() const {
        return matrix<float>::empty({ get_m(), get_n() }, get_C_layout());
    }

    void check_ones_matrix_dot(const matrix<float>& c) const {
        ASSERT_TRUE(c.get_shape() == shape(get_m(), get_n()));
        const float* c_ptr = c.get_data();

        for (std::int64_t i = 0; i < c.get_count(); i++) {
            const float x = c_ptr[i];
            ASSERT_TRUE(std::int64_t(x) == get_k());
        }
    }

};

TEST_P(dot_test, check_simple_AxB) {
    const auto C = dot(get_A(), get_B());
    check_ones_matrix_dot(C);
}

TEST_P(dot_test, check_simple_AtxB) {
    const auto C = dot(get_At(), get_B());
    check_ones_matrix_dot(C);
}

TEST_P(dot_test, check_simple_AxBt) {
    const auto C = dot(get_A(), get_Bt());
    check_ones_matrix_dot(C);
}

TEST_P(dot_test, check_simple_AtxBt) {
    const auto C = dot(get_At(), get_Bt());
    check_ones_matrix_dot(C);
}

TEST_P(dot_test, check_inplace_AxB) {
    auto C = get_C();
    dot(get_A(), get_B(), C);
    check_ones_matrix_dot(C);
}

TEST_P(dot_test, check_inplace_AtxB) {
    auto C = get_C();
    dot(get_At(), get_B(), C);
    check_ones_matrix_dot(C);
}

TEST_P(dot_test, check_inplace_AxBt) {
    auto C = get_C();
    dot(get_A(), get_Bt(), C);
    check_ones_matrix_dot(C);
}

TEST_P(dot_test, check_inplace_AtxBt) {
    auto C = get_C();
    dot(get_At(), get_Bt(), C);
    check_ones_matrix_dot(C);
}

INSTANTIATE_TEST_SUITE_P(
    dot_params,
    dot_test,
    ::testing::Combine(
        ::testing::Values(3, 4, 5),
        ::testing::Values(4, 5, 6),
        ::testing::Values(5, 6, 7),
        ::testing::Values(layout::row_major, layout::column_major),
        ::testing::Values(layout::row_major, layout::column_major),
        ::testing::Values(layout::row_major, layout::column_major)
    )
);

} // namespace oneapi::dal::backend::linalg::test
