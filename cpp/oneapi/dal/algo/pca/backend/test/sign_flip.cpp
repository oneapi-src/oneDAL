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

#include "oneapi/dal/algo/pca/backend/sign_flip.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::pca::backend::test {

namespace pr = dal::backend::primitives;

template <typename Float>
class sign_flip_test {
public:
    pr::ndarray<Float, 2> get_negative_data() const {
        const std::int64_t row_count = 3;
        const std::int64_t column_count = 2;
        const Float data[] = { -1.3, -1.5, //
                               -5.1, -4.2, //
                               -4.5, -8.1 };
        return pr::ndarray<Float, 2>::copy(data, { row_count, column_count });
    }

    pr::ndarray<Float, 2> get_positive_data() const {
        const std::int64_t row_count = 3;
        const std::int64_t column_count = 2;
        const Float data[] = { 5.3, 3.5, //
                               4.1, 6.2, //
                               1.5, 9.1 };
        return pr::ndarray<Float, 2>::copy(data, { row_count, column_count });
    }

    void check_if_flipped_data_positive(const pr::ndview<Float, 2>& origin,
                                        const pr::ndview<Float, 2>& result) {
        REQUIRE(origin.get_shape() == result.get_shape());

        const Float* origin_ptr = origin.get_data();
        const Float* result_ptr = result.get_data();

        for (std::int64_t i = 0; i < origin.get_count(); i++) {
            REQUIRE(result_ptr[i] > 0);
            REQUIRE(std::abs(result_ptr[i]) == std::abs(origin_ptr[i]));
        }
    }
};

TEMPLATE_TEST_M(sign_flip_test, "flips if all negative", "[negative]", float, double) {
    auto data = this->get_negative_data();

    sign_flip(data);

    this->check_if_flipped_data_positive(this->get_negative_data(), data);
}

TEMPLATE_TEST_M(sign_flip_test, "does not flips if all positive", "[positive]", float, double) {
    auto data = this->get_positive_data();

    sign_flip(data);

    this->check_if_flipped_data_positive(this->get_positive_data(), data);
}

} // namespace oneapi::dal::pca::backend::test
