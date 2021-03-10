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

#include <cmath>
#include <type_traits>

#include "oneapi/dal/backend/primitives/blas/gemm.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename Param>
class reduce_test_rm_rw : public te::policy_fixture {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate_small_dimensions() {
        arg    = GENERATE(-3., -1.e-3, 0, 1.e-3,   3.);
        width  = GENERATE(  7,    707, 1,   251,    5);
        stride = GENERATE(768,    707, 6,   517,    4);
        height = GENERATE( 17,    999, 1,     5, 1001);
        REQUIRE(width <= stride);
        CAPTURE(arg, width, stride, height);
    }

    auto input() {
        check_if_initialized();
        return ndarray<float_t, 2, ao>::full(get_queue(), { stride, height }, arg);
    }

    auto output() {
        check_if_initialized();
        return ndarray<float_t, 2, ao>::ones(get_queue(), { height }, val);
    }

    float_t val() const {
        if (std::is_same_v<sum, binary_t>) {
            if(std::is_same_v<identity, unary_t>) {
                return width * arg;
            }
            if(std::is_same_v<abs, unary_t>) {
                return width * atd::abs(arg);
            }
            if(std::is_same_v<square, unary_t>) {
                return width * (arg * arg);
            }
        }
        if (std::is_same_v<min, binary_t>) {
            if(std::is_same_v<identity, unary_t>) {
                return arg;
            }
            if(std::is_same_v<abs, unary_t>) {
                return std::abs(arg);
            }
            if(std::is_same_v<square, unary_t>) {
                return (arg * arg);
            }
        }
        if (std::is_same_v<max, binary_t>) {
            if(std::is_same_v<identity, unary_t>) {
                return arg;
            }
            if(std::is_same_v<abs, unary_t>) {
                return std::abs(arg);
            }
            if(std::is_same_v<square, unary_t>) {
                return (arg * arg);
            }
        }
        REQUIRE(false);
        return 0;
    } 

    void test_raw_reduce() {
        auo inp = 
    }


private:
    float_t arg;
    std::int64_t width;
    std::int64_t stride;
    std::int64_t height;
};

using reduction_types = COMBINE_TYPES((float, double),
                                      (sum, min, max),
                                      (identity, abs, square));

TEMPLATE_LIST_TEST_M(gemm_test, "ones matrix gemm on small sizes", "[gemm][small]", gemm_types) {
    // DPC++ GEMM from micro MKL libs is not supported on GPU
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_small_dimensions();
    this->test_gemm();
}

} // namespace oneapi::dal::backend::primitives::test
