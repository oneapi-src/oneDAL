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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

#include "oneapi/dal/backend/primitives/reduction/functors.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

constexpr auto rm_order = ndorder::c;

template <typename Param>
class reduction_rm_rw_test : public te::policy_fixture {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        arg    = GENERATE(-3., -1.e-3,   0, 1.e-3,   3.);
        width  = GENERATE(  7,    707,   1,   251,    5);
        stride = GENERATE(707,    812, 999,  1001, 1024);
        height = GENERATE( 17,    999,   1,     5, 1001);
        REQUIRE(width <= stride);
        CAPTURE(arg, width, stride, height);
    }

    bool is_initialized() const {
        return width > 0 && stride > 0 && height > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    auto input() {
        check_if_initialized();
        return ndarray<float_t, 2, rm_order>::full(get_queue(), { stride, height }, arg);
    }

    auto output() {
        check_if_initialized();
        return ndarray<float_t, 1, rm_order>::zeros(get_queue(), { height });
    }

    float_t val() const {
        if (std::is_same_v<sum<float_t>, binary_t>) {
            if(std::is_same_v<identity<float_t>, unary_t>) {
                return width * arg;
            }
            if(std::is_same_v<abs<float_t>, unary_t>) {
                return width * std::abs(arg);
            }
            if(std::is_same_v<square<float_t>, unary_t>) {
                return width * (arg * arg);
            }
        }
        if (std::is_same_v<min<float_t>, binary_t>) {
            if(std::is_same_v<identity<float_t>, unary_t>) {
                return arg;
            }
            if(std::is_same_v<abs<float_t>, unary_t>) {
                return std::abs(arg);
            }
            if(std::is_same_v<square<float_t>, unary_t>) {
                return (arg * arg);
            }
        }
        if (std::is_same_v<max<float_t>, binary_t>) {
            if(std::is_same_v<identity<float_t>, unary_t>) {
                return arg;
            }
            if(std::is_same_v<abs<float_t>, unary_t>) {
                return std::abs(arg);
            }
            if(std::is_same_v<square<float_t>, unary_t>) {
                return (arg * arg);
            }
        }
        REQUIRE(false);
        return 0;
    } 

    void check_output(ndarray<float_t, 1, rm_order>& outarr, 
                                const float_t tol = 1.e-5) {
        const auto gtv = val();
        const auto arr = outarr.flatten();
        for(auto i = 0; i < height; ++i) {
            const auto diff = arr[i] - gtv;
            if(diff < -tol || tol < diff) {
                CAPTURE(gtv, arr[i], diff, tol);
                FAIL();
            }
        }
    }

    void test_raw_reduce_narrow() {
        using namespace oneapi::dal::backend::primitives;
        using reduction_t = reduction_rm_rw_narrow<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output();

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(get_queue());
        reducer(inp_ptr, 
                out_ptr,
                width,
                height,
                binary_t{},
                unary_t{},
                {inp_event, out_event}).wait_and_throw();    

        check_output(out_array);     
    }

    void test_raw_reduce_wide() {
        using namespace oneapi::dal::backend::primitives;
        using reduction_t = reduction_rm_rw_wide<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output();

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(get_queue());
        reducer(inp_ptr, 
                out_ptr,
                width,
                height,
                binary_t{},
                unary_t{},
                {inp_event, out_event}).wait_and_throw(); 

        check_output(out_array);     
    }


private:
    float_t arg;
    std::int64_t width;
    std::int64_t stride;
    std::int64_t height;
};

using reduction_types = std::tuple< std::tuple<float, sum<float>, square<float>   >,
                                    std::tuple<float, sum<float>, identity<float> >,
                                    std::tuple<float, sum<float>, abs<float>      >,
                                    std::tuple<double, sum<double>, square<double>   >,
                                    std::tuple<double, sum<double>, identity<double> >,
                                    std::tuple<double, sum<double>, abs<double>      > >;

TEMPLATE_LIST_TEST_M(reduction_rm_rw_test, "Uniformly filled Row-Major Row-Wise reduction", 
                                                    "[reduction][rm][small]", reduction_types) {
    this->generate();
    this->test_raw_reduce_wide();
    this->test_raw_reduce_narrow();
}

} // namespace oneapi::dal::backend::primitives::test
