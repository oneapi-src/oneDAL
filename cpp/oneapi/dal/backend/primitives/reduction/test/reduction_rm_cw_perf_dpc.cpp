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

#include <array>
#include <cmath>
#include <type_traits>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/reduction/functors.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_rw_dpc.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw_dpc.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

constexpr auto rm_order = ndorder::c;

using reduction_types = std::tuple<std::tuple<float, sum<float>, square<float>>>;

template <typename Param>
class reduction_rm_test_uniform : public te::policy_fixture {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        width = GENERATE(16, 128, 1024);
        stride = GENERATE(16, 128, 1024);
        height = GENERATE(16, 128, 1024, 16384, 32768);
        SKIP_IF(width > stride);
        REQUIRE(width <= stride);
        CAPTURE(width, stride, height);
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
        return ndarray<float_t, 2, rm_order>::zeros(get_queue(),
                                                    { stride, height },
                                                    sycl::usm::alloc::device);
    }

    auto output(std::int64_t size) {
        check_if_initialized();
        return ndarray<float_t, 1, rm_order>::zeros(get_queue(),
                                                    { size },
                                                    sycl::usm::alloc::device);
    }

    auto fpt_desc() {
        if constexpr (std::is_same_v<float_t, float>) {
            return "float";
        }
        if constexpr (std::is_same_v<float_t, double>) {
            return "double";
        }
        REQUIRE(false);
        return "unknown type";
    }

    auto type_desc() {
        return fmt::format("Floating Point Type: {}", fpt_desc());
    }

    auto matrix_desc() {
        check_if_initialized();
        return fmt::format("Row-Major Matrix with parameters: "
                           "width = {}, stride = {}, height = {}",
                           width,
                           stride,
                           height);
    }

    auto unary_desc() {
        if (std::is_same_v<identity<float_t>, unary_t>) {
            return "Identity Unary Functor";
        }
        if (std::is_same_v<abs<float_t>, unary_t>) {
            return "Abs Unary Functor";
        }
        if (std::is_same_v<square<float_t>, unary_t>) {
            return "Square Unary Functor";
        }
        REQUIRE(false);
        return "Unknown Unary Functor";
    }

    auto binary_desc() {
        if (std::is_same_v<sum<float_t>, binary_t>) {
            return "Sum Binary Functor";
        }
        if (std::is_same_v<max<float_t>, binary_t>) {
            return "Max Binary Functor";
        }
        if (std::is_same_v<min<float_t>, binary_t>) {
            return "Min Binary Functor";
        }
        REQUIRE(false);
        return "Unknown Binary Functor";
    }

    auto desc() {
        return fmt::format("{}; {}; {}; {}",
                           matrix_desc(),
                           type_desc(),
                           unary_desc(),
                           binary_desc());
    }

    void test_raw_cw_reduce_inplace() {
        using reduction_t = reduction_rm_cw_inplace<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(width);

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        const auto name = fmt::format("Inplace CW Reduction: {}", desc());

        get_queue().wait_and_throw();

        BENCHMARK(name.c_str()) {
            reduction_t reducer(get_queue());
            reducer(inp_ptr, out_ptr, width, height, stride, binary_t{}, unary_t{})
                .wait_and_throw();
        };
    }

    void test_raw_cw_reduce_inplace_local() {
        using reduction_t = reduction_rm_cw_inplace_local<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(width);

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        const auto name = fmt::format("Inplace Local CW Reduction: {}", desc());

        get_queue().wait_and_throw();

        BENCHMARK(name.c_str()) {
            reduction_t reducer(get_queue());
            reducer(inp_ptr, out_ptr, width, height, stride, binary_t{}, unary_t{})
                .wait_and_throw();
        };
    }

    void test_raw_cw_reduce_wrapper() {
        using reduction_t = reduction_rm_cw<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output(width);

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        const auto name = fmt::format("Inplace CW Reduction Wrapper: {}", desc());

        get_queue().wait_and_throw();

        BENCHMARK(name.c_str()) {
            reduction_t reducer(get_queue());
            reducer(inp_ptr, out_ptr, width, height, stride, binary_t{}, unary_t{})
                .wait_and_throw();
        };
    }

    auto get_width() const {
        return width;
    }

    auto get_stride() const {
        return stride;
    }

    auto get_height() const {
        return height;
    }

private:
    std::int64_t width;
    std::int64_t stride;
    std::int64_t height;
};

TEMPLATE_LIST_TEST_M(reduction_rm_test_uniform,
                     "Uniformly filled Row-Major Col-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    this->generate();
    SKIP_IF(this->get_width() > this->get_stride());
    this->test_raw_cw_reduce_inplace();
    this->test_raw_cw_reduce_inplace_local();
    this->test_raw_cw_reduce_wrapper();
}

} // namespace oneapi::dal::backend::primitives::test
