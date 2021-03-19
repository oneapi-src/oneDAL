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
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_rw.hpp"
#include "oneapi/dal/backend/primitives/reduction/reduction_rm_cw.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

constexpr auto rm_order = ndorder::c;

using reduction_types = std::tuple<std::tuple<float, sum<float>, square<float>>,
                                   std::tuple<float, sum<float>, identity<float>>,
                                   std::tuple<float, sum<float>, abs<float>>,
                                   std::tuple<double, sum<double>, square<double>>,
                                   std::tuple<double, sum<double>, identity<double>>,
                                   std::tuple<double, sum<double>, abs<double>>>;

template <typename Param>
class reduction_rm_test_uniform : public te::policy_fixture {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        arg = GENERATE(-3., -1.e-3, 0, 1.e-3, 3.);
        width = GENERATE(7, 707, 1, 251, 5);
        stride = GENERATE(707, 812, 999, 1001, 1024);
        height = GENERATE(17, 999, 1, 5, 1001);
        SKIP_IF(width > stride);
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

    float_t val_rw() const {
        if (std::is_same_v<sum<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return width * arg;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return width * std::abs(arg);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return width * (arg * arg);
            }
        }
        if (std::is_same_v<min<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return arg;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return std::abs(arg);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return (arg * arg);
            }
        }
        if (std::is_same_v<max<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return arg;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return std::abs(arg);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return (arg * arg);
            }
        }
        REQUIRE(false);
        return 0;
    }

    float_t val_cw() const {
        if (std::is_same_v<sum<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return height * arg;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return height * std::abs(arg);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return height * (arg * arg);
            }
        }
        if (std::is_same_v<min<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return arg;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return std::abs(arg);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return (arg * arg);
            }
        }
        if (std::is_same_v<max<float_t>, binary_t>) {
            if (std::is_same_v<identity<float_t>, unary_t>) {
                return arg;
            }
            if (std::is_same_v<abs<float_t>, unary_t>) {
                return std::abs(arg);
            }
            if (std::is_same_v<square<float_t>, unary_t>) {
                return (arg * arg);
            }
        }
        REQUIRE(false);
        return 0;
    }

    void check_output_rw(ndarray<float_t, 1, rm_order>& outarr, const float_t tol = 1.e-5) {
        CAPTURE(__func__, arg, width, height, stride);
        const auto gtv = val_rw();
        const auto arr = outarr.flatten();
        for (auto i = 0; i < height; ++i) {
            const auto diff = arr[i] - gtv;
            if (diff < -tol || tol < diff) {
                CAPTURE(gtv, arr[i], diff, tol);
                FAIL();
            }
        }
    }

    void check_output_cw(ndarray<float_t, 1, rm_order>& outarr, const float_t tol = 1.e-5) {
        CAPTURE(__func__, arg, width, height, stride);
        const auto gtv = val_cw();
        const auto arr = outarr.flatten();
        for (auto i = 0; i < width; ++i) {
            const auto diff = arr[i] - gtv;
            if (diff < -tol || tol < diff) {
                CAPTURE(gtv, arr[i], diff, tol);
                FAIL();
            }
        }
    }

    void test_raw_rw_reduce_narrow() {
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
                stride,
                binary_t{},
                unary_t{},
                { inp_event, out_event })
            .wait_and_throw();

        check_output_rw(out_array);
    }

    void test_raw_rw_reduce_wide() {
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
                stride,
                binary_t{},
                unary_t{},
                { inp_event, out_event })
            .wait_and_throw();

        check_output_rw(out_array);
    }

    void test_raw_cw_reduce_inplace() {
        using namespace oneapi::dal::backend::primitives;
        using reduction_t = reduction_rm_cw_inplace<float_t, binary_t, unary_t>;
        auto [inp_array, inp_event] = input();
        auto [out_array, out_event] = output();

        const float_t* inp_ptr = inp_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(get_queue());
        reducer(inp_ptr,
                out_ptr,
                width,
                height,
                stride,
                binary_t{},
                unary_t{},
                { inp_event, out_event })
            .wait_and_throw();

        check_output_cw(out_array);
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
    float_t arg;
    std::int64_t width;
    std::int64_t stride;
    std::int64_t height;
};

TEMPLATE_LIST_TEST_M(reduction_rm_test_uniform,
                     "Uniformly filled Row-Major Row-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    this->generate();
    SKIP_IF(this->get_width() > this->get_stride());
    this->test_raw_rw_reduce_wide();
    this->test_raw_rw_reduce_narrow();
}

TEMPLATE_LIST_TEST_M(reduction_rm_test_uniform,
                     "Uniformly filled Row-Major Col-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    this->generate();
    SKIP_IF(this->get_width() > this->get_stride());
    this->test_raw_cw_reduce_inplace();
}

template <typename Param>
class reduction_rm_test_random : public te::policy_fixture {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        width = GENERATE(7, 707, 1, 251, 5);
        stride = GENERATE(707, 812, 999, 1001, 1024);
        height = GENERATE(17, 999, 1, 5, 1001);
        SKIP_IF(width > stride);
        REQUIRE(width <= stride);
        CAPTURE(width, stride, height);
        generate_input();
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    auto output() {
        check_if_initialized();
        return ndarray<float_t, 1, rm_order>::zeros(get_queue(), { height });
    }

    void generate_input() {
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ height, stride }.fill_uniform(-0.2, 0.5));
        this->input_table = train_dataframe.get_table(this->get_homogen_table_id());
    }

    bool is_initialized() const {
        return width > 0 && stride > 0 && height > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    array<float_t> groundtruth_cw() const {
        auto res = array<float_t>::full(width, binary.init_value);
        auto* res_ptr = res.get_mutable_data();
        for (std::int64_t j = 0; j < height; ++j) {
            const auto row_acc = row_accessor<const float_t>{ input_table }.pull({ j, j + 1 });
            for (std::int64_t i = 0; i < width; ++i) {
                const auto val = row_acc[i];
                res_ptr[i] = binary(res_ptr[i], unary(val));
            }
        }
        return res;
    }

    array<float_t> groundtruth_rw() const {
        auto res = array<float_t>::full(height, binary.init_value);
        auto* res_ptr = res.get_mutable_data();
        for (std::int64_t j = 0; j < height; ++j) {
            const auto row_acc = row_accessor<const float_t>{ input_table }.pull({ j, j + 1 });
            for (std::int64_t i = 0; i < width; ++i) {
                const auto val = row_acc[i];
                res_ptr[j] = binary(res_ptr[j], unary(val));
            }
        }
        return res;
    }

    void check_output_rw(ndarray<float_t, 1, rm_order>& outarr, const float_t tol = 1.e-3) {
        CAPTURE(__func__, width, height, stride);
        const auto gtv = groundtruth_rw();
        const auto arr = outarr.flatten();
        for (auto i = 0; i < height; ++i) {
            const auto diff = arr[i] - gtv[i];
            if (diff < -tol || tol < diff) {
                CAPTURE(gtv[i], arr[i], diff, tol);
                FAIL();
            }
        }
    }

    void check_output_cw(ndarray<float_t, 1, rm_order>& outarr, const float_t tol = 1.e-3) {
        CAPTURE(__func__, width, height, stride);
        const auto gtv = groundtruth_cw();
        const auto arr = outarr.flatten();
        for (auto i = 0; i < width; ++i) {
            const auto diff = arr[i] - gtv[i];
            if (diff < -tol || tol < diff) {
                CAPTURE(gtv[i], arr[i], diff, tol);
                FAIL();
            }
        }
    }

    void test_raw_rw_reduce_narrow() {
        using namespace oneapi::dal::backend::primitives;
        using reduction_t = reduction_rm_rw_narrow<float_t, binary_t, unary_t>;
        const auto input_array = row_accessor<const float_t>{ input_table }.pull(get_queue());
        auto [out_array, out_event] = output();

        const float_t* inp_ptr = input_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(get_queue());
        reducer(inp_ptr, out_ptr, width, height, stride, binary, unary, { out_event })
            .wait_and_throw();

        check_output_rw(out_array);
    }

    void test_raw_rw_reduce_wide() {
        using namespace oneapi::dal::backend::primitives;
        using reduction_t = reduction_rm_rw_wide<float_t, binary_t, unary_t>;
        const auto input_array = row_accessor<const float_t>{ input_table }.pull(get_queue());
        auto [out_array, out_event] = output();

        const float_t* inp_ptr = input_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(get_queue());
        reducer(inp_ptr, out_ptr, width, height, stride, binary, unary, { out_event })
            .wait_and_throw();

        check_output_rw(out_array);
    }

    void test_raw_cw_reduce_inplace() {
        using namespace oneapi::dal::backend::primitives;
        using reduction_t = reduction_rm_cw_inplace<float_t, binary_t, unary_t>;
        const auto input_array = row_accessor<const float_t>{ input_table }.pull(get_queue());
        auto [out_array, out_event] = output();

        const float_t* inp_ptr = input_array.get_data();
        float_t* out_ptr = out_array.get_mutable_data();

        reduction_t reducer(get_queue());
        reducer(inp_ptr, out_ptr, width, height, stride, binary, unary, { out_event })
            .wait_and_throw();

        check_output_cw(out_array);
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
    const binary_t binary{};
    const unary_t unary{};

private:
    std::int64_t width;
    std::int64_t stride;
    std::int64_t height;
    table input_table;
};

TEMPLATE_LIST_TEST_M(reduction_rm_test_random,
                     "Randomly filled Row-Major Row-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    this->generate();
    SKIP_IF(this->get_width() > this->get_stride());
    this->test_raw_rw_reduce_wide();
    this->test_raw_rw_reduce_narrow();
}

TEMPLATE_LIST_TEST_M(reduction_rm_test_random,
                     "Randomly filled Row-Major Col-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    this->generate();
    SKIP_IF(this->get_width() > this->get_stride());
    this->test_raw_cw_reduce_inplace();
}

} // namespace oneapi::dal::backend::primitives::test
