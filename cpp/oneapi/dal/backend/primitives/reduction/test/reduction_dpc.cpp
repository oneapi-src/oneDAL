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
#include "oneapi/dal/backend/primitives/reduction/reduction_dpc.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using reduction_types = std::tuple<std::tuple<float, sum<float>, square<float>>,
                                   std::tuple<double, sum<double>, square<double>>>;

template <typename Param>
class reduction_test_random : public te::policy_fixture {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        width = GENERATE(7, 707, 1, 251, 5);
        height = GENERATE(17, 999, 1, 5, 1001);
        CAPTURE(width, height);
        generate_input();
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    auto output(std::int64_t size) {
        check_if_initialized();
        return ndarray<float_t, 1, ndorder::c>::zeros(get_queue(), { size });
    }

    void generate_input() {
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ height, width }.fill_uniform(-0.2, 0.5));
        this->input_table = train_dataframe.get_table(this->get_homogen_table_id());
    }

    bool is_initialized() const {
        return width > 0 && height > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    array<float_t> groundtruth_rm_cw() const {
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

    array<float_t> groundtruth_rm_rw() const {
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

    void check_array(const array<float_t>& gtv,
                     const array<float_t>& arr,
                     const float_t tol = 1.e-3) {
        CAPTURE(__func__, gtv.get_count(), arr.get_count(), width, height);
        REQUIRE(gtv.get_count() == arr.get_count());
        for (auto i = 0; i < arr.get_count(); ++i) {
            const auto diff = arr[i] - gtv[i];
            if (diff < -tol || tol < diff) {
                CAPTURE(gtv[i], arr[i], diff, tol);
                FAIL();
            }
        }
    }

    void check_output_rm_rw(ndarray<float_t, 1, ndorder::c>& outarr, const float_t tol = 1.e-3) {
        CAPTURE(__func__, width, height, outarr.get_count());
        const auto gtv = groundtruth_rm_rw();
        const auto arr = outarr.flatten();
        check_array(gtv, arr, tol);
    }

    void check_output_cm_rw(ndarray<float_t, 1, ndorder::c>& outarr, const float_t tol = 1.e-3) {
        CAPTURE(__func__, width, height, outarr.get_count());
        const auto gtv = groundtruth_rm_cw();
        const auto arr = outarr.flatten();
        check_array(gtv, arr, tol);
    }

    void check_output_rm_cw(ndarray<float_t, 1, ndorder::c>& outarr, const float_t tol = 1.e-3) {
        CAPTURE(__func__, width, height, outarr.get_count());
        const auto gtv = groundtruth_rm_cw();
        const auto arr = outarr.flatten();
        check_array(gtv, arr, tol);
    }

    void check_output_cm_cw(ndarray<float_t, 1, ndorder::c>& outarr, const float_t tol = 1.e-3) {
        CAPTURE(__func__, width, height);
        const auto gtv = groundtruth_rm_rw();
        const auto arr = outarr.flatten();
        check_array(gtv, arr, tol);
    }

    void test_rm_rw_reduce() {
        auto input_array = row_accessor<const float_t>{ input_table }.pull(get_queue());
        auto [output_array, out_event] = output(height);
        auto input =
            ndview<float_t, 2, ndorder::c>::wrap(input_array.get_mutable_data(), { height, width });
        auto output =
            ndview<float_t, 1, ndorder::c>::wrap(output_array.get_mutable_data(), { height });

        auto reduce_event =
            reduce_by_rows(get_queue(), input, output, binary_t{}, unary_t{}, { out_event });
        reduce_event.wait_and_throw();

        check_output_rm_rw(output_array);
    }

    void test_rm_cw_reduce() {
        auto input_array = row_accessor<const float_t>{ input_table }.pull(get_queue());
        auto [output_array, out_event] = output(width);
        auto input =
            ndview<float_t, 2, ndorder::c>::wrap(input_array.get_mutable_data(), { height, width });
        auto output =
            ndview<float_t, 1, ndorder::c>::wrap(output_array.get_mutable_data(), { width });

        auto reduce_event =
            reduce_by_columns(get_queue(), input, output, binary_t{}, unary_t{}, { out_event });
        reduce_event.wait_and_throw();

        check_output_rm_cw(output_array);
    }

    void test_cm_cw_reduce() {
        auto input_array = row_accessor<const float_t>{ input_table }.pull(get_queue());
        auto [output_array, out_event] = output(height);
        auto input_tr =
            ndview<float_t, 2, ndorder::c>::wrap(input_array.get_mutable_data(), { height, width });
        auto input = input_tr.t();
        auto output =
            ndview<float_t, 1, ndorder::c>::wrap(output_array.get_mutable_data(), { height });

        auto reduce_event =
            reduce_by_columns(get_queue(), input, output, binary_t{}, unary_t{}, { out_event });
        reduce_event.wait_and_throw();

        check_output_rm_rw(output_array);
    }

    void test_cm_rw_reduce() {
        auto input_array = row_accessor<const float_t>{ input_table }.pull(get_queue());
        auto [output_array, out_event] = output(width);
        auto input_tr =
            ndview<float_t, 2, ndorder::c>::wrap(input_array.get_mutable_data(), { height, width });
        auto input = input_tr.t();
        auto output =
            ndview<float_t, 1, ndorder::c>::wrap(output_array.get_mutable_data(), { width });

        auto reduce_event =
            reduce_by_rows(get_queue(), input, output, binary_t{}, unary_t{}, { out_event });
        reduce_event.wait_and_throw();

        check_output_cm_rw(output_array);
    }

    auto get_width() const {
        return width;
    }

    auto get_height() const {
        return height;
    }

private:
    const binary_t binary{};
    const unary_t unary{};

private:
    std::int64_t width;
    std::int64_t height;
    table input_table;
};

TEMPLATE_LIST_TEST_M(reduction_test_random,
                     "Randomly filled Row-Major Row-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    this->generate();
    this->test_rm_rw_reduce();
}

TEMPLATE_LIST_TEST_M(reduction_test_random,
                     "Randomly filled Row-Major Col-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    this->generate();
    this->test_rm_cw_reduce();
}

TEMPLATE_LIST_TEST_M(reduction_test_random,
                     "Randomly filled Col-Major Col-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    this->generate();
    this->test_cm_cw_reduce();
}

TEMPLATE_LIST_TEST_M(reduction_test_random,
                     "Randomly filled Col-Major Row-Wise reduction",
                     "[reduction][rm][small]",
                     reduction_types) {
    this->generate();
    this->test_cm_rw_reduce();
}

} // namespace oneapi::dal::backend::primitives::test
