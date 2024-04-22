/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include <limits>
#include <type_traits>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/reduction/reduction_1d.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using reduction_types = std::tuple<std::tuple<float, sum<float>, square<float>>,
                                   std::tuple<double, sum<double>, square<double>>>

    using finiteness_types = std::tuple<std::tuple<float, sum<float>, identity<float>>,
                                        std::tuple<double, sum<double>, identity<double>>,
                                        std::tuple<float, logical_or<float>, isinfornan<float>>,
                                        std::tuple<double, logical_or<double>, isinfornan<double>>>;

template <typename Param>
class reduction_test_random_1d : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        n_ = GENERATE(17, 999, 1, 5, 1001);
        CAPTURE(n_);
        generate_input();
    }

    bool is_initialized() const {
        return n_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    float_t groundtruth() const {
        float_t res = float_t(binary_.init_value);
        auto inp = row_accessor<const float_t>{ this->input_table_ }.pull({ 0, 1 });
        for (std::int64_t i = 0; i < inp.get_count(); ++i) {
            res = binary_(res, unary_(inp[i]));
        }
        return res;
    }

    void generate_input() {
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, n_ }.fill_uniform(-0.2, 0.5));
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }

    void test_1d_reduce(const float_t tol = 1.e-3) {
        auto input_array = row_accessor<const float_t>{ input_table_ }.pull(this->get_queue());
        auto input = ndview<float_t, 1>::wrap(input_array.get_data(), { n_ });

        float_t out = reduce_1d(this->get_queue(), input, binary_t{}, unary_t{}, {});

        float_t ans = groundtruth();

        if (out - ans < -tol || out - ans > tol) {
            CAPTURE(out, ans, out - ans, tol);
            FAIL();
        }
        SUCCEED();
    }

protected:
    table input_table_;
    const binary_t binary_{};
    const unary_t unary_{};
    std::int64_t n_;
};

template <typename Param>
class infinite_sum_test_random_1d : public reduction_test_random_1d<Param> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate(bool maxval) {
        n_ = GENERATE(17, 999, 1, 5, 1001);
        CAPTURE(n_);
        generate_input(maxval);
    }

    void generate_input(bool maxval) {
        float_t inp = 0.9 * (float_t)maxval * services::internal::MaxVal<float_t>;
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, n_ }.fill_uniform(0.0, inp));
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }
}

template <typename Param>
class single_infinite_test_random_1d : public reduction_test_random_1d<Param> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate(bool infval) {
        n_ = GENERATE(17, 999, 1, 5, 1001);
        CAPTURE(n_);
        generate_input(infval);
    }

    void generate_input(bool infval) {
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, n_ }.fill_uniform(-0.2, 0.5));
        auto inner_iter_count_arr_host = train_dataframe.get_array();

        inner_iter_count_arr_host[5] = infval ? std::numeric_limits<float_t>::infinity
                                              : std::numeric_limits<float_t>::quiet_NaN();
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }
}

TEMPLATE_LIST_TEST_M(reduction_test_random_1d,
                     "Randomly filled array",
                     "[reduction][1d][small]",
                     reduction_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_1d_reduce();
}

TEMPLATE_LIST_TEST_M(infinite_sum_test_random_1d,
                     "Randomly filled array with infinite sum",
                     "[reduction][1d][small]",
                     finiteness_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate(true);
    this->test_1d_reduce();
    this->generate(false);
    this->test_1d_reduce();
}

TEMPLATE_LIST_TEST_M(single_infinite_test_random_1d,
                     "Randomly filled array with a single inf or nan",
                     "[reduction][1d][small]",
                     finiteness_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate(true);
    this->test_1d_reduce();
    this->generate(false);
    this->test_1d_reduce();
}

} // namespace oneapi::dal::backend::primitives::test
