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

#include <cmath>
#include <type_traits>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/placement/cumulative_sum.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using cumsum_types = std::tuple<float, double>;

template <typename Type>
class cumsum_test_random_1d : public te::float_algo_fixture<Type> {
public:
    void generate() {
        this->m_ = GENERATE(4, 8, 16, 32, 64, 128);
        this->n_ = GENERATE(64,
                            511,
                            512,
                            513,
                            1023, //
                            1024,
                            1025,
                            2047,
                            2048,
                            4096,
                            4097,
                            8191, //
                            8192,
                            8193,
                            16383,
                            16384,
                            16385,
                            65537, //
                            262143,
                            262144,
                            262145,
                            270000,
                            213456);
        this->generate_input();
    }

    bool is_initialized() const {
        return (this->n_ > 0) && (this->m_ > 0);
    }

    void check_if_initialized() const {
        if (!this->is_initialized()) {
            throw std::runtime_error{ "cumsum test is not initialized" };
        }
    }

    auto groundtruth() const {
        row_accessor<const Type> accessor{ this->input_table_ };
        auto inp = accessor.pull({ 0, -1 });
        auto res = ndarray<Type, 1, ndorder::c>::zeros(this->n_ + 1);

        for (std::int64_t i = 0; i < this->n_; ++i) {
            res.at(i + 1) = res.at(i) + inp[i];
        }

        return res;
    }

    void generate_input() {
        auto dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, this->n_ }.fill_uniform(0.0, 2.0));
        this->input_table_ = dataframe.get_table(this->get_homogen_table_id());
    }

    void test_1d_cumsum(const Type tol = 1.e-5) {
        check_if_initialized();

        constexpr auto eps = std::numeric_limits<Type>::epsilon();
        row_accessor<const Type> accessor{ this->input_table_ };
        auto input_array = accessor.pull(this->get_queue(), { 0, -1 }, sycl::usm::alloc::device);
        auto immut_input = ndview<Type, 1>::wrap(input_array.get_data(), { this->n_ });
        auto mut_input =
            ndarray<Type, 1>::empty(this->get_queue(), { this->n_ }, sycl::usm::alloc::device);

        auto mut_2d = mut_input.template reshape<2>({ 1, this->n_ });
        auto immut_2d = immut_input.template reshape<2>({ 1, this->n_ });
        auto copy_event = copy(this->get_queue(), mut_2d, immut_2d);
        auto cum_event = cumulative_sum_1d(this->get_queue(), mut_input, this->m_, { copy_event });

        const auto gtr = this->groundtruth();
        const auto res = mut_input.to_host(this->get_queue(), { cum_event });

        for (std::int64_t i = 0; i < this->n_; ++i) {
            const auto gtr_val = gtr.at(i + 1);
            const auto res_val = res.at(i);

            const auto adiff = std::abs(res_val - gtr_val);
            const auto rdiff = adiff / std::max({ eps, std::abs(gtr_val), std::abs(res_val) });
            if (tol < rdiff) {
                CAPTURE(this->n_, i, gtr_val, res_val, adiff, rdiff, tol);
                FAIL();
            }
        }

        SUCCEED();
    }

private:
    table input_table_;
    std::int64_t m_, n_;
};

TEMPLATE_LIST_TEST_M(cumsum_test_random_1d,
                     "Randomly filled array",
                     "[cumsum][1d][small]",
                     cumsum_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_1d_cumsum();
}

} // namespace oneapi::dal::backend::primitives::test
