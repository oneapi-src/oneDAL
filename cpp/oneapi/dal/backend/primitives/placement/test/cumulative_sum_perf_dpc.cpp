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
class cumsum_bench_random_1d : public te::float_algo_fixture<Type> {
public:
    void generate() {
        this->m_ = device_max_wg_size(this->get_queue());
        this->n_ = GENERATE(1'048'576, 33'554'432, 134'217'728, 268'435'456);
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

    void generate_input() {
        auto dataframe = GENERATE_DATAFRAME(te::dataframe_builder{ 1, n_ }.fill_uniform(0.0, 2.0));
        this->input_table_ = dataframe.get_table(this->get_homogen_table_id());
    }

    auto fpt_desc() const {
        if constexpr (std::is_same_v<Type, float>) {
            return "float";
        }
        if constexpr (std::is_same_v<Type, double>) {
            return "double";
        }
        REQUIRE(false);
        return "unknown type";
    }

    auto type_desc() const {
        return fmt::format("Floating Point Type: {}", fpt_desc());
    }

    auto data_desc() const {
        check_if_initialized();
        return fmt::format("Dataset of size: {}", this->n_);
    }

    auto workgroup_desc() const {
        check_if_initialized();
        return fmt::format("Workgroup of size: {}", this->m_);
    }

    auto desc() const {
        return fmt::format("{}; {}; {}", type_desc(), data_desc(), workgroup_desc());
    }

    void bench_1d_cumsum() {
        row_accessor<const Type> accessor{ this->input_table_ };
        auto input_array = accessor.pull(this->get_queue(), { 0, -1 }, sycl::usm::alloc::device);
        auto immut_input = ndview<Type, 1>::wrap(input_array.get_data(), { this->n_ });
        auto mut_input =
            ndarray<Type, 1>::empty(this->get_queue(), { this->n_ }, sycl::usm::alloc::device);

        auto mut_2d = mut_input.template reshape<2>({ 1, this->n_ });
        auto immut_2d = immut_input.template reshape<2>({ 1, this->n_ });
        copy(this->get_queue(), mut_2d, immut_2d).wait_and_throw();

        const auto name = desc();
        BENCHMARK(name.c_str()) {
            cumulative_sum_1d(this->get_queue(), mut_input, m_, {}).wait_and_throw();
        };

        REQUIRE(true);
    }

private:
    table input_table_;
    std::int64_t m_, n_;
};

TEMPLATE_LIST_TEST_M(cumsum_bench_random_1d,
                     "Randomly filled array",
                     "[cumsum][1d][small]",
                     cumsum_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->bench_1d_cumsum();
}

} // namespace oneapi::dal::backend::primitives::test
