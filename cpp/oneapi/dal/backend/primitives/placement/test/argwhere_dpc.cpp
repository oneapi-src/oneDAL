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

#include <type_traits>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/placement/argwhere.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using argwhere_types = std::tuple<float, double>;

template <typename Float>
class argwhere_test_random_1d : public te::float_algo_fixture<Float> {
public:
    void generate() {
        this->n_ = GENERATE(4097, 8191, 16385, 65536);
        this->m_ = GENERATE(1, 2, 128);
        this->generate_input();
    }

    bool is_initialized() const {
        return (this->m_ > 0) && (this->n_ > 0);
    }

    void check_if_initialized() const {
        if (!this->is_initialized()) {
            throw std::runtime_error{ "argwhere test is not initialized" };
        }
    }

    void generate_input() {
        check_if_initialized();

        const auto dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, this->n_ }.fill_uniform(0.5, 2.5, 3333));

        this->input_table_ = dataframe.get_table(this->get_homogen_table_id());
    }



    void test_1d_argwhere() {
        check_if_initialized();

        auto queue = this->get_queue();
        constexpr auto alloc = sycl::usm::alloc::device;
        row_accessor<const Float> accessor(this->input_table_);
        const auto device_array = accessor.pull(queue, {0, -1}, alloc);
        const auto device = ndview<Float, 1>::wrap(device_array);
        const auto host_array = accessor.pull({0, -1});

        SECTION("Random index") {

            auto idx = std::rand() % this->m_;
            const auto val = host_array[idx];

            auto comp = [=](Float check) { return val == check; };

            const auto res = argwhere_one(queue, comp, device);

            CAPTURE(idx, val, res, host_array[res]);
            REQUIRE(idx == res);
        };

        SECTION("Invalid number") {

            const auto val = std::numeric_limits<Float>::max();

            auto comp = [=](Float check) { return val == check; };

            const auto res = argwhere_one(queue, comp, device);

            CAPTURE(val, res);
            REQUIRE(-1l == res);
        };
    }

    void test_1d_argmin() {
        check_if_initialized();


    }

private:
    table input_table_;
    std::int64_t m_, n_;
};

TEMPLATE_LIST_TEST_M(argwhere_test_random_1d,
                     "Random argwhere",
                     "[argwhere][1d][small]",
                     argwhere_types) {
    SKIP_IF(this->not_float64_friendly());

    this->generate();
    this->test_1d_argwhere();
}

TEMPLATE_LIST_TEST_M(argwhere_test_random_1d,
                     "Random argmin",
                     "[argmin][1d][small]",
                     argwhere_types) {
    SKIP_IF(this->not_float64_friendly());

    this->generate();
    this->test_1d_argmin();
}

} // namespace oneapi::dal::backend::primitives::test
