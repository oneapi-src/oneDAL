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
#include <random>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/placement/argwhere.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

struct left_align {};
struct right_align {};

template <typename Align>
struct align_map {};

template <>
struct align_map<left_align> {
    constexpr static auto value = where_alignment::left;

    template <typename Type>
    static inline bool comp(const Type& left, const Type& right) {
        return left <= right;
    }
};

template <>
struct align_map<right_align> {
    constexpr static auto value = where_alignment::right;

    template <typename Type>
    static inline bool comp(const Type& left, const Type& right) {
        return left < right;
    }
};

using argwhere_types = std::tuple<std::tuple<float, left_align>,
                                  std::tuple<float, right_align>,
                                  std::tuple<double, left_align>,
                                  std::tuple<double, right_align>>;

template <typename Param>
class argwhere_test_random_1d : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using index_t = std::int64_t;
    using float_t = std::tuple_element_t<0, Param>;
    using align_t = std::tuple_element_t<1, Param>;

    constexpr static auto align_v = align_map<align_t>::value;
    using params_t = where_alignment_map<index_t, align_v>;

    void generate() {
        this->n_ = GENERATE(7, 4097, 8191, 16385, 65536);
        this->m_ = GENERATE(1, 2, 4, 7, 16, 32, 64, 128);
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
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, this->n_ }.fill_uniform(0.5, 2.5, 1111));

        this->input_table_ = dataframe.get_table(this->get_homogen_table_id());
    }

    void test_1d_argwhere() {
        check_if_initialized();

        auto queue = this->get_queue();
        constexpr auto alloc = sycl::usm::alloc::device;
        row_accessor<const float_t> accessor(this->input_table_);
        const auto device_array = accessor.pull(queue, { 0, -1 }, alloc);
        const auto device = ndview<float_t, 1>::wrap(device_array);
        const auto host_array = accessor.pull({ 0, -1 });

        SECTION("Random index") {
            std::mt19937_64 generator(777);
            for (index_t i = 0; i < this->m_; ++i) {
                const index_t idx = generator() % this->n_;
                const float_t val = host_array[idx];

                auto comp = [val](float_t check) {
                    return val == check;
                };

                const auto res = argwhere_one(queue, comp, device, align_v);

                CAPTURE(idx, val, res, params_t::identity);
                REQUIRE(idx == res);
            }
        };

        SECTION("Invalid huge number") {
            const auto val = std::numeric_limits<float_t>::max();

            auto comp = [=](float_t check) {
                return val == check;
            };

            const auto res = argwhere_one(queue, comp, device, align_v);

            CAPTURE(val, res, params_t::identity);
            REQUIRE(params_t::identity == res);
        };

        SECTION("Invalid small number") {
            const auto val = std::numeric_limits<float_t>::lowest();

            auto comp = [=](float_t check) {
                return val == check;
            };

            const auto res = argwhere_one(queue, comp, device, align_v);

            CAPTURE(val, res, params_t::identity);
            REQUIRE(params_t::identity == res);
        };
    }

    void test_1d_argmin() {
        check_if_initialized();

        auto queue = this->get_queue();
        constexpr auto alloc = sycl::usm::alloc::device;
        row_accessor<const float_t> accessor(this->input_table_);
        const auto device_array = accessor.pull(queue, { 0, -1 }, alloc);
        const auto device = ndview<float_t, 1>::wrap(device_array);
        const auto host_array = accessor.pull({ 0, -1 });

        index_t idx = -1;
        float_t val = std::numeric_limits<float_t>::max();
        for (index_t i = 0; i < this->n_; ++i) {
            const auto curr = host_array[i];
            if (align_map<align_t>::comp(curr, val)) {
                val = curr;
                idx = i;
            }
        }

        auto [min, res] = argmin(queue, device, align_v);

        CAPTURE(val, idx, min, res);

        REQUIRE(res == idx);
        REQUIRE(min == val);
    }

    void test_1d_argmax() {
        check_if_initialized();

        auto queue = this->get_queue();
        constexpr auto alloc = sycl::usm::alloc::device;
        row_accessor<const float_t> accessor(this->input_table_);
        const auto device_array = accessor.pull(queue, { 0, -1 }, alloc);
        const auto device = ndview<float_t, 1>::wrap(device_array);
        const auto host_array = accessor.pull({ 0, -1 });

        index_t idx = -1;
        float_t val = std::numeric_limits<float_t>::lowest();
        for (index_t i = 0; i < this->n_; ++i) {
            const auto curr = host_array[i];
            if (align_map<align_t>::comp(val, curr)) {
                val = curr;
                idx = i;
            }
        }

        auto [max, res] = argmax(queue, device, align_v);

        CAPTURE(val, idx, max, res);

        REQUIRE(res == idx);
        REQUIRE(max == val);
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
                     "Random argextreme",
                     "[argmin][1d][small]",
                     argwhere_types) {
    SKIP_IF(this->not_float64_friendly());

    this->generate();
    this->test_1d_argmin();
    this->test_1d_argmax();
}

} // namespace oneapi::dal::backend::primitives::test
