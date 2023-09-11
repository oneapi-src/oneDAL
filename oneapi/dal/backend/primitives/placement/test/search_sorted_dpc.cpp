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
#include "oneapi/dal/backend/primitives/placement/search_sorted.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

struct left_alignment_tag {};
struct right_alignment_tag {};

template <typename AlignmentTag>
struct alignment_tag_map {};

template <>
struct alignment_tag_map<left_alignment_tag> {
    constexpr static auto value = search_alignment::left;
};

template <>
struct alignment_tag_map<right_alignment_tag> {
    constexpr static auto value = search_alignment::right;
};

template <typename AlignmentTag>
constexpr auto alignment_v = alignment_tag_map<AlignmentTag>::value;

using searchsorted_types = std::tuple<std::tuple<left_alignment_tag, float, std::int32_t>,
                                      std::tuple<left_alignment_tag, float, std::int64_t>,
                                      std::tuple<right_alignment_tag, float, std::int32_t>,
                                      std::tuple<right_alignment_tag, double, std::int64_t>>;

template <typename Param>
class searchsorted_test_random_1d : public te::float_algo_fixture<std::tuple_element_t<1, Param>> {
public:
    using type_t = std::tuple_element_t<1, Param>;
    using index_t = std::tuple_element_t<2, Param>;
    using align_t = std::tuple_element_t<0, Param>;

    constexpr static auto alignment = alignment_v<align_t>;

    void generate() {
        this->m_ = GENERATE(5, 9, 511, 1027, 4096);
        this->n_ = GENERATE(511, 512, 513, 1025, 2047, 4097, 8191, 16385, 65536);
        this->generate_input();
    }

    bool is_initialized() const {
        return (this->n_ > 0) && (this->m_ > 0);
    }

    void check_if_initialized() const {
        if (!this->is_initialized()) {
            throw std::runtime_error{ "searchsorted test is not initialized" };
        }
    }

    void generate_input() {
        check_if_initialized();

        const auto input_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, this->n_ }.fill_uniform(0.5, 2.5, 3333));
        const auto point_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, this->m_ }.fill_uniform(0.0, 3.0, 9999));

        const auto raw_input = input_dataframe.get_table(this->get_homogen_table_id());
        const auto arr_input =
            row_accessor<const type_t>(raw_input).pull({ 0, -1 }).need_mutable_data();

        auto* first = arr_input.get_mutable_data();
        std::sort(first, first + this->n_);

        this->input_table_ = homogen_table::wrap(arr_input, 1, this->n_);
        this->point_table_ = point_dataframe.get_table(this->get_homogen_table_id());
    }

    bool check_order(type_t left, type_t mid, type_t right) {
        constexpr auto is_left = alignment == search_alignment::left;
        constexpr auto is_right = alignment == search_alignment::right;

        if constexpr (is_left) {
            return (left < mid) && (mid <= right);
        }

        if constexpr (is_right) {
            return (left <= mid) && (mid < right);
        }

        return false;
    }

    void test_1d_searchsorted() {
        check_if_initialized();

        constexpr auto alloc = sycl::usm::alloc::device;

        row_accessor<const type_t> inputs_acc{ this->input_table_ };
        row_accessor<const type_t> points_acc{ this->point_table_ };

        auto inputs_host = inputs_acc.pull({ 0, -1 });
        auto points_host = points_acc.pull({ 0, -1 });

        auto inputs_device = inputs_acc.pull(this->get_queue(), { 0, -1 }, alloc);
        auto points_device = points_acc.pull(this->get_queue(), { 0, -1 }, alloc);

        auto inputs = ndview<type_t, 1>::wrap(inputs_device.get_data(), { this->n_ });
        auto points = ndview<type_t, 1>::wrap(points_device.get_data(), { this->m_ });

        auto [result_device, result_event] =
            ndarray<index_t, 1>::zeros(this->get_queue(), { this->m_ }, alloc);

        auto search_event = search_sorted_1d(this->get_queue(),
                                             alignment,
                                             inputs,
                                             points,
                                             result_device,
                                             { result_event });

        auto result = result_device.to_host(this->get_queue(), { search_event });

        const auto prev_val = [&](index_t idx) -> type_t {
            constexpr auto min = -std::numeric_limits<type_t>::max();
            return idx == 0 ? min : inputs_host[idx - 1];
        };

        const auto next_val = [&](index_t idx) -> type_t {
            constexpr auto max = +std::numeric_limits<type_t>::max();
            return idx < (n_ - 1) ? inputs_host[idx + 1] : max;
        };

        for (std::int64_t i = 0; i < this->m_; ++i) {
            const auto val = points_host[i];
            const auto idx = result.at(i);

            const auto prev = prev_val(idx);
            const auto next = next_val(idx);
            CAPTURE(i, val, idx, prev, next);
            REQUIRE(this->check_order(prev, val, next));
        }
    }

private:
    table input_table_;
    table point_table_;
    std::int64_t m_, n_;
};

TEMPLATE_LIST_TEST_M(searchsorted_test_random_1d,
                     "Random sorted array",
                     "[searchsorted][1d][medium]",
                     searchsorted_types) {
    SKIP_IF(this->not_float64_friendly());

    this->generate();
    this->test_1d_searchsorted();
}

} // namespace oneapi::dal::backend::primitives::test
