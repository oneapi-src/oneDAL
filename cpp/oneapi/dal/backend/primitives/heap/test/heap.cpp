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

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/heap.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using heap_types = std::tuple<std::tuple<float, std::int32_t>, std::tuple<double, std::int64_t>>;

template <typename Param>
class heap_test_random : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
    using float_t = std::tuple_element_t<0, Param>;
    using index_t = std::tuple_element_t<1, Param>;

public:
    void generate() {
        width_ = GENERATE(1, 2, 3, 4, 5, 6, 7, 8, 9);
        generate_input();
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    void generate_input() {
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder(1, width_).fill_uniform(-0.2, 0.5));
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }

    bool is_initialized() const {
        return width_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    auto input() const {
        auto input_array = row_accessor<const float_t>{ input_table_ }.pull();
        return ndview<float_t, 1>::wrap(input_array.get_data(), { width_ });
    }

    auto indices() const {
        auto res = ndarray<index_t, 1>::empty(width_);
        for (index_t i = 0; i < width_; ++i)
            *(res.get_mutable_data() + i) = i;
        return res;
    }

    auto make_heap_groundtruth() const {
        auto res = indices();
        const auto inp = input();
        const auto comp = [&](const auto& left, const auto& right) -> bool {
            return *(inp.get_data() + left) < *(inp.get_data() + right);
        };
        auto* from = res.get_mutable_data();
        auto* to = res.get_mutable_data() + width_;
        std::make_heap(from, to, comp);
        return res;
    }

    auto make_heap_res_to_test() const {
        auto res = indices();
        const auto inp = input();
        const auto comp = [&](const auto& left, const auto& right) -> bool {
            return *(inp.get_data() + left) < *(inp.get_data() + right);
        };
        auto* from = res.get_mutable_data();
        auto* to = res.get_mutable_data() + width_;
        pr::detail::make_heap_impl(from, to, comp);
        return res;
    }

    void check_make_heap() {
        check_if_initialized();

        const auto res = make_heap_res_to_test();
        const auto gtr = make_heap_groundtruth();

        for (std::int32_t i = 0; i < width_; ++i) {
            const auto r = *(res.get_data() + i);
            const auto g = *(gtr.get_data() + i);
            CAPTURE(width_, i, r, g);
            REQUIRE(r == g);
        }
    }

    auto sort_heap_groundtruth() const {
        auto heap = make_heap_groundtruth();
        const auto inp = input();
        const auto comp = [&](const auto& left, const auto& right) -> bool {
            return *(inp.get_data() + left) < *(inp.get_data() + right);
        };
        auto* from = heap.get_mutable_data();
        auto* to = heap.get_mutable_data() + width_;
        std::sort_heap(from, to, comp);
        return heap;
    }

    auto sort_heap_res_to_test() const {
        auto heap = make_heap_groundtruth();
        const auto inp = input();
        const auto comp = [&](const auto& left, const auto& right) -> bool {
            return *(inp.get_data() + left) < *(inp.get_data() + right);
        };
        auto* from = heap.get_mutable_data();
        auto* to = heap.get_mutable_data() + width_;
        detail::sort_heap_impl(from, to, comp);
        return heap;
    }

    void check_sort_heap() {
        check_if_initialized();

        auto res = sort_heap_res_to_test();
        auto gtr = sort_heap_groundtruth();

        for (std::int32_t i = 0; i < width_; ++i) {
            const auto r = *(res.get_data() + i);
            const auto g = *(gtr.get_data() + i);
            CAPTURE(width_, i, r, g);
            REQUIRE(r == g);
        }
    }

    auto push_heap_groundtruth() const {
        auto res = indices();
        const auto inp = input();
        const auto comp = [&](const auto& left, const auto& right) -> bool {
            return *(inp.get_data() + left) < *(inp.get_data() + right);
        };
        auto* from = res.get_mutable_data();
        for (std::int32_t i = 1; i < width_; ++i) {
            std::push_heap(from, from + i, comp);
        }
        return res;
    }

    auto push_heap_res_to_test() const {
        auto res = indices();
        const auto inp = input();
        const auto comp = [&](const auto& left, const auto& right) -> bool {
            return *(inp.get_data() + left) < *(inp.get_data() + right);
        };
        auto* from = res.get_mutable_data();
        for (std::int32_t i = 1; i < width_; ++i) {
            detail::push_heap_impl(from, from + i, comp);
        }
        return res;
    }

    void check_push_heap() {
        check_if_initialized();

        const auto res = push_heap_res_to_test();
        const auto gtr = push_heap_groundtruth();

        for (std::int32_t i = 0; i < width_; ++i) {
            const auto r = *(res.get_data() + i);
            const auto g = *(gtr.get_data() + i);
            CAPTURE(width_, i, r, g);
            REQUIRE(r == g);
        }
    }

private:
    std::int32_t width_;
    table input_table_;
};

TEMPLATE_LIST_TEST_M(heap_test_random,
                     "Randomly filled heap creation",
                     "[heap][make][small]",
                     heap_types) {
    this->generate();
    this->generate_input();
    this->check_make_heap();
}

TEMPLATE_LIST_TEST_M(heap_test_random,
                     "Randomly created heap sorting",
                     "[heap][sort][small]",
                     heap_types) {
    this->generate();
    this->generate_input();
    this->check_sort_heap();
}

TEMPLATE_LIST_TEST_M(heap_test_random,
                     "Randomly created heap pushing",
                     "[heap][push][small]",
                     heap_types) {
    this->generate();
    this->generate_input();
    this->check_push_heap();
}

} // namespace oneapi::dal::backend::primitives::test
