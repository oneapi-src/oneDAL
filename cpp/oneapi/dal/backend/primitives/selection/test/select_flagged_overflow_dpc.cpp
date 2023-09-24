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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/io.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/selection/select_flagged.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace de = dal::detail;

template <typename TestType>
class select_flagged_overflow_test : public te::float_algo_fixture<TestType> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Flag = std::tuple_element_t<1, TestType>;

    auto allocate_dummy_arrays(std::int64_t dummy_elem_count) {
        auto policy = de::data_parallel_policy{ this->get_queue() };
        Float* val_ptr = de::data_parallel_allocator<Float>{ policy }.allocate(elem_count_);
        Flag* mask_ptr = detail::data_parallel_allocator<Flag>{ policy }.allocate(elem_count_);

        auto val = ndarray<Float, 1>::wrap(val_ptr,
                                           { dummy_elem_count },
                                           detail::make_default_delete<Float>(policy));
        auto mask = ndarray<Flag, 1>::wrap(mask_ptr,
                                           { dummy_elem_count },
                                           detail::make_default_delete<Flag>(policy));

        return std::make_tuple(val, mask);
    }

    void check_select(ndarray<Float, 1>& in, ndarray<Flag, 1>& mask) {
        auto policy = de::data_parallel_policy{ this->get_queue() };
        Float* out_ptr = de::data_parallel_allocator<Float>{ policy }.allocate(elem_count_);

        auto out = ndarray<Float, 1>::wrap(out_ptr,
                                           { in.get_count() },
                                           detail::make_default_delete<Float>(policy));
        std::int64_t total_sum_res;

        select_flagged<Float, Flag>{ this->get_queue() }(mask, in, out, total_sum_res)
            .wait_and_throw();
    }

    std::int64_t get_value_to_overflow() {
        return 0x7FFFFFFFFFFFFFFF;
    }

private:
    const std::int64_t elem_count_ = 1;
};

template <typename TestType>
class select_flagged_index_overflow_test : public te::float_algo_fixture<TestType> {
public:
    using Data = std::tuple_element_t<0, TestType>;
    using Flag = std::tuple_element_t<1, TestType>;

    auto allocate_dummy_arrays(std::int64_t dummy_elem_count) {
        auto policy = de::data_parallel_policy{ this->get_queue() };
        Data* val_ptr = de::data_parallel_allocator<Data>{ policy }.allocate(elem_count_);
        Flag* mask_ptr = detail::data_parallel_allocator<Flag>{ policy }.allocate(elem_count_);

        auto val = ndarray<Data, 1>::wrap(val_ptr,
                                          { dummy_elem_count },
                                          detail::make_default_delete<Data>(policy));
        auto mask = ndarray<Flag, 1>::wrap(mask_ptr,
                                           { dummy_elem_count },
                                           detail::make_default_delete<Flag>(policy));

        return std::make_tuple(val, mask);
    }

    void check_select(ndarray<Data, 1>& in, ndarray<Flag, 1>& mask) {
        auto policy = de::data_parallel_policy{ this->get_queue() };
        Data* out_ptr = de::data_parallel_allocator<Data>{ policy }.allocate(elem_count_);

        auto out = ndarray<Data, 1>::wrap(out_ptr,
                                          { in.get_count() },
                                          detail::make_default_delete<Data>(policy));
        std::int64_t total_sum_res;

        select_flagged_index<Data, Flag>{ this->get_queue() }(mask, in, out, total_sum_res)
            .wait_and_throw();
    }

    std::int64_t get_value_to_overflow() {
        return 0x7FFFFFFFFFFFFFFF;
    }

private:
    const std::int64_t elem_count_ = 1;
};

using select_flagged_types = COMBINE_TYPES((float, double), (std::uint8_t, std::uint32_t));
using select_flagged_index_types = COMBINE_TYPES((std::int32_t, std::uint32_t),
                                                 (std::uint8_t, std::uint32_t));

#define SELECT_FLAGGED_OVERFLOW_TEST(name)             \
    TEMPLATE_LIST_TEST_M(select_flagged_overflow_test, \
                         name,                         \
                         "[select_flagged][overflow]", \
                         select_flagged_types)

#define SELECT_FLAGGED_INDEX_OVERFLOW_TEST(name)             \
    TEMPLATE_LIST_TEST_M(select_flagged_index_overflow_test, \
                         name,                               \
                         "[select_flagged][overflow]",       \
                         select_flagged_index_types)

SELECT_FLAGGED_OVERFLOW_TEST("select_flagged throws if element_count exceeds uint32") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    auto [val, mask] = this->allocate_dummy_arrays(this->get_value_to_overflow());
    REQUIRE_THROWS_AS(this->check_select(val, mask), domain_error);
}

SELECT_FLAGGED_INDEX_OVERFLOW_TEST("select_flagged_index throws if element_count exceeds uint32") {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());
    auto [val, mask] = this->allocate_dummy_arrays(this->get_value_to_overflow());
    REQUIRE_THROWS_AS(this->check_select(val, mask), domain_error);
}

} // namespace oneapi::dal::backend::primitives::test
