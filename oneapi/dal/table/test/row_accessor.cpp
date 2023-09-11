/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/linalg.hpp"

namespace oneapi::dal {

namespace te = dal::test::engine;
namespace la = te::linalg;

TEST("can read table data via row accessor") {
    using oneapi::dal::detail::empty_delete;

    double data[] = { 1.0, 2.0, 3.0, -1.0, -2.0, -3.0 };

    homogen_table t{ data, 2, 3, empty_delete<const double>() };
    const auto rows_block = row_accessor<const double>(t).pull({ 0, -1 });

    REQUIRE(t.get_row_count() * t.get_column_count() == rows_block.get_count());
    REQUIRE(data == rows_block.get_data());

    for (std::int64_t i = 0; i < rows_block.get_count(); i++) {
        REQUIRE(rows_block[i] == data[i]);
    }
}

TEST("can read table data via row accessor with conversion") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };

    homogen_table t{ data, 2, 3, empty_delete<const float>() };
    auto rows_block = row_accessor<const double>(t).pull({ 0, -1 });

    REQUIRE(t.get_row_count() * t.get_column_count() == rows_block.get_count());
    REQUIRE((void*)data != (void*)rows_block.get_data());

    for (std::int64_t i = 0; i < rows_block.get_count(); i++) {
        REQUIRE(rows_block[i] == Approx(static_cast<double>(data[i])));
    }
}

TEST("can read table data via row accessor and array outside") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };

    homogen_table t{ data, 2, 3, empty_delete<const float>() };
    auto arr = array<float>::empty(10);

    auto rows_ptr = row_accessor<const float>(t).pull(arr, { 0, -1 });

    REQUIRE(t.get_row_count() * t.get_column_count() == arr.get_count());

    REQUIRE(data == rows_ptr);
    REQUIRE(data == arr.get_data());

    auto data_ptr = arr.get_data();
    for (std::int64_t i = 0; i < arr.get_count(); i++) {
        REQUIRE(rows_ptr[i] == data[i]);
        REQUIRE(data_ptr[i] == data[i]);
    }
}

TEST("can read rows from column major table") {
    float data[] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };

    auto t = homogen_table::wrap(data, 3, 2, data_layout::column_major);

    auto rows_data = row_accessor<const float>(t).pull({ 1, -1 });

    REQUIRE(rows_data.get_count() == 2 * t.get_column_count());

    REQUIRE(rows_data[0] == Approx(2.0f));
    REQUIRE(rows_data[1] == Approx(-2.0f));
    REQUIRE(rows_data[2] == Approx(3.0f));
    REQUIRE(rows_data[3] == Approx(-3.0f));
}

TEST("can read rows from column major table with conversion") {
    float data[] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };

    auto t = homogen_table::wrap(data, 3, 2, data_layout::column_major);

    auto rows_data = row_accessor<const std::int32_t>(t).pull({ 1, 2 });

    REQUIRE(rows_data.get_count() == 1 * t.get_column_count());
    REQUIRE(rows_data[0] == 2);
    REQUIRE(rows_data[1] == -2);
}

TEST("pull returns immutable data from homogen_table") {
    constexpr std::int64_t row_count = 3;
    constexpr std::int64_t column_count = 2;
    float data[row_count * column_count] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };
    array<float> block;

    SECTION("pull from homogen_table created via raw pointer") {
        const auto t = homogen_table::wrap(data, row_count, column_count);
        block = row_accessor<const float>{ t }.pull();
    }

    SECTION("pull from homogen_table created via array") {
        const auto ary = array<float>::wrap(data, row_count * column_count);
        const auto t = homogen_table::wrap(ary, row_count, column_count);
        block = row_accessor<const float>{ t }.pull();
    }

    SECTION("pull from homogen_table created via builder") {
        const auto ary = array<float>::wrap(data, row_count * column_count);
        const auto t = detail::homogen_table_builder{}.reset(ary, row_count, column_count).build();
        block = row_accessor<const float>{ t }.pull();
    }

    REQUIRE(block.has_mutable_data() == false);
}

TEST("pull returns mutable data from homogen_table_builder") {
    constexpr std::int64_t row_count = 3;
    constexpr std::int64_t column_count = 2;
    float data[row_count * column_count] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };

    const auto ary = array<float>::wrap(data, row_count * column_count);
    const auto builder = detail::homogen_table_builder{}.reset(ary, row_count, column_count);
    const auto block = row_accessor<const float>{ builder }.pull();

    REQUIRE(block.has_mutable_data() == true);
}

TEST("pull does not copy if contigious block is requested") {
    constexpr std::int64_t row_count = 3;
    constexpr std::int64_t column_count = 2;
    float data[row_count * column_count] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };
    array<float> block;

    SECTION("pull from homogen_table") {
        auto t = homogen_table::wrap(data, row_count, column_count);
        block = row_accessor<const float>{ t }.pull();
    }

    SECTION("pull from homogen_table_builder") {
        auto ary = array<float>::wrap(data, row_count * column_count);
        auto builder = detail::homogen_table_builder{}.reset(ary, row_count, column_count);
        block = row_accessor<const float>{ builder }.pull();
    }

    REQUIRE(block.get_data() == data);
    REQUIRE(block.get_count() == row_count * column_count);
}

TEST("pull throws exception if invalid range") {
    float data[] = { 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f };
    auto t = homogen_table::wrap(data, 3, 2, data_layout::column_major);
    row_accessor<const float> acc{ t };

    REQUIRE_THROWS_AS(acc.pull({ 1, 4 }), dal::range_error);
}

TEST("push throws exception if invalid range") {
    detail::homogen_table_builder b;
    b.reset(array<float>::zeros(3 * 2), 3, 2);
    row_accessor<float> acc{ b };
    auto rows_data = acc.pull({ 1, 2 });

    REQUIRE_THROWS_AS(acc.push(rows_data, { 0, 2 }), dal::range_error);
    REQUIRE_THROWS_AS(acc.push(rows_data, { 3, 4 }), dal::range_error);
}

#ifdef ONEDAL_DATA_PARALLEL
TEST("pull with queue throws exception if invalid range") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    detail::homogen_table_builder b;
    b.reset(array<float>::zeros(q, 3 * 2), 3, 2);
    row_accessor<float> acc{ b };

    REQUIRE_THROWS_AS(acc.pull(q, { 1, 4 }), dal::range_error);
}

TEST("push with queue throws exception if invalid range") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    detail::homogen_table_builder b;
    b.reset(array<float>::zeros(q, 3 * 2), 3, 2);
    row_accessor<float> acc{ b };

    auto rows_data = acc.pull(q, { 1, 2 });
    REQUIRE_THROWS_AS(acc.push(q, rows_data, { 0, 2 }), dal::range_error);
    REQUIRE_THROWS_AS(acc.push(q, rows_data, { 3, 4 }), dal::range_error);
}

TEST("pull as device usm from host-allocated homogen table") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    const float data_ptr[] = { 1.0f,  2.0f, //
                               3.0f,  -1.0f, //
                               -2.0f, -3.0f };
    const std::int64_t row_count = 3;
    const std::int64_t column_count = 2;
    const auto data = homogen_table::wrap(data_ptr, row_count, column_count);

    const auto data_arr_device = //
        row_accessor<const float>{ data } //
            .pull(q, { 1, 3 }, sycl::usm::alloc::device);

    const auto data_arr_host = la::matrix<float>::wrap(data_arr_device).to_host();
    const float* data_arr_host_ptr = data_arr_host.get_data();

    REQUIRE(data_arr_host_ptr[0] == 3.0f);
    REQUIRE(data_arr_host_ptr[1] == -1.0f);
    REQUIRE(data_arr_host_ptr[2] == -2.0f);
    REQUIRE(data_arr_host_ptr[3] == -3.0f);
}

TEST("pull as host from device-allocated homogen table") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 3;
    const auto ary = la::matrix<float>::full(
        q,
        { row_count, column_count },
        [](std::int64_t i) {
            return float(i);
        },
        sycl::usm::alloc::device);
    const auto shared_table = homogen_table::wrap(q, ary.get_data(), row_count, column_count);

    const auto data_arr_host = //
        row_accessor<const float>{ shared_table }.pull({ 1, 3 });
    const float* data_arr_host_ptr = data_arr_host.get_data();

    REQUIRE(data_arr_host_ptr[0] == 3.0f);
    REQUIRE(data_arr_host_ptr[1] == 4.0f);
    REQUIRE(data_arr_host_ptr[2] == 5.0f);
    REQUIRE(data_arr_host_ptr[3] == 6.0f);
}

TEST("pull does not copy if alloc kind matches") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();
    const auto alloc_kind = GENERATE(sycl::usm::alloc::device, //
                                     sycl::usm::alloc::host,
                                     sycl::usm::alloc::shared);

    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 3;
    const auto ary = la::matrix<float>::full(
        q,
        { row_count, column_count },
        [](std::int64_t i) {
            return float(i);
        },
        alloc_kind);
    const auto table = homogen_table::wrap(q, ary.get_data(), row_count, column_count);

    const auto data_arr_device = //
        row_accessor<const float>{ table } //
            .pull(q, { 0, 2 }, alloc_kind);

    REQUIRE(data_arr_device.get_data() == ary.get_data());
}

TEST("pull does not copy if device usm requested from shared usm table") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 3;
    const auto shared_ary = la::matrix<float>::full(
        q,
        { row_count, column_count },
        [](std::int64_t i) {
            return float(i);
        },
        sycl::usm::alloc::shared);
    const auto shared_table =
        homogen_table::wrap(q, shared_ary.get_data(), row_count, column_count);

    const auto data_arr_device = //
        row_accessor<const float>{ shared_table } //
            .pull(q, { 0, 2 }, sycl::usm::alloc::device);

    REQUIRE(data_arr_device.get_data() == shared_ary.get_data());
    REQUIRE(sycl::get_pointer_type(data_arr_device.get_data(), q.get_context()) ==
            sycl::usm::alloc::shared);
}

TEST("pull from column-major shared usm homogen table") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 3;
    constexpr std::int64_t data_size = row_count * column_count;

    auto data = sycl::malloc_shared<float>(data_size, q);

    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
             data[idx[0]] = idx[0];
         });
     }).wait();

    auto t = homogen_table::wrap(q, data, row_count, column_count, {}, data_layout::column_major);
    row_accessor<const float> acc{ t };
    auto block = acc.pull(q, { 1, 3 });

    REQUIRE(block.get_count() == 2 * column_count);

    REQUIRE(block[0] == Approx(1));
    REQUIRE(block[1] == Approx(5));
    REQUIRE(block[2] == Approx(9));

    REQUIRE(block[3] == Approx(2));
    REQUIRE(block[4] == Approx(6));
    REQUIRE(block[5] == Approx(10));

    sycl::free(data, q);
}
#endif

} // namespace oneapi::dal
