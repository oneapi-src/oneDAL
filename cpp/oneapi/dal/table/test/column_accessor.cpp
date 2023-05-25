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
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::test {

TEST("Column accessor host test: Can get the first column from homogen_table") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.f, 2.f, //
                     3.f, 4.f, //
                     5.f, 6.f, //
                     7.f, 8.f };

    homogen_table t{ data, 4, 2, empty_delete<const float>() };
    column_accessor<const float> acc{ t };
    auto col = acc.pull(0);

    REQUIRE(col.get_count() == t.get_row_count());
    REQUIRE(col.has_mutable_data());

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        REQUIRE(col[i] == t.get_data<float>()[i * t.get_column_count()]);
    }
}

TEST("Column accessor host test: Can get the second column from homogen_table with conversion") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };

    homogen_table t{ data, 4, 2, empty_delete<const float>() };
    column_accessor<const double> acc{ t };
    auto col = acc.pull(1);

    REQUIRE(col.get_count() == t.get_row_count());
    REQUIRE(col.has_mutable_data());

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        REQUIRE(col[i] == double(t.get_data<float>()[i * t.get_column_count() + 1]));
    }
}

TEST("Column accessor host test: Can get the first column from homogen_table with subset of rows") {
    using oneapi::dal::detail::empty_delete;

    float data[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };

    homogen_table t{ data, 4, 2, empty_delete<const float>() };
    column_accessor<const float> acc{ t };
    auto col = acc.pull(0, { 1, 3 });

    REQUIRE(col.get_count() == 2);
    REQUIRE(col.has_mutable_data());

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        REQUIRE(col[i] == t.get_data<float>()[2 + i * t.get_column_count()]);
    }
}

TEST("Column accessor host test: Can get columns from homogen_table_builder") {
    detail::homogen_table_builder b;
    b.reset(array<float>::zeros(3 * 2), 3, 2);
    {
        column_accessor<double> acc{ b };
        for (std::int64_t col_idx = 0; col_idx < 2; col_idx++) {
            auto col = acc.pull(col_idx);

            REQUIRE(col.get_count() == 3);
            col.need_mutable_data();
            double* col_data = col.get_mutable_data();
            for (std::int64_t i = 0; i < col.get_count(); i++) {
                REQUIRE(col[i] == 0.0);
                col_data[i] = col_idx + 1;
            }

            acc.push(col, col_idx);
        }
    }

    auto t = b.build();
    {
        column_accessor<const float> acc{ t };
        for (std::int64_t col_idx = 0; col_idx < 2; col_idx++) {
            const auto col = acc.pull(col_idx);

            REQUIRE(col.get_count() == 3);
            for (std::int64_t i = 0; i < col.get_count(); i++) {
                REQUIRE(col[i] == col_idx + 1);
            }
        }
    }
}

TEST("Column accessor host test: Can get column values from column major homogen_table") {
    float data[] = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f };

    auto t = homogen_table::wrap(data, 4, 3, data_layout::column_major);
    column_accessor<const float> acc{ t };
    auto col = acc.pull(1, { 1, 3 });

    REQUIRE(col.get_count() == 2);
    REQUIRE(col.get_data() == &data[5]);

    REQUIRE(col[0] == 5.f);
    REQUIRE(col[1] == 6.f);
}

TEST(
    "Column accessor host test: Can get column values from major column homogen_table with conversion") {
    float data[] = { 0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f };

    auto t = homogen_table::wrap(data, 4, 3, data_layout::column_major);
    column_accessor<const std::int32_t> acc{ t };
    auto col = acc.pull(1, { 1, 3 });

    REQUIRE(col.get_count() == 2);

    REQUIRE(col[0] == 5);
    REQUIRE(col[1] == 6);
}

TEST("Column accessor host test: Invalid range") {
    detail::homogen_table_builder b;
    b.reset(array<float>::zeros(3 * 2), 3, 2);
    column_accessor<float> acc{ b };

    REQUIRE_THROWS_AS(acc.pull(0, { 1, 4 }), dal::range_error);
    REQUIRE_THROWS_AS(acc.pull(2, { 1, 2 }), dal::range_error);

    auto column_data = acc.pull(0, { 1, 2 });
    REQUIRE_THROWS_AS(acc.push(column_data, 0, { 0, 2 }), dal::range_error);
    REQUIRE_THROWS_AS(acc.push(column_data, 0, { 3, 4 }), dal::range_error);
    REQUIRE_THROWS_AS(acc.push(column_data, 2, { 1, 2 }), dal::range_error);
}

#ifdef ONEDAL_DATA_PARALLEL
TEST("Column accessor DPC test: Can get the first column from homogen_table") {
    sycl::queue q;
    constexpr std::int64_t data_size = 8;
    auto data = sycl::malloc_shared<float>(data_size, q);

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
            data[idx[0]] = idx[0] + 1;
        });
    });

    homogen_table t{ q, data, 4, 2, detail::make_default_delete<const float>(q), { event } };
    column_accessor<const float> acc{ t };
    auto col = acc.pull(q, 0);

    REQUIRE(col.get_count() == t.get_row_count());
    REQUIRE(col.has_mutable_data());
    REQUIRE(sycl::get_pointer_type(col.get_data(), q.get_context()) == sycl::usm::alloc::shared);

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        REQUIRE(col[i] == i * 2 + 1);
    }
}

TEST("Column accessor DPC test: Can get the second column from homogen_table with conversion") {
    sycl::queue q;
    constexpr std::int64_t data_size = 8;
    auto data = sycl::malloc_shared<float>(data_size, q);

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
            data[idx[0]] = idx[0] + 1;
        });
    });

    homogen_table t{ q, data, 4, 2, detail::make_default_delete<const float>(q), { event } };
    column_accessor<const std::int32_t> acc{ t };
    auto col = acc.pull(q, 1);

    REQUIRE(col.get_count() == t.get_row_count());
    REQUIRE(col.has_mutable_data());
    REQUIRE(sycl::get_pointer_type(col.get_data(), q.get_context()) == sycl::usm::alloc::shared);

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        REQUIRE(col[i] == i * 2 + 2);
    }
}

TEST("Column accessor DPC test: Can get the first column from homogen_table with subset of rows") {
    sycl::queue q;
    constexpr std::int64_t data_size = 8;
    auto data = sycl::malloc_shared<float>(data_size, q);

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
            data[idx[0]] = idx[0] + 1;
        });
    });

    homogen_table t{ q, data, 4, 2, detail::make_default_delete<const float>(q), { event } };
    column_accessor<const float> acc{ t };
    auto col = acc.pull(q, 0, { 1, 3 });

    REQUIRE(col.get_count() == 2);
    REQUIRE(col.has_mutable_data());
    REQUIRE(sycl::get_pointer_type(col.get_data(), q.get_context()) == sycl::usm::alloc::shared);

    for (std::int64_t i = 0; i < col.get_count(); i++) {
        REQUIRE(Approx(col[i]) == t.get_data<float>()[2 + i * t.get_column_count()]);
    }
}

TEST("Column accessor DPC test: Can get columns from homogen_table_builder") {
    sycl::queue q;

    detail::homogen_table_builder b;
    b.reset(array<float>::zeros(q, 3 * 2), 3, 2);
    {
        column_accessor<std::int32_t> acc{ b };
        for (std::int64_t col_idx = 0; col_idx < 2; col_idx++) {
            auto col = acc.pull(q, col_idx);

            REQUIRE(col.get_count() == 3);
            REQUIRE(sycl::get_pointer_type(col.get_data(), q.get_context()) ==
                    sycl::usm::alloc::shared);

            col.need_mutable_data();
            auto col_data = col.get_mutable_data();
            for (std::int64_t i = 0; i < col.get_count(); i++) {
                REQUIRE(col_data[i] == 0);
                col_data[i] = col_idx + 1;
            }

            acc.push(q, col, col_idx);
        }
    }

    auto t = b.build();
    {
        column_accessor<const float> acc{ t };
        for (std::int64_t col_idx = 0; col_idx < 2; col_idx++) {
            const auto col = acc.pull(q, col_idx);

            REQUIRE(col.get_count() == 3);
            REQUIRE(sycl::get_pointer_type(col.get_data(), q.get_context()) ==
                    sycl::usm::alloc::shared);

            for (std::int64_t i = 0; i < col.get_count(); i++) {
                REQUIRE(Approx(col[i]) == col_idx + 1);
            }
        }
    }
}

TEST("Column accessor DPC test: Can get column values from column major homogen_table") {
    sycl::queue q;
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 3;
    constexpr std::int64_t data_size = row_count * column_count;

    auto data = sycl::malloc_shared<float>(data_size, q);

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
            data[idx[0]] = idx[0];
        });
    });

    auto t =
        homogen_table::wrap(q, data, row_count, column_count, { event }, data_layout::column_major);
    column_accessor<const float> acc{ t };
    auto col = acc.pull(q, 1, { 1, 3 });

    REQUIRE(col.get_count() == 2);
    REQUIRE(col.get_data() == &data[5]);

    REQUIRE(col[0] == Approx(5.f));
    REQUIRE(col[1] == Approx(6.f));

    sycl::free(data, q);
}

TEST(
    "Column accessor DPC test: Can get column values from major column homogen_table with conversion") {
    sycl::queue q;
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 3;
    constexpr std::int64_t data_size = row_count * column_count;

    auto data = sycl::malloc_shared<float>(data_size, q);

    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
            data[idx[0]] = idx[0];
        });
    });

    auto t =
        homogen_table::wrap(q, data, row_count, column_count, { event }, data_layout::column_major);
    column_accessor<const std::int32_t> acc{ t };
    auto col = acc.pull(q, 1, { 1, 3 });

    REQUIRE(col.get_count() == 2);

    REQUIRE(col[0] == 5);
    REQUIRE(col[1] == 6);

    sycl::free(data, q);
}

TEST("Column accessor DPC test: Invalid range") {
    sycl::queue q;

    detail::homogen_table_builder b;
    b.reset(array<float>::zeros(q, 3 * 2), 3, 2);
    column_accessor<float> acc{ b };

    REQUIRE_THROWS_AS(acc.pull(q, 0, { 1, 4 }), dal::range_error);
    REQUIRE_THROWS_AS(acc.pull(q, 2, { 1, 2 }), dal::range_error);

    auto column_data = acc.pull(q, 0, { 1, 2 });
    REQUIRE_THROWS_AS(acc.push(q, column_data, 0, { 0, 2 }), dal::range_error);
    REQUIRE_THROWS_AS(acc.push(q, column_data, 0, { 3, 4 }), dal::range_error);
    REQUIRE_THROWS_AS(acc.push(q, column_data, 2, { 1, 2 }), dal::range_error);
}
#endif

} // namespace oneapi::dal::test
