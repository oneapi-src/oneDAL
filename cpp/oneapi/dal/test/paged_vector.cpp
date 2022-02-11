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

#include "oneapi/dal/detail/paged_vector.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test {

TEST("can construct empty paged_vector_page", "[paged_vector_page]") {
    const std::int64_t page_size = 10;
    detail::paged_vector_page<byte_t> page{ page_size };

    REQUIRE(page.get_count() == 0);
    REQUIRE(page.get_next() == nullptr);
    REQUIRE(page.get_data() == nullptr);
}

TEST("can write elements to paged_vector_page", "[paged_vector_page]") {
    const std::int64_t page_size = 10;
    constexpr std::int64_t count = 10;
    const std::array<byte_t, count> data = { 4, 5, 1, 3, 2, 9, 8, 7, 1, 6 };
    detail::paged_vector_page<byte_t> page{ page_size };

    for (std::int64_t i = 0; i < count; i++) {
        REQUIRE(page.try_write(&data[i], 1) == true);
        REQUIRE(page.get_count() == i + 1);
    }

    REQUIRE(page.get_count() == count);
    for (std::int64_t i = 0; i < count; i++) {
        REQUIRE(page.get_data()[i] == data[i]);
    }
}

TEST("paged_vector_page::try_write returns false if no space left", "[paged_vector_page]") {
    const std::int64_t page_size = 10;
    constexpr std::int64_t count = 10;
    const std::array<byte_t, count> data = { 4, 5, 1, 3, 2, 9, 8, 7, 1, 6 };
    detail::paged_vector_page<byte_t> page{ page_size };

    REQUIRE(page.try_write(data.data(), count) == true);
    REQUIRE(page.try_write(data.data(), 1) == false);
    REQUIRE(page.get_count() == count);
}

TEST("paged_vector_page::try_write increases page size if page is empty", "[paged_vector_page]") {
    const std::int64_t page_size = 5;
    constexpr std::int64_t count = 10;
    const std::array<byte_t, count> data = { 4, 5, 1, 3, 2, 9, 8, 7, 1, 6 };
    detail::paged_vector_page<byte_t> page{ page_size };

    REQUIRE(page.get_capacity() == page_size);
    REQUIRE(page.try_write(data.data(), count) == true);
    REQUIRE(page.get_capacity() == count);
    REQUIRE(page.get_count() == count);
}

TEST("can create empty paged_vector", "[paged_vector]") {
    const std::int64_t page_size = 10;
    detail::paged_vector<byte_t> vector{ page_size };

    REQUIRE(vector.get_count() == 0);
    REQUIRE(vector.get_page_count() == 1);
    REQUIRE(vector.to_array().get_count() == 0);
}

TEST("paged_vector::push_back handles zero data", "[paged_vector]") {
    const std::int64_t page_size = 10;
    detail::paged_vector<byte_t> vector{ page_size };

    vector.push_back(nullptr, 0);

    REQUIRE(vector.get_count() == 0);
}

TEST("paged_vector::push_back does not lead to new page creation if page size is enough",
     "[paged_vector]") {
    constexpr std::int64_t count = 10;
    const std::int64_t page_size = 10;
    detail::paged_vector<byte_t> vector{ page_size };
    const std::array<byte_t, count> data = { 4, 5, 1, 3, 2, 9, 8, 7, 1, 6 };

    for (std::int64_t i = 0; i < count; i++) {
        vector.push_back(&data[i], 1);
        REQUIRE(vector.get_page_count() == 1);
    }
}

TEST("paged_vector::push_back creates new page if element count exceeds page size",
     "[paged_vector]") {
    constexpr std::int64_t count = 10;
    const std::int64_t page_size = 2;
    detail::paged_vector<byte_t> vector{ page_size };
    const std::array<byte_t, count> data = { 4, 5, 1, 3, 2, 9, 8, 7, 1, 6 };

    for (std::int64_t i = 0; i < count; i++) {
        vector.push_back(&data[i], 1);
        REQUIRE(vector.get_page_count() == i / page_size + 1);
    }
}

TEST("paged_vector is converted to array", "[paged_vector]") {
    constexpr std::int64_t count = 10;
    const std::int64_t page_size = 2;
    detail::paged_vector<byte_t> vector{ page_size };
    const std::array<byte_t, count> data = { 4, 5, 1, 3, 2, 9, 8, 7, 1, 6 };

    for (std::int64_t i = 0; i < count; i++) {
        vector.push_back(&data[i], 1);
    }

    const auto ary = vector.to_array();

    REQUIRE(ary.get_count() == count);
    for (std::int64_t i = 0; i < count; i++) {
        REQUIRE(ary[i] == data[i]);
    }
}

TEST("can reset paged_vector", "[paged_vector]") {
    constexpr std::int64_t count = 10;
    const std::int64_t page_size = 2;
    detail::paged_vector<byte_t> vector{ page_size };
    const std::array<byte_t, count> data = { 4, 5, 1, 3, 2, 9, 8, 7, 1, 6 };

    for (std::int64_t i = 0; i < count; i++) {
        vector.push_back(&data[i], 1);
    }

    vector.reset();

    REQUIRE(vector.get_count() == 0);
    REQUIRE(vector.get_page_count() == 1);
    REQUIRE(vector.to_array().get_count() == 0);
}

} // namespace oneapi::dal::test
