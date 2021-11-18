/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#pragma once

#include <cstdint>
#include <utility>

#if defined(_WIN32) || defined(_WIN64)
#ifdef __ONEDAL_ENABLE_DLL_EXPORT__
#define ONEDAL_EXPORT __declspec(dllexport)
#else
#define ONEDAL_EXPORT
#endif
#else
#define ONEDAL_EXPORT
#endif

#ifndef ONEDAL_ENABLE_ASSERT
#define ONEDAL_ASSERT(...)
#else
#include <cassert>
#include <iostream>
#define __ONEDAL_ASSERT_NO_MESSAGE__(condition) assert(condition)

#define __ONEDAL_ASSERT_MESSAGE__(condition, message) \
    do {                                              \
        if (!(condition)) {                           \
            std::cerr << (message) << std::endl;      \
            assert((condition));                      \
        }                                             \
    } while (0)

#define __ONEDAL_ASSERT_GET__(_1, _2, F, ...) F

#define ONEDAL_ASSERT(...)                                                                         \
    __ONEDAL_ASSERT_GET__(__VA_ARGS__, __ONEDAL_ASSERT_MESSAGE__, __ONEDAL_ASSERT_NO_MESSAGE__, 0) \
    (__VA_ARGS__)
#endif

namespace oneapi::dal {
namespace v1 {

using byte_t = std::uint8_t;

class base {
public:
    virtual ~base() = default;
};

enum class data_type {
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
    bfloat16
};

struct range {
public:
    range(std::int64_t start, std::int64_t end) : start_idx(start), end_idx(end) {}

    std::int64_t get_element_count(std::int64_t max_end_index) const noexcept {
        // TODO: handle error if (max_end_index + end_idx) < 0
        std::int64_t final_row = (end_idx < 0) ? max_end_index + end_idx + 1 : end_idx;
        return (final_row - start_idx - 1) + 1;
    }

    std::int64_t start_idx;
    std::int64_t end_idx;
};

} // namespace v1

using v1::byte_t;
using v1::base;
using v1::data_type;
using v1::range;

} // namespace oneapi::dal

namespace oneapi::dal::preview {

struct empty_value {};

template <typename T>
using range = std::pair<T, T>;

} // namespace oneapi::dal::preview
