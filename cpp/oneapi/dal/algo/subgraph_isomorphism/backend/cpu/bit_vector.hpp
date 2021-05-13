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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/gcc_adapt.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/inner_alloc.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph_status.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/detail/common.hpp"

#if defined(__INTEL_COMPILER)
#define ONEAPI_RESTRICT
//restrict
#else
#define ONEAPI_RESTRICT
#endif

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

enum register_size { r8 = 1, r16 = 2, r32 = 4, r64 = 8, r128 = 16, r256 = 32, r512 = 64 };

void inversion(std::uint8_t* ONEAPI_RESTRICT vec, const std::int64_t size);
void or_equal(std::uint8_t* ONEAPI_RESTRICT vec,
              const std::uint8_t* ONEAPI_RESTRICT pa,
              const std::int64_t size);
void and_equal(std::uint8_t* ONEAPI_RESTRICT vec,
               const std::uint8_t* ONEAPI_RESTRICT pa,
               const std::int64_t size);

void or_equal(std::uint8_t* ONEAPI_RESTRICT vec,
              const std::int64_t* ONEAPI_RESTRICT bit_index,
              const std::int64_t list_size);
void and_equal(std::uint8_t* ONEAPI_RESTRICT vec,
               const std::int64_t* ONEAPI_RESTRICT bit_index,
               const std::int64_t bit_size,
               const std::int64_t list_size,
               std::int64_t* ONEAPI_RESTRICT tmp_array = nullptr,
               const std::int64_t tmp_size = 0);
void set(std::uint8_t* ONEAPI_RESTRICT vec, std::int64_t size, const std::uint8_t byte_val = 0x0);

class bit_vector {
public:
    // precomputed count of ones in a number from 0 to 255
    // e.g. bit_set_table[2] = 1, because of 2 is 0x00000010
    // e.g. bit_set_table[7] = 7, because of 7 is 0x00000111
    static constexpr std::uint8_t bit_set_table[256] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3,
        4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
        4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4,
        5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5,
        4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
        3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
        5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
        5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
    };

    static constexpr std::int64_t byte(std::int64_t x) {
        return x >> 3;
    };

    static constexpr std::uint8_t bit(std::int64_t x) {
        return 1 << (x & 7);
    };

    static constexpr std::int64_t bit_vector_size(std::int64_t vertex_count) {
        return -~(vertex_count >> 3); /* (vertex_count / 8) + 1 */
    };

    static constexpr std::uint8_t power_of_two(const std::uint8_t bit_val) {
        return 31 - ONEDAL_lzcnt_u32(bit_val);
    };

    static void split_by_registers(const std::int64_t vector_size,
                                   std::int64_t* registers,
                                   const register_size max_size);

    bit_vector(const inner_alloc allocator);
    bit_vector(bit_vector& bvec);
    bit_vector(const std::int64_t vector_size, inner_alloc allocator);
    bit_vector(const std::int64_t vector_size, const std::uint8_t byte_val, inner_alloc allocator);
    bit_vector(const std::int64_t vector_size, std::uint8_t* pvector, inner_alloc allocator);
    virtual ~bit_vector();
    graph_status unset_bit(const std::int64_t vertex);
    graph_status set_bit(const std::int64_t vertex);
    void set(const std::uint8_t byte_val = 0x0);
    std::uint8_t* get_vector_pointer();
    std::int64_t size() const;
    std::int64_t popcount() const;

    bit_vector& operator&=(bit_vector& a);
    bit_vector& operator|=(bit_vector& a);
    bit_vector& operator^=(bit_vector& a);
    bit_vector& operator=(bit_vector& a);
    bit_vector& operator~();
    bit_vector& operator&=(const std::uint8_t* pa);
    bit_vector& operator|=(const std::uint8_t* pa);
    bit_vector& operator^=(const std::uint8_t* pa);
    bit_vector& operator=(const std::uint8_t* pa);

    bit_vector& or_equal(const std::int64_t* bit_index, const std::int64_t list_size);
    bit_vector& and_equal(const std::int64_t* bit_index,
                          const std::int64_t list_size,
                          std::int64_t* tmp_array = nullptr,
                          const std::int64_t tmp_size = 0);

    bit_vector(bit_vector&& a);
    bit_vector& operator=(bit_vector&& a);

    bit_vector& andn(const std::uint8_t* pa);
    bit_vector& andn(bit_vector& a);
    bit_vector& orn(const std::uint8_t* pa);
    bit_vector& orn(bit_vector& a);

    std::int64_t min_index() const;
    std::int64_t max_index() const;

    graph_status get_vertices_ids(std::int64_t* vertices_ids);
    bool test_bit(const std::int64_t vertex);

    static graph_status set(const std::int64_t vector_size,
                            std::uint8_t* result_vector,
                            const std::uint8_t byte_val = 0);
    static graph_status set(const std::int64_t vector_size,
                            std::uint8_t* result_vector,
                            const std::uint8_t* vector);
    static graph_status unset_bit(std::uint8_t* result_vector, const std::int64_t vertex);
    static graph_status set_bit(std::uint8_t* result_vector, const std::int64_t vertex);
    static std::int64_t get_bit_index(const std::int64_t vector_size,
                                      const std::uint8_t* vector,
                                      const std::int64_t vertex);
    static std::int64_t popcount(const std::int64_t vector_size, const std::uint8_t* vector);
    static bool test_bit(const std::int64_t vector_size,
                         const std::uint8_t* vector,
                         const std::int64_t vertex);

private:
    inner_alloc allocator_;
    std::uint8_t* vector;
    std::int64_t n;
};

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail