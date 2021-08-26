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

#include "oneapi/dal/graph/detail/container.hpp"
#include "oneapi/dal/test/engine/common.hpp"

#include <vector>

namespace oneapi::dal::graph::test {
using namespace oneapi::dal::preview::detail;

template <class T>
struct CountingAllocator {
    typedef T value_type;
    typedef T *pointer;

    std::int64_t& currently_bytes_allocated;

    CountingAllocator(std::int64_t& currently_bytes_allocated)
            : currently_bytes_allocated(currently_bytes_allocated) {}

    template <class U>
    CountingAllocator(const CountingAllocator<U> &other): currently_bytes_allocated(other.currently_bytes_allocated) {
        std::cout << "CountingAllocator copy constructor" << std::endl;
        // currently_bytes_allocated = other.currently_bytes_allocated;
    }

    T *allocate(const size_t n) {
        std::cout << "allocated " << n * sizeof(T) << " bytes" << std::endl;
        currently_bytes_allocated += n * sizeof(T);
        if (n > static_cast<size_t>(-1) / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        void *const pv = malloc(n * sizeof(T));
        if (!pv) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(pv);
    }

    void deallocate(T *const p, size_t n) noexcept {
        std::cout << "deallocated " << n * sizeof(T) << " bytes" << std::endl;
        currently_bytes_allocated -= n * sizeof(T);
        free(p);
    }
};

std::int64_t lower_bound_pow(std::int64_t value){
    int n = 1;
    while(n < value){
        n <<= 1;
    }
    return n;
}

template< typename T>
void check_construct_empty(){
    vector_container<T> vec;

    REQUIRE(vec.empty());
    REQUIRE(vec.get_count() == 0);
    REQUIRE(vec.size() == 0);
    REQUIRE(vec.capacity() == 1);
}

template< typename T>
void check_construct_empty_with_custom_allocator(){
    std::int64_t count = 0;
    CountingAllocator<T> alloc(count);
    {
        vector_container<T, CountingAllocator<T>> vec(alloc);

        REQUIRE(vec.empty());
        REQUIRE(vec.get_count() == 0);
        REQUIRE(vec.size() == 0);
        REQUIRE(vec.capacity() == 1);
    }
    REQUIRE(alloc.currently_bytes_allocated == 0);
}

template<typename T>
void check_construct_given_size(std::int64_t n){
    vector_container<T> vec(n);

    REQUIRE(!vec.empty());
    REQUIRE(vec.get_count() == n);
    REQUIRE(vec.size() == n);
    REQUIRE(vec.capacity() == lower_bound_pow(n));
}

template<typename T>
void check_construct_given_size_with_custom_allocator(std::int64_t n){
    std::int64_t count = 0;
    CountingAllocator<T> alloc(count);
    {
        vector_container<T, CountingAllocator<T>> vec(n, alloc);

        REQUIRE(!vec.empty());
        REQUIRE(vec.get_count() == n);
        REQUIRE(vec.size() == n);
        REQUIRE(vec.capacity() == lower_bound_pow(n));
    }
    REQUIRE(alloc.currently_bytes_allocated == 0);
}

// template<typename T>
// void check_construct_given_size_with_value(std::int64_t n, const T& value){
//     vector_container<T> vec(n, value);

//     REQUIRE(!vec.empty());
//     REQUIRE(vec.get_count() == n);
//     REQUIRE(vec.size() == n);
//     REQUIRE(vec.capacity() == lower_bound_pow(n));
// }

template<typename T>
void check_construct_given_size_with_value_with_custom_allocator(std::int64_t n, const T& value){
    std::int64_t count = 0;
    CountingAllocator<T> alloc(count);
    {
        vector_container<T, CountingAllocator<T>> vec(n, value, alloc);

        REQUIRE(!vec.empty());
        REQUIRE(vec.get_count() == n);
        REQUIRE(vec.size() == n);
        REQUIRE(vec.capacity() == lower_bound_pow(n));
    }
    REQUIRE(alloc.currently_bytes_allocated == 0);
}

TEST("can construct empty vector_container of primitive type") {
    check_construct_empty<float>();
}

TEST("can construct empty vector_container of vector_container") {
    check_construct_empty<vector_container<float>>();
}

// TEST("can construct empty vector_container of vector_container of vector_container") {
//     check_construct_empty<vector_container<vector_container<float>>>();
// }

TEST("can construct empty vector_container of std::container") {
    check_construct_empty<std::vector<float>>();
}

TEST("can construct empty vector_container of primitive type with custom allocator") {
    check_construct_empty_with_custom_allocator<float>();
}

// Fails because of memory leak
TEST("can construct empty vector_container of vector_container with custom allocator") {
    std::cout << "start" << std::endl;
    check_construct_empty_with_custom_allocator<vector_container<float, CountingAllocator<float>>>();
}

TEST("can construct empty vector_container of std::container with custom allocator") {
    check_construct_empty_with_custom_allocator<std::vector<float>>();
}

TEST("can construct vector_container of given size of primitive type") {
    check_construct_given_size<float>(10);
}

TEST("can construct vector_container of given size of vector_container") {
    check_construct_given_size<vector_container<float>>(10);
}

TEST("can construct vector_container of given size of std::container") {
    check_construct_given_size<std::vector<float>>(10);
}

TEST("can construct vector_container of given size of primitive type with custom allocator") {
    check_construct_given_size_with_custom_allocator<float>(10);
}

// TEST("can construct vector_container of given size of vector_container with custom allocator") {
//     check_construct_given_size_with_custom_allocator<vector_container<float, CountingAllocator<float>>>(10);
// }

TEST("can construct vector_container of given size of std::container with custom allocator") {
    check_construct_given_size_with_custom_allocator<std::vector<float>>(10);
}

// TEST("can construct vector_container of given size with value of primitive type") {
//     check_construct_given_size_with_value<float>(10, 1);
// }

// TEST("can construct vector_container of given size with value of vector_container") {
//     vector_container<float> value(10);
//     check_construct_given_size_with_value<vector_container<float>>(10, value);
// }

// TEST("can construct vector_container of given size with value of std::container") {
//     std::vector<float> value(10);
//     check_construct_given_size_with_value<std::vector<float>>(10, value);
// }

TEST("can construct vector_container of given size with value of primitive type with custom allocator") {
    check_construct_given_size_with_value_with_custom_allocator<float>(10, 1);
}

// TEST("can construct vector_container of given size with value of vector_container with custom allocator") {
//     std::int64_t currently_bytes_allocated = 0;
//     CountingAllocator<float> alloc(currently_bytes_allocated);
//     {
//         vector_container<float, CountingAllocator<float>> value(10, 1, alloc);
//         check_construct_given_size_with_value_with_custom_allocator<vector_container<float, CountingAllocator<float>>>(10, value);
//     }
//     REQUIRE(alloc.currently_bytes_allocated == 0);
// }

TEST("can construct vector_container of given size with value of std::container with custom allocator") {
    std::vector<float> value(10);
    check_construct_given_size_with_value_with_custom_allocator<std::vector<float>>(10, value);
}

} // namespace oneapi::dal::graph::test
