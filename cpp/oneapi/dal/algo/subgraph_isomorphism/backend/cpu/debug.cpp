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

#include "debug.hpp"

std::uint64_t to_byte(std::uint64_t x) {
    return x / 8;
}
std::uint64_t to_bit(std::uint64_t x) {
    return x % 8;
}

int get_bit_val(const std::uint8_t *array, int pos) {
    return (array[to_byte(pos)] >> (to_bit(pos))) & 1;
}

void pa_bit(char *str, std::uint8_t **a, size_t n) {
    std::cout << "BIT" << std::endl;
    std::cout << str << std::endl;
    if (a == nullptr) {
        std::cout << "    NULL";
        return;
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            std::cout << get_bit_val(a[i], j);
        }
        std::cout << std::endl;
    }
}

void pa_bit8(char *str, std::uint8_t **a, size_t n) {
    std::cout << "BIT" << std::endl;
    std::cout << str << std::endl;
    if (a == nullptr) {
        std::cout << "    NULL";
        return;
    }
    int n_bytes = to_byte(n);
    for (int i = 0; i < n_bytes; i++) {
        for (int j = 0; j < n_bytes; j++) {
            std::cout << (int)a[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
