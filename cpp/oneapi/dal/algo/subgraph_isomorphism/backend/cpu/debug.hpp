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

#pragma once

#include <cstdint>
#include <iostream>

std::uint64_t to_byte(std::uint64_t x);
std::uint64_t to_bit(std::uint64_t x);

int get_bit_val(const std::uint8_t *array, int pos);

template <typename fpType>
void pa(char *str, const fpType *a, size_t nCols, size_t nRows = 1) {
    std::cout << str << std::endl;
    if (a == nullptr) {
        std::cout << "    NULL";
        return;
    }
    for (size_t i = 0; i < nRows; i++) {
        for (size_t j = 0; j < nCols; j++) {
            std::cout << a[i * nCols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void pa_bit(char *str, std::uint8_t **a, size_t n);
void pa_bit8(char *str, std::uint8_t **a, size_t n);

#define PA(array_name, array_size)     pa(#array_name, array_name, array_size, 1);
#define PA_BIT(array_name, array_size) pa_bit(#array_name, array_name, array_size);
