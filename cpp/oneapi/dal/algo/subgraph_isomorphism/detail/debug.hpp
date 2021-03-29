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
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            std::cout << a[i * nCols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void pa_bit(char *str, std::uint8_t **a, size_t n);
void pa_bit8(char *str, std::uint8_t **a, size_t n);

#define PA(array_name, array_size)     pa(#array_name, array_name, array_size, 1);
#define PA_BIT(array_name, array_size) pa_bit(#array_name, array_name, array_size);