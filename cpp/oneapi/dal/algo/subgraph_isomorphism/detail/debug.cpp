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
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
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
