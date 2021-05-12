#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/bit_vector.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

void or_equal(std::uint8_t* ONEAPI_RESTRICT vec,
              const std::uint8_t* ONEAPI_RESTRICT pa,
              std::int64_t size) {
    ONEDAL_IVDEP
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] |= pa[i];
    }
}
void and_equal(std::uint8_t* ONEAPI_RESTRICT vec,
               const std::uint8_t* ONEAPI_RESTRICT pa,
               std::int64_t size) {
    ONEDAL_IVDEP
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] &= pa[i];
    }
}

void inversion(std::uint8_t* ONEAPI_RESTRICT vec, std::int64_t size) {
    ONEDAL_IVDEP
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] = ~vec[i];
    }
}

void or_equal(std::uint8_t* ONEAPI_RESTRICT vec,
              const std::int64_t* ONEAPI_RESTRICT bit_index,
              const std::int64_t list_size) {
    ONEDAL_IVDEP
    for (std::int64_t i = 0; i < list_size; i++) {
        vec[bit_vector::byte(bit_index[i])] |= bit_vector::bit(bit_index[i]);
    }
}

void set(std::uint8_t* ONEAPI_RESTRICT vec, std::int64_t size, const std::uint8_t byte_val) {
    ONEDAL_VECTOR_ALWAYS
    for (std::int64_t i = 0; i < size; i++) {
        vec[i] = byte_val;
    }
}

void and_equal(std::uint8_t* ONEAPI_RESTRICT vec,
               const std::int64_t* ONEAPI_RESTRICT bit_index,
               const std::int64_t bit_size,
               const std::int64_t list_size,
               std::int64_t* ONEAPI_RESTRICT tmp_array,
               const std::int64_t tmp_size) {
    std::int64_t counter = 0;
    ONEDAL_IVDEP
    for (std::int64_t i = 0; i < list_size; i++) {
        tmp_array[counter] = bit_index[i];
        counter += bit_vector::bit_set_table[vec[bit_vector::byte(bit_index[i])] &
                                             bit_vector::bit(bit_index[i])];
    }

    set(vec, bit_size, 0x0);

    ONEDAL_IVDEP
    for (std::int64_t i = 0; i < counter; i++) {
        vec[bit_vector::byte(tmp_array[i])] |= bit_vector::bit(tmp_array[i]);
    }
}

graph_status bit_vector::set(const std::int64_t vector_size,
                             std::uint8_t* result_vector,
                             const std::uint8_t* vector) {
    if (result_vector == nullptr || vector == nullptr) {
        return bad_allocation;
    }

    for (std::int64_t i = 0; i < vector_size; i++) {
        result_vector[i] = vector[i];
    }

    /*
      TODO improvements  use register and sets functions
    */

    return ok;
}

graph_status bit_vector::set(const std::int64_t vector_size,
                             std::uint8_t* result_vector,
                             const std::uint8_t byte_val) {
    if (result_vector == nullptr) {
        return bad_allocation;
    }

    for (std::int64_t i = 0; i < vector_size; i++) {
        result_vector[i] = byte_val;
    }

    /*
        TODO improvements  use register and sets functions
    */

    return ok;
}

void bit_vector::set(const std::uint8_t byte_val) {
    const std::int64_t nn = n;
    ONEDAL_VECTOR_ALWAYS
    for (std::int64_t i = 0; i < nn; i++) {
        vector[i] = byte_val;
    }
}

graph_status bit_vector::set_bit(std::uint8_t* result_vector, const std::int64_t vertex) {
    if (result_vector == nullptr) {
        return bad_allocation;
    }
    result_vector[byte(vertex)] |= bit(vertex);
    return ok;
}

graph_status bit_vector::unset_bit(std::uint8_t* result_vector, const std::int64_t vertex) {
    if (result_vector == nullptr) {
        return bad_allocation;
    }
    result_vector[byte(vertex)] &= ~bit(vertex);
    return ok;
}

bit_vector::bit_vector(const inner_alloc allocator) : allocator_(allocator) {
    vector = nullptr;
    n = 0;
}

bit_vector::bit_vector(bit_vector& bvec) : allocator_(bvec.allocator_.get_byte_allocator()) {
    n = bvec.n;
    vector = allocator_.allocate<std::uint8_t>(n);
    set(n, vector, bvec.vector);
}

bit_vector::bit_vector(const std::int64_t vector_size, inner_alloc allocator)
        : bit_vector(allocator) {
    n = vector_size;
    vector = allocator_.allocate<std::uint8_t>(n);
    set(n, vector);
}

bit_vector::bit_vector(const std::int64_t vector_size,
                       const std::uint8_t byte_val,
                       inner_alloc allocator)
        : allocator_(allocator) {
    n = vector_size;
    vector = allocator_.allocate<std::uint8_t>(n);
    set(n, vector, byte_val);
}

bit_vector::bit_vector(bit_vector&& a)
        : allocator_(a.allocator_.get_byte_allocator()),
          vector(a.vector) {
    n = a.n;
    a.n = 0;
    a.vector = nullptr;
}

bit_vector& bit_vector::operator=(bit_vector&& a) {
    if (&a == this) {
        return *this;
    }
    vector = a.vector;
    n = a.n;

    a.vector = nullptr;
    a.n = 0;
    return *this;
}

bit_vector::~bit_vector() {
    if (vector != nullptr) {
        allocator_.deallocate<std::uint8_t>(vector, n);
        vector = nullptr;
        n = 0;
    }
}

bit_vector::bit_vector(const std::int64_t vector_size, std::uint8_t* pvector, inner_alloc allocator)
        : allocator_(allocator) {
    n = vector_size;
    vector = pvector;
    pvector = nullptr;
}

graph_status bit_vector::unset_bit(const std::int64_t vertex) {
    return unset_bit(vector, vertex);
}

graph_status bit_vector::set_bit(const std::int64_t vertex) {
    return set_bit(vector, vertex);
}

std::uint8_t* bit_vector::get_vector_pointer() {
    return vector;
}

std::int64_t bit_vector::size() const {
    return n;
}

bit_vector& bit_vector::operator&=(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] &= pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::operator|=(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] |= pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::operator^=(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] ^= pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::operator=(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] = pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::operator~() {
    const std::int64_t nn = n;
    ONEDAL_IVDEP
    for (std::int64_t i = 0; i < nn; i++) {
        vector[i] = ~vector[i];
    }
    return *this;
}

bit_vector& bit_vector::operator&=(const std::uint8_t* ONEAPI_RESTRICT pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] &= pa[i];
    }
    return *this;
}

bit_vector& bit_vector::operator|=(const std::uint8_t* ONEAPI_RESTRICT pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] |= pa[i];
    }
    return *this;
}

bit_vector& bit_vector::operator^=(const std::uint8_t* pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] ^= pa[i];
    }
    return *this;
}

bit_vector& bit_vector::operator=(const std::uint8_t* pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] = pa[i];
    }
    return *this;
}

bit_vector& bit_vector::andn(const std::uint8_t* pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] &= ~pa[i];
    }
    return *this;
}

bit_vector& bit_vector::andn(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] &= ~pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::orn(const std::uint8_t* pa) {
    for (std::int64_t i = 0; i < n; i++) {
        vector[i] |= ~pa[i];
    }
    return *this;
}

bit_vector& bit_vector::orn(bit_vector& a) {
    if (n <= a.size()) {
        std::uint8_t* pa = a.get_vector_pointer();
        for (std::int64_t i = 0; i < n; i++) {
            vector[i] |= ~pa[i];
        }
    }
    return *this;
}

bit_vector& bit_vector::or_equal(const std::int64_t* bit_index, const std::int64_t list_size) {
    for (std::int64_t i = 0; i < list_size; i++) {
        vector[byte(bit_index[i])] |= bit(bit_index[i]);
    }
    return *this;
}

bit_vector& bit_vector::and_equal(const std::int64_t* bit_index,
                                  const std::int64_t list_size,
                                  std::int64_t* tmp_array,
                                  const std::int64_t tmp_size) {
    std::int64_t counter = 0;
    for (std::int64_t i = 0; i < list_size; i++) {
        tmp_array[counter] = bit_index[i];
        counter += bit_set_table[vector[byte(bit_index[i])] & bit(bit_index[i])];
    }

    set(0x0);
    for (std::int64_t i = 0; i < counter; i++) {
        vector[byte(tmp_array[i])] |= bit(tmp_array[i]);
    }
    return *this;
}

graph_status bit_vector::get_vertices_ids(std::int64_t* vertices_ids) {
    std::int64_t vertex_iterator = 0;
    if (vertices_ids == nullptr) {
        return bad_arguments;
    }
    std::uint8_t current_byte = 0;
    std::uint8_t current_bit = 0;
    for (std::int64_t i = 0; i < n; i++) {
        current_byte = vector[i];
        while (current_byte > 0) {
            current_bit = current_byte & (-current_byte);
            current_byte &= ~current_bit;
            vertices_ids[vertex_iterator] = 8 * i + power_of_two(current_bit);
            vertex_iterator++;
        }
    }
    return static_cast<graph_status>(vertex_iterator);
}

bool bit_vector::test_bit(const std::int64_t vertex) {
    return vector[byte(vertex)] & bit(vertex);
}

bool bit_vector::test_bit(const std::int64_t vector_size,
                          const std::uint8_t* vector,
                          const std::int64_t vertex) {
    if (vertex / 8 > vector_size) {
        return false;
    }
    return vector[byte(vertex)] & bit(vertex);
}

std::int64_t bit_vector::get_bit_index(const std::int64_t vector_size,
                                       const std::uint8_t* vector,
                                       const std::int64_t vertex) {
    if (vertex / 8 > vector_size) {
        return bad_arguments;
    }

    std::int64_t nbyte = byte(vertex);
    std::uint8_t bit = vertex & 7;
    std::int64_t result = 0;

    for (std::int64_t i = 0; i < nbyte; i++)
        result += bit_vector::bit_set_table[vector[i]];

    std::uint8_t checkbyte = vector[nbyte];
    for (std::uint8_t i = 0; i < bit; i++) {
        result += checkbyte & 1;
        checkbyte >>= 1;
    }
    return result;
}

std::int64_t bit_vector::popcount(const std::int64_t vector_size, const std::uint8_t* vector) {
    if (vector == nullptr) {
        return bad_arguments;
    }

    std::int64_t result = 0;
    for (std::int64_t i = 0; i < vector_size; i++) {
        result += bit_vector::bit_set_table[vector[i]];
    }

    return result;
}

std::int64_t bit_vector::popcount() const {
    return popcount(n, vector);
}

void bit_vector::split_by_registers(const std::int64_t vector_size,
                                    std::int64_t* registers,
                                    const register_size max_size) {
    std::int64_t size = vector_size;
    for (std::int64_t i = power_of_two(max_size); i >= 0; i--) {
        registers[i] = size >> i;
        size -= registers[i] << i;
    }
}

std::int64_t bit_vector::min_index() const {
    std::int64_t result = 0;

    for (std::int64_t i = n; i > 0; i--) {
        if (vector[i] > 0) {
            result = power_of_two(vector[i]);
            result += (i << 3);
            break;
        }
    }
    return result;
}

std::int64_t bit_vector::max_index() const {
    std::int64_t result = 1 << n;

    for (std::int64_t i = 0; i < n; i++) {
        if (vector[i] > 0) {
            result = power_of_two(vector[i]);
            result += (i << 3);
            break;
        }
    }
    return result;
}

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail