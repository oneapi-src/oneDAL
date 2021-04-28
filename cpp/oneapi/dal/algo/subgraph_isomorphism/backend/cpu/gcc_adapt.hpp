#pragma once

#if defined(__INTEL_COMPILER)
#define ONEDAL_IVDEP         _Pragma("ivdep")
#define ONEDAL_VECTOR_ALWAYS _Pragma("vector always")
#elif defined(__GNUC__)
#define ONEDAL_IVDEP _Pragma("GCC ivdep")
#define ONEDAL_VECTOR_ALWAYS
#elif defined(_MSC_VER)
#define ONEDAL_IVDEP
#define ONEDAL_VECTOR_ALWAYS
#else
#define ONEDAL_IVDEP
#define ONEDAL_VECTOR_ALWAYS
#endif

constexpr std::int32_t ONEDAL_lzcnt_u32(std::uint32_t a) {
#if defined(__INTEL_COMPILER)
    return _lzcnt_u32(a);
#else
    if (a == 0)
        return 32;
    std::uint32_t one_bit = 0x80000000; // binary: 1000 0000 0000 0000 0000 0000 0000 0000
    std::int32_t bit_pos = 0;
    while ((a & one_bit) == 0) {
        bit_pos++;
        one_bit >>= 1;
    }
    return bit_pos;
#endif
}

constexpr std::int32_t ONEDAL_lzcnt_u64(std::uint64_t a) {
#if defined(__INTEL_COMPILER)
    return _lzcnt_u64(a);
#else
    if (a == 0)
        return 64;
    std::uint64_t one_bit = 0x8000000000000000; // binary: 1000 ... 0000
    std::int32_t bit_pos = 0;
    while ((a & one_bit) == 0) {
        bit_pos++;
        one_bit >>= 1;
    }
    return bit_pos;
#endif
}

constexpr std::int32_t ONEDAL_popcnt64(std::uint64_t a) {
#if defined(__INTEL_COMPILER)
    return _popcnt64(a);
#else
    if (a == 0)
        return 0;
    std::uint64_t last_bit = 1;
    std::int32_t bit_cnt = 0;
    for (std::int32_t i = 0; i < 64; i++) {
        if (a & last_bit) {
            bit_cnt++;
        }
        a = a >> 1;
    }
    return bit_cnt;
#endif
}
