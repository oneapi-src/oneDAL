/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/detail/cpu_info.hpp"

namespace oneapi::dal::test {

#if defined(__x86_64__)

/// x86 CPU registers relevant to CPUID instruction
struct cpuid_registers {
    uint32_t eax_;
    uint32_t ebx_;
    uint32_t ecx_;
    uint32_t edx_;
};

/// Runs CPUID instruction
///
/// @param[in]  eax       The input value of EAX register, the ID of CPUID subfunction.
///                       By that value CPUID defines which information should be returned.
///                       For example:
///                           eax == 0 - CPU vendor ID is returned;
///                           eax == 1 - Processor type, family, stepping, ... are returned;
///                           ...
/// @param[out] registers x86 CPU registers to write the results of CPUID
static inline void cpuid(uint32_t eax, cpuid_registers& registers) {
    asm volatile(
        "cpuid"
        : "=a"(registers.eax_), "=b"(registers.ebx_), "=c"(registers.ecx_), "=d"(registers.edx_)
        : "a"(eax));
}

/// Copies the value of x86 register into the buffer of characters
///
/// @param[in]  reg    The x86 register value
/// @param[out] buffer Pointer to the output character buffer
void register_to_buffer(uint32_t reg, char* buffer) {
    ONEDAL_ASSERT(buffer);
    constexpr uint32_t mask = 0xFF; // lower 8 bits
    constexpr uint32_t register_size = sizeof(uint32_t);
    for (uint32_t i = 0; i < register_size; ++i) {
        buffer[i] = reg & mask;
        reg >>= 8;
    }
}

/// Copies the values of x86 EBX, EDX, ECX (in that order) registers that contain the vendor ID
/// into the buffer of characters after the call of CPUID with EAX == 0
///
/// @param[in]  registers   x86 CPU registers with the CPUID results
/// @param[out] buffer      Pointer to the output character buffer
void registers_to_vendor_id(const cpuid_registers& registers, char* buffer) {
    ONEDAL_ASSERT(buffer);
    register_to_buffer(registers.ebx_, buffer);
    register_to_buffer(registers.edx_, buffer + 4);
    register_to_buffer(registers.ecx_, buffer + 8);
}

#endif

/// Retreives the CPU vendor
///
/// @return CPU vendor ID
detail::cpu_vendor get_vendor() {
#if defined(__x86_64__)
    /// Calls CPUID with EAX == 0 to retrieve CPU vendor
    cpuid_registers registers;
    cpuid(0, registers);

    char vendor_buffer[13] = { '\0' };
    registers_to_vendor_id(registers, vendor_buffer);

    if (std::string(vendor_buffer) == std::string("GenuineIntel")) {
        return detail::cpu_vendor::intel;
    }
    else if (std::string(vendor_buffer) == std::string("AuthenticAMD")) {
        return detail::cpu_vendor::amd;
    }
    else {
        return detail::cpu_vendor::unknown;
    }
#elif defined(__ARM_ARCH)
    /// ARM architecture
    return detail::cpu_vendor::arm;
#endif
}

TEST("can create default CPU info") {
    const detail::cpu_info default_cpu_info;

    std::cout << "CPU INFO DUMP:" << std::endl;
    std::cout << default_cpu_info.dump() << std::endl;
    REQUIRE(get_vendor() == default_cpu_info.get_cpu_vendor());
    /// REQUIRE(detail::cpu_vendor::amd == default_cpu_info.get_cpu_vendor());
}

} // namespace oneapi::dal::test
