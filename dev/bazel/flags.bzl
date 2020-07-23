
# Always flags

# Opt flags
# -O2
# -D_FORTIFY_SOURCE=2

# Dbg flags
# -g
# -DDEBUG_ASSERT

lnx_cc_common_flags = [
    "-fwrapv",
    "-fstack-protector-strong",
    "-fno-strict-overflow",
    "-fno-delete-null-pointer-checks",
    "-pedantic",
    "-Wall",
    "-Wextra",
    "-Wformat",
    "-Wformat-security",
    "-Werror=unknown-pragmas",
    "-Werror=uninitialized",
    "-Werror=return-type",
    "-Wno-unused-parameter",
    "-Wno-unused-command-line-argument",
]

def get_default_flags(arch_id, os_id, compiler_id):
    if os_id == "lnx":
        return lnx_cc_common_flags
    elif os_id == "win":
        return [
            # "-nologo",
            "/EHsc",
            "/WX",
            "/DEBUG",
            "/Z7",
        ]
    else:
        return []


def get_cpu_flags(arch_id, os_id, compiler_id):
    cpu_flag_pattern = "-march={}"
    cpu_flags = {
        "sse2":       "",
        "ssse3":      "",
        "sse42":      "",
        "avx":        "",
        "avx2":       "",
        "avx512_mic": "",
        "avx512":     "",
    }
    if compiler_id == "gcc":
        cpu_flags = {
            "sse2":       "pentium4" if arch_id == "ia32" else "nocona",
            "ssse3":      "pentium4" if arch_id == "ia32" else "nocona",
            "sse42":      "corei7",
            "avx":        "sandybridge",
            "avx2":       "haswell",
            "avx512_mic": "haswell",
            "avx512":     "haswell",
        }
    return { k: cpu_flag_pattern.format(v) for k, v in cpu_flags.items() }
