
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
    "-Wformat",
    "-Wformat-security",
]

lnx_cc_pedantic_flags = [
    "-pedantic",
    "-Wall",
    "-Wextra",
    "-Werror=uninitialized",
    "-Werror=unknown-pragmas",
    "-Werror=return-type",
    "-Wno-unused-parameter",
    "-Wno-unused-command-line-argument",
]

lnx_cc_flags = {
    "common": lnx_cc_common_flags,
    "pedantic": lnx_cc_pedantic_flags,
}

def get_default_flags(arch_id, os_id, compiler_id, category="common"):
    _check_flag_category(category)
    if os_id == "lnx":
        return lnx_cc_flags[category]
    fail("Unsupported OS")

def get_cpu_flags(arch_id, os_id, compiler_id):
    sse2 = []
    ssse3 = []
    sse42 = []
    avx = []
    avx2 = []
    avx512_mic = []
    avx512 = []
    if compiler_id == "gcc":
        sse2       = ["-march={}".format("pentium4" if arch_id == "ia32" else "nocona")]
        ssse3      = ["-march={}".format("pentium4" if arch_id == "ia32" else "nocona")]
        sse42      = ["-march=corei7"]
        avx        = ["-march=sandybridge"]
        avx2       = ["-march=haswell"]
        avx512_mic = ["-march=haswell"]
        avx512     = ["-march=haswell"]
    elif compiler_id == "icc":
        sse2       = ["-xSSE2"]
        ssse3      = ["-xSSE3"]
        sse42      = ["-xSSE4.2"]
        avx        = ["-xAVX"]
        avx2       = ["-xCORE-AVX2"]
        avx512_mic = ["-xMIC-AVX512"]
        avx512     = ["-xCORE-AVX512", "-qopt-zmm-usage=high"]
    return {
        "sse2":       sse2,
        "ssse3":      ssse3,
        "sse42":      sse42,
        "avx":        avx,
        "avx2":       avx2,
        "avx512_mic": avx512_mic,
        "avx512":     avx512,
    }

def _check_flag_category(category):
    if not category in ["common", "pedantic"]:
        fail("Unsupported compilre flag category '{}' ".format(category) +
             "expected 'common' or 'pedantic'")
