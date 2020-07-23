load("@onedal//dev/bazel/toolchains:configure_lnx.bzl", "configure_cc_toolchain_lnx")

def _detect_os(repo_ctx):
    if "linux" in repo_ctx.os.name:
        return "lnx"
    elif "mac" in repo_ctx.os.name:
        return "mac"
    elif "windows" in repo_ctx.os.name:
        return "win"

def _detect_default_compiler(repo_ctx, os_id):
    compiler_id = "icc"
    is_icc_available = repo_ctx.which("icc") != None
    if not is_icc_available:
        compiler_id = {
            "lnx": "gcc",
            "mac": "clang",
            "win": "cl",
        }[os_id]
    return compiler_id

def _detect_compiler(repo_ctx, os_id):
    if not "CC" in repo_ctx.os.environ:
        return _detect_default_compiler(repo_ctx, os_id)
    compiler_path = repo_ctx.os.environ["CC"]
    # TODO: Use more relieble way to detect compiler
    if "gcc" in compiler_path:
        return "gcc"
    elif "clang" in compiler_path:
        return "clang"
    elif "cl" in compiler_path:
        return "cl"
    elif "icc" in compiler_path:
        return "icc"

def _detect_requirements(repo_ctx):
    os_id = _detect_os(repo_ctx)
    compiler_id = _detect_compiler(repo_ctx, os_id)
    return struct(
        os_id = os_id,
        compiler_id = compiler_id,

        libc_version = "local",
        libc_abi_version = "local",
        compiler_abi_version = "local",

        host_arch_id = "intel64",
        target_arch_id = "intel64",

        # TODO: Detect compiler version
        compiler_version = "local",

        # TODO: Detect DPC++ compiler, use $env{DPCC}
        dpc_compiler_id = "dpcpp",

        # TODO: Detect compiler version
        dpc_compiler_version = "local",
    )

def _configure_cc_toolchain(repo_ctx, reqs):
    configure_cc_toolchain_os = {
        "lnx": configure_cc_toolchain_lnx,
    }[reqs.os_id]
    return configure_cc_toolchain_os(repo_ctx, reqs)

def _onedal_cc_toolchain_impl(repo_ctx):
    reqs = _detect_requirements(repo_ctx)
    _configure_cc_toolchain(repo_ctx, reqs)

onedal_cc_toolchain = repository_rule(
    implementation = _onedal_cc_toolchain_impl,
    environ = [
        "CC",
        "PATH",
        "INCLUDE",
        "LIB",
    ],
)

def declare_onedal_cc_toolchain(name):
    onedal_cc_toolchain(name = name)
    native.register_toolchains("@{}//:all".format(name))
