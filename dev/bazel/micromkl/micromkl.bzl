_MICROMKL_ROOT = "MKLFPKROOT"
_MICROMKL_INC_PATH = "lnx/include"
_MICROMKL_LIB_PATH = "lnx/lib/intel64"
_MICROMKL_MKL_THR = "libdaal_mkl_thread.a"
_MICROMKL_MKL_SEQ = "libdaal_mkl_sequential.a"
_MICROMKL_MKL_VML = "libdaal_vmlipp_core.a"
_MICROMKL_LIBS = [
    _MICROMKL_MKL_THR,
    _MICROMKL_MKL_SEQ,
    _MICROMKL_MKL_VML,
]

_MICROMKL_DPC_ROOT = "MKLGPUFPKROOT"
_MICROMKL_DPC_LIB_PATH = "lib/intel64"
_MICROMKL_DPC_INC_PATH = "include"
_MICROMKL_DPC_MKL = "libdaal_sycl.a"
_MICROMKL_DPC_LIBS = [
    _MICROMKL_DPC_MKL,
]

def _join(*args):
    return "/".join(args)

def _create_symlinks(repo_ctx, root, libs):
    for lib in libs:
        src_lib_path = _join(root, lib)
        dst_lib_path = lib
        repo_ctx.symlink(src_lib_path, dst_lib_path)

def _micromkl_repository_impl(repo_ctx):
    root = repo_ctx.os.environ.get(_MICROMKL_ROOT)
    lib_path = _join(root, _MICROMKL_LIB_PATH)
    _create_symlinks(repo_ctx, lib_path, _MICROMKL_LIBS)
    repo_ctx.symlink(_join(root, _MICROMKL_INC_PATH), "include")
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/micromkl:micromkl.BUILD.tpl"),
        substitutions = {},
    )

micromkl_repo = repository_rule(
    implementation = _micromkl_repository_impl,
    environ = [
        _MICROMKL_ROOT,
    ],
)

def _micromkl_dpc_repository_impl(repo_ctx):
    root = repo_ctx.os.environ.get(_MICROMKL_DPC_ROOT)
    lib_path = _join(root, _MICROMKL_DPC_LIB_PATH)
    _create_symlinks(repo_ctx, lib_path, _MICROMKL_DPC_LIBS)
    repo_ctx.symlink(_join(root, _MICROMKL_DPC_INC_PATH), "include")
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/micromkl:micromkldpc.BUILD.tpl"),
        substitutions = {},
    )

micromkl_dpc_repo = repository_rule(
    implementation = _micromkl_dpc_repository_impl,
    environ = [
        _MICROMKL_DPC_ROOT,
    ],
)
