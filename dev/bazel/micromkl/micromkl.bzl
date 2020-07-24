load("@bazel_skylib//lib:paths.bzl", "paths")

def _create_symlinks(repo_ctx, os, root, entries):
    for entry in entries:
        entry_fmt = entry.format(os=os)
        src_entry_path = paths.join(root, entry_fmt)
        dst_entry_path = entry_fmt
        repo_ctx.symlink(src_entry_path, dst_entry_path)

def _prebuilt_libs_repo_impl(repo_ctx):
    # TODO: Detect OS
    os = "lnx"
    root = repo_ctx.os.environ.get(repo_ctx.attr.root_env_var)
    _create_symlinks(repo_ctx, os, root, repo_ctx.attr.includes)
    _create_symlinks(repo_ctx, os, root, repo_ctx.attr.libs)
    repo_ctx.template(
        "BUILD",
        repo_ctx.attr.build_template,
        substitutions = {
            "%{os}": os,
        },
    )

def prebuilt_libs_repo_rule(root_env_var, includes, libs, build_template):
    return repository_rule(
        implementation = _prebuilt_libs_repo_impl,
        environ = [
            root_env_var,
        ],
        attrs = {
            "root_env_var": attr.string(default=root_env_var),
            "includes": attr.string_list(default=includes),
            "libs": attr.string_list(default=libs),
            "build_template": attr.label(allow_files=True,
                                         default=Label(build_template)),
        }
    )

micromkl_repo = prebuilt_libs_repo_rule(
    root_env_var = "MKLFPKROOT",
    includes = [
        "include",
        "{os}/include",
    ],
    libs = [
        "{os}/lib/intel64/libdaal_mkl_thread.a",
        "{os}/lib/intel64/libdaal_mkl_sequential.a",
        "{os}/lib/intel64/libdaal_vmlipp_core.a",
    ],
    build_template = "@onedal//dev/bazel/micromkl:micromkl.BUILD.tpl",
)

micromkl_dpc_repo = prebuilt_libs_repo_rule(
    root_env_var = "MKLGPUFPKROOT",
    includes = [
        "include",
    ],
    libs = [
        "lib/intel64/libdaal_sycl.a",
    ],
    build_template = "@onedal//dev/bazel/micromkl:micromkldpc.BUILD.tpl",
)
