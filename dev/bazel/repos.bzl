load("@onedal//dev/bazel:utils.bzl", "utils", "paths")

def _create_symlinks(repo_ctx, root, entries, substitutions={}):
    for entry in entries:
        entry_fmt = utils.substitude(entry, substitutions)
        src_entry_path = paths.join(root, entry_fmt)
        dst_entry_path = entry_fmt
        repo_ctx.symlink(src_entry_path, dst_entry_path)

def _prebuilt_libs_repo_impl(repo_ctx):
    root = repo_ctx.os.environ.get(repo_ctx.attr.root_env_var)
    substitutions = {
        # TODO: Detect OS
        "%{os}": "lnx",
    }
    _create_symlinks(repo_ctx, root, repo_ctx.attr.includes, substitutions)
    _create_symlinks(repo_ctx, root, repo_ctx.attr.libs, substitutions)
    repo_ctx.template(
        "BUILD",
        repo_ctx.attr.build_template,
        substitutions = substitutions,
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

repos = struct(
    prebuilt_libs_repo_rule = prebuilt_libs_repo_rule,
)
