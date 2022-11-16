load("@onedal//dev/bazel:release.bzl",
    "release",
    "release_include",
)

release(
    name = "release",
    include = [
        release_include(
            hdrs = [ "@onedal//cpp/daal:public_includes" ],
            skip_prefix = "cpp/daal/include",
        ),
        release_include(
            hdrs = [ "@onedal//cpp/daal:kernel_defines" ],
            add_prefix = "services/internal",
        ),
        release_include(
            hdrs = [ "@onedal//cpp/oneapi/dal:public_includes" ],
            skip_prefix = "cpp",
        ),
    ],
    lib = [
        "@onedal//cpp/daal:core_static",
        "@onedal//cpp/daal:thread_static",
        "@onedal//cpp/daal:core_dynamic",
        "@onedal//cpp/daal:thread_dynamic",
        "@onedal//cpp/oneapi/dal:static",
        "@onedal//cpp/oneapi/dal:dynamic",
    ] + select({
        "@config//:release_dpc_enabled": [
            "@onedal//cpp/oneapi/dal:static_dpc",
            "@onedal//cpp/oneapi/dal:dynamic_dpc",
            # TODO: Add onedal_sycl
        ],
        "//conditions:default": [],
    }),
)
