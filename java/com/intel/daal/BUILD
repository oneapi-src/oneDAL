package(default_visibility = ["//visibility:public"])
load("@onedal//dev/bazel:daal.bzl",
    "daal_java_module",
    "daal_java_jni_module",
    "daal_jni_dynamic_lib",
    "daal_jar",
    "daal_dynamic_lib",
)

daal_java_module(
    name = "utils_java",
    srcs = glob(["utils/*.java"]),
)

# daal_java_module(
#     name = "services_java",
#     srcs = glob(["services/*.java"]),
#     deps = [
#         ":utils_java",
#     ]
# )

daal_java_jni_module(
    name = "services",
    java_deps = [
        ":utils_java",
    ],
    java_srcs = glob(["services/*.java"]),
    cc_srcs = glob(["services/*.cpp"]),
    cc_deps = [
        "@onedal//cpp/daal:includes",
    ],
)

daal_jni_dynamic_lib(
    name = "jni",
    lib_name = "JavaApi",
    deps = [
        ":services_cc",
    ]
)

# daal_jar(
#     name = "jar",
#     lib_name = "daal",
#     deps = [
#         ":services_java",
#     ],
# )

# daal_dynamic_lib(
#     name = "jni",
#     lib_name = "JavaAPI",
#     deps = [
#         ":services_cc",
#     ],
# )
