package(default_visibility = ["//visibility:public"])

cc_library(
    name = "boost",
    srcs = glob([
        "boost/libs/serialization/src/**/*.cpp",
        "boost/libs/libboost*.a",
    ]),
    hdrs = glob([
        "boost/**/*.h",
        "boost/**/*.hpp",
        "boost/**/*.ipp",
    ]),
    includes = [
        "boost",
    ],
    visibility = ["//visibility:public"],
)

