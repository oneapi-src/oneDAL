package(default_visibility = ["//visibility:public"])

cc_library(
    name = "headers",
    features = [ "dpc++" ],
    hdrs = glob(["include/cpu_gpu_dpcpp/oneapi/*.hpp"]) +
           glob(["include/cpu_gpu_dpcpp/oneapi/ccl/*.hpp"]) +
           glob(["include/cpu_gpu_dpcpp/oneapi/ccl/*.h"]) +
           glob(["include/cpu_gpu_dpcpp/oneapi/ccl/native_device_api/*.hpp"]) +
           glob(["include/cpu_gpu_dpcpp/oneapi/ccl/native_device_api/sycl/*.hpp"]) +
           glob(["include/cpu_gpu_dpcpp/oneapi/ccl/native_device_api/sycl_l0/*.hpp"]),
    includes = [ "include/cpu_gpu_dpcpp/oneapi" ] + [ "include/cpu_gpu_dpcpp/" ],
)

cc_library(
    name = "libccl",
    srcs = [
        "lib/cpu_gpu_dpcpp/libccl.so.1.0",
    ],
)

cc_library(
    name = "ccl",
    deps = [
        ":headers",
        ":libccl",
    ],
)
