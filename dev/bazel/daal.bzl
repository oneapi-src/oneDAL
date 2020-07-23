load("@onedal//dev/bazel:cc.bzl",
    "cc_objects",
    "cc_static_lib",
)

def daal_module(name, hdrs=[], srcs=[], auto=False, **kwargs):
    pass


def _daal_module(name, lib_tag="daal", **kwargs):




_daal_extra_copts = [
    "-w", # Temporary options to disable warning
]

_daal_extra_defines = [
    "DAAL_HIDE_DEPRECATED",
    "DAAL_NOTHROW_EXCEPTIONS",
    # "__DAAL_IMPLEMENTATION", # Valid only for DLL
]

def daal_module(name, copts=[], local_defines=[], **kwargs):
    native.cc_library(
        name = name,
        copts = copts + _daal_extra_copts,
        local_defines = local_defines + _daal_extra_defines,
        **kwargs,
    )

def daal_kernel_module(name, hdrs=[], copts=[],
                       local_defines=[], deps=[], **kwargs):
    native.cc_library(
        name = name + "_headers",
        hdrs = hdrs,
    )
    cc_multidef_lib(
        name = name,
        cpus = {
            "sse2":       [ "DAAL_CPU=sse2"       ],
            "ssse3":      [ "DAAL_CPU=ssse3"      ],
            "sse42":      [ "DAAL_CPU=sse42"      ],
            "avx":        [ "DAAL_CPU=avx"        ],
            "avx2":       [ "DAAL_CPU=avx2"       ],
            "avx512_mic": [ "DAAL_CPU=avx512_mic" ],
            "avx512":     [ "DAAL_CPU=avx512"     ],
        },
        fpts = {
            "float":  [ "DAAL_FPTYPE=float"  ],
            "double": [ "DAAL_FPTYPE=double" ],
        },
        copts = copts + _daal_extra_copts,
        local_defines = local_defines + _daal_extra_defines,
        deps = deps + [":" + name + "_headers"],
        **kwargs
    )

def daal_shared_lib(name, **kwargs):
    cc_shared_lib(
        name = name,
        **kwargs
    )
