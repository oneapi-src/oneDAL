load("@onedal//dev/bazel:cc.bzl",
    "cc_module",
    "cc_static_lib",
)

def daal_module(name, features=[], **kwargs):
    cc_module(
        name = name,
        features = [ "c++11" ] + features,
        cpu_defines = {
            "sse2":       [ "DAAL_CPU=sse2"       ],
            "ssse3":      [ "DAAL_CPU=ssse3"      ],
            "sse42":      [ "DAAL_CPU=sse42"      ],
            "avx":        [ "DAAL_CPU=avx"        ],
            "avx2":       [ "DAAL_CPU=avx2"       ],
            "avx512_mic": [ "DAAL_CPU=avx512_mic" ],
            "avx512":     [ "DAAL_CPU=avx512"     ],
        },
        fpt_defines = {
            "f32": [ "DAAL_FPTYPE=float"  ],
            "f64": [ "DAAL_FPTYPE=double" ],
        },
        **kwargs,
    )
