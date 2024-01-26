#ifndef __aarch64_KERNEL_DEFINES_H__
#define __aarch64_KERNEL_DEFINES_H__

#define DAAL_KERNEL_SVE

#if defined(DAAL_KERNEL_SVE)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID               daal::sve
    #define DAAL_KERNEL_SVE_ONLY(something)                        , something
    #define DAAL_KERNEL_SVE_ONLY_CODE(...)                         __VA_ARGS__
    #define DAAL_KERNEL_SVE_CONTAINER(ContainerTemplate, ...)      , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sve, __VA_ARGS__)
    #define DAAL_KERNEL_SVE_CONTAINER1(ContainerTemplate, ...)     extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sve, __VA_ARGS__);
    #define DAAL_KERNEL_SVE_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, sve, __VA_ARGS__)
    #define DAAL_KERNEL_SVE_CONTAINER_CASE_SYCL(ContainerTemplate, ...)
#else
    #define DAAL_KERNEL_SVE_ONLY(something)
    #define DAAL_KERNEL_SVE_ONLY_CODE(...)
    #define DAAL_KERNEL_SVE_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_SVE_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_SVE_CONTAINER_CASE(ContainerTemplate, ...)
    #define DAAL_KERNEL_SVE_CONTAINER_CASE_SYCL(ContainerTemplate, ...)
#endif

#endif