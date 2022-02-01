/* file: algorithm_container_base_common.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of base classes defining algorithm interface.
//--
*/

#ifndef __ALGORITHM_CONTAINER_BASE_COMMON_H__
#define __ALGORITHM_CONTAINER_BASE_COMMON_H__

#include "algorithms/algorithm_container_base.h"
#include "services/error_handling.h"
#include "services/internal/gpu_support_checker.h"
#include "services/internal/execution_context.h"

namespace daal
{
namespace algorithms
{
/**
 * @addtogroup base_algorithms
 * @{
 */
/**
* \brief Contains version 1.0 of Intel(R) oneAPI Data Analytics Library interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMDISPATCHCONTAINER"></a>
 * \brief Implements a container to dispatch algorithms to cpu-specific implementations.
 *
 *
 * \tparam mode                 Computation mode of the algorithm, \ref ComputeMode
 * \tparam sse2Container        Implementation for Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2)
 * \tparam ssse3Container       Implementation for Supplemental Streaming SIMD Extensions 3 (SSSE3)
 * \tparam sse42Container       Implementation for Intel(R) Streaming SIMD Extensions 42 (Intel(R) SSE42)
 * \tparam avxContainer         Implementation for Intel(R) Advanced Vector Extensions (Intel(R) AVX)
 * \tparam avx2Container        Implementation for Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)
 * \tparam avx512_micContainer  Implementation for Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector
 *                              Extensions 512 (Intel(R) AVX512)
 * \tparam avx512Container      Implementation for Intel(R) Xeon(R) processors based on Intel AVX-512
 */
template <ComputeMode mode, typename sse2Container DAAL_KERNEL_SSSE3_ONLY(typename ssse3Container) DAAL_KERNEL_SSE42_ONLY(typename sse42Container)
                                DAAL_KERNEL_AVX_ONLY(typename avxContainer) DAAL_KERNEL_AVX2_ONLY(typename avx2Container)
                                    DAAL_KERNEL_AVX512_MIC_ONLY(typename avx512_micContainer) DAAL_KERNEL_AVX512_ONLY(typename avx512Container)>
class DAAL_EXPORT AlgorithmDispatchContainer : public AlgorithmContainerImpl<mode>
{
public:
    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmDispatchContainer(daal::services::Environment::env * daalEnv);

    virtual ~AlgorithmDispatchContainer() { delete _cntr; }

    virtual services::Status compute() DAAL_C11_OVERRIDE
    {
        services::internal::sycl::ExecutionContextIface & context = services::internal::getDefaultContext();
        services::internal::sycl::InfoDevice & deviceInfo         = context.getInfoDevice();
        if (!daal::services::internal::isImplementedForDevice(deviceInfo, _cntr)) return services::Status(services::ErrorDeviceSupportNotImplemented);
        _cntr->setArguments(this->_in, this->_pres, this->_par);
        return _cntr->compute();
    }

    virtual services::Status finalizeCompute() DAAL_C11_OVERRIDE
    {
        _cntr->setArguments(this->_in, this->_pres, this->_par);
        _cntr->setResult(this->_res);
        return _cntr->finalizeCompute();
    }

    virtual services::Status setupCompute() DAAL_C11_OVERRIDE
    {
        _cntr->setArguments(this->_in, this->_pres, this->_par);
        _cntr->setResult(this->_res);
        return _cntr->setupCompute();
    }

    virtual services::Status resetCompute() DAAL_C11_OVERRIDE { return _cntr->resetCompute(); }

protected:
    AlgorithmContainerImpl<mode> * _cntr;

private:
    AlgorithmDispatchContainer(const AlgorithmDispatchContainer &);
    AlgorithmDispatchContainer & operator=(const AlgorithmDispatchContainer &);
};

#define __DAAL_ALGORITHM_CONTAINER(Mode, ContainerTemplate, ...)                                                                                    \
    algorithms::AlgorithmDispatchContainer<Mode, ContainerTemplate<__VA_ARGS__, sse2> DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__)   \
                                                     DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX_CONTAINER(         \
                                                         ContainerTemplate, __VA_ARGS__) DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__) \
                                                         DAAL_KERNEL_AVX512_MIC_CONTAINER(ContainerTemplate, __VA_ARGS__)                           \
                                                             DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>

/** @} */
} // namespace interface1
using interface1::AlgorithmDispatchContainer;

} // namespace algorithms
} // namespace daal
#endif
