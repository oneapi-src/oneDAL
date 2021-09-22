/* file: algorithm_container_base_batch.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Implementation of base classes defining algorithm interface for batch processing mode.
//--
*/

#ifndef __ALGORITHM_CONTAINER_BASE_BATCH_H__
#define __ALGORITHM_CONTAINER_BASE_BATCH_H__

#include "services/daal_memory.h"
#include "services/internal/daal_kernel_defines.h"
#include "services/internal/gpu_support_checker.h"
#include "services/internal/execution_context.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains version 1.0 of Intel(R) oneAPI Data Analytics Library interface.
 */
namespace interface1
{
/**
 * @addtogroup base_algorithms
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINER"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms in %batch mode. It is associated with the Algorithm<batch> class
 *        and supports the methods for computation of the algorithm results.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 */
template <>
class AlgorithmContainer<batch> : public AlgorithmContainerIfaceImpl
{
public:
    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmContainer(daal::services::Environment::env * daalEnv) : AlgorithmContainerIfaceImpl(daalEnv) {}

    virtual ~AlgorithmContainer() DAAL_C11_OVERRIDE {}

    /**
     * Computes final results of the algorithm.
     * This method behaves similarly to compute method of the Algorithm<batch> class.
     */
    virtual services::Status compute() = 0;

    /**
     * Setups internal datastructures for compute function.
     */
    virtual services::Status setupCompute() = 0;

    /**
     * Resets internal datastructures for compute function.
     */
    virtual services::Status resetCompute() = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINERIMPL"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms in %batch mode. It is associated with the Algorithm<batch> class
 *        and supports the methods for computation of the algorithm results.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 */
template <>
class AlgorithmContainerImpl<batch> : public AlgorithmContainer<batch>
{
public:
    DAAL_NEW_DELETE();

    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmContainerImpl(daal::services::Environment::env * daalEnv = 0) : AlgorithmContainer<batch>(daalEnv), _par(0), _in(0), _res(0) {};

    virtual ~AlgorithmContainerImpl() {}

    /**
     * Sets arguments of the algorithm
     * \param[in] in    Pointer to the input arguments of the algorithm
     * \param[in] res   Pointer to the final results of the algorithm
     * \param[in] par   Pointer to the parameters of the algorithm
     */
    void setArguments(Input * in, Result * res, Parameter * par)
    {
        _in  = in;
        _par = par;
        _res = res;
    }

    /**
     * Retrieves final results of the algorithm
     * \return   Pointer to the final results of the algorithm
     */
    Result * getResult() { return _res; }

    virtual services::Status setupCompute() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status resetCompute() DAAL_C11_OVERRIDE { return services::Status(); }

protected:
    Parameter * _par;
    Input * _in;
    Result * _res;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMDISPATCHCONTAINER"></a>
 * \brief Implements a container to dispatch batch processing algorithms to CPU-specific implementations.
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
template <typename sse2Container DAAL_KERNEL_SSSE3_ONLY(typename ssse3Container) DAAL_KERNEL_SSE42_ONLY(typename sse42Container)
              DAAL_KERNEL_AVX_ONLY(typename avxContainer) DAAL_KERNEL_AVX2_ONLY(typename avx2Container)
                  DAAL_KERNEL_AVX512_MIC_ONLY(typename avx512_micContainer) DAAL_KERNEL_AVX512_ONLY(typename avx512Container)>
class DAAL_EXPORT AlgorithmDispatchContainer<batch, sse2Container DAAL_KERNEL_SSSE3_ONLY(ssse3Container) DAAL_KERNEL_SSE42_ONLY(sse42Container)
                                                        DAAL_KERNEL_AVX_ONLY(avxContainer) DAAL_KERNEL_AVX2_ONLY(avx2Container)
                                                            DAAL_KERNEL_AVX512_MIC_ONLY(avx512_micContainer) DAAL_KERNEL_AVX512_ONLY(avx512Container)>
    : public AlgorithmContainerImpl<batch>
{
public:
    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmDispatchContainer(daal::services::Environment::env * daalEnv);

    virtual ~AlgorithmDispatchContainer()
    {
        delete _cntr;
        _cntr = 0;
    }

    virtual services::Status compute() DAAL_C11_OVERRIDE
    {
        services::internal::sycl::ExecutionContextIface & context = services::internal::getDefaultContext();
        services::internal::sycl::InfoDevice & deviceInfo         = context.getInfoDevice();
        if (!daal::services::internal::isImplementedForDevice(deviceInfo, _cntr)) return services::Status(services::ErrorDeviceSupportNotImplemented);
        _cntr->setArguments(this->_in, this->_res, this->_par);
        return _cntr->compute();
    }

    virtual services::Status setupCompute() DAAL_C11_OVERRIDE
    {
        _cntr->setArguments(this->_in, this->_res, this->_par);
        return _cntr->setupCompute();
    }

    virtual services::Status resetCompute() DAAL_C11_OVERRIDE { return _cntr->resetCompute(); }

protected:
    AlgorithmContainerImpl<batch> * _cntr;

private:
    AlgorithmDispatchContainer(const AlgorithmDispatchContainer &);
    AlgorithmDispatchContainer & operator=(const AlgorithmDispatchContainer &);
};

/** @} */
} // namespace interface1

} // namespace algorithms
} // namespace daal

#endif
