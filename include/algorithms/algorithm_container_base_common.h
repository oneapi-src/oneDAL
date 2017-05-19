/* file: algorithm_container_base_common.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

#include "services/daal_memory.h"
#include "services/daal_kernel_defines.h"
#include "services/error_handling.h"
#include "services/env_detect.h"
#include "algorithms/algorithm_types.h"

namespace daal
{
namespace algorithms
{

/**
 * @addtogroup base_algorithms
 * @{
 */
/**
* \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
*/
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL"></a>
 * \brief %Base class to represent algorithm implementation
 */
class Kernel
{
public:
    Kernel() : _errors(new services::KernelErrorCollection()) {};

    virtual ~Kernel () {}

    /**
     * Sets the collection of errors
     * \param[in] errors    Pointer to the collection of errors
     * \DAAL_DEPRECATED
     */
    void setErrorCollection(const services::KernelErrorCollectionPtr& errors)
    {
        _errors = errors;
    }

    /**
     * Gets the collection of errors
     * \DAAL_DEPRECATED
     */
    services::KernelErrorCollectionPtr getErrorCollection()
    {
        return _errors;
    }

protected:
    services::SharedPtr<services::KernelErrorCollection> _errors;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINERIFACE"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms. It is associated with the Algorithm class
 *        and supports the methods for computation and finalization of the algorithm results
 *        in the batch, distributed, and online modes.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template<ComputeMode mode> class AlgorithmContainerIface
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty container */
    AlgorithmContainerIface(daal::services::Environment::env *daalEnv = 0) : _in(0), _pres(0), _res(0), _par(0),
        _env(daalEnv), _errors(new services::ErrorCollection()),
        _kernel(NULL) {};

    virtual ~AlgorithmContainerIface() {}

    /**
     * Sets the information about the environment
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    void setEnvironment(daal::services::Environment::env *daalEnv)
    {
        _env = daalEnv;
    }

    /**
     * Sets arguments of the algorithm
     * \param[in] in    Pointer to the input arguments of the algorithm
     * \param[in] pres  Pointer to the partial results of the algorithm
     * \param[in] par   Pointer to the parameters of the algorithm
     */
    void setArguments(Input *in, PartialResult *pres, Parameter *par)
    {
        _in  = in;
        _pres = pres;
        _par = par;
    }

    /**
     * Sets the collection of errors
     * \param[in] errors    Pointer to the collection of errors
     * \DAAL_DEPRECATED
     */
    void setErrorCollection(const services::ErrorCollectionPtr& errors)
    {
        _errors = errors;
        setKernelErrorCollection();
    }

    /**
     * Sets the collection of errors to kernels
     * \DAAL_DEPRECATED
     */
    void setKernelErrorCollection()
    {
        if(_kernel)
            _kernel->setErrorCollection(_errors->getErrors());
    }

    /**
     * Sets partial results of the algorithm
     * \param[in] pres   Pointer to the partial results of the algorithm
     */
    void setPartialResult(PartialResult *pres)
    {
        _pres = pres;
    }

    /**
     * Sets final results of the algorithm
     * \param[in] res   Pointer to the final results of the algorithm
     */
    void setResult(Result *res)
    {
        _res = res;
    }

    /**
     * Retrieves final results of the algorithm
     * \return   Pointer to the final results of the algorithm
     */
    Result *getResult() const
    {
        return _res;
    }

    /**
     * Computes final results of the algorithm in the %batch mode,
     * or partial results of the algorithm in %online and %distributed modes.
     * This method behaves similarly to compute method of the Algorithm class.
     */
    virtual services::Status compute() = 0;

    /**
     * Computes final results of the algorithm using partial results in %online and %distributed modes.
     * This method behaves similarly to finalizeCompute method of the Algorithm class.
     */
    virtual services::Status finalizeCompute() = 0;

    virtual services::Status setupCompute() { return services::Status();  }

    virtual services::Status resetCompute() { return services::Status(); }

    virtual services::Status setupFinalizeCompute() { return services::Status(); }

    virtual services::Status resetFinalizeCompute() { return services::Status(); }

protected:
    Input                                *_in;
    PartialResult                        *_pres;
    Result                               *_res;
    Parameter                            *_par;
    daal::services::Environment::env     *_env;
    services::ErrorCollectionPtr        _errors;

    Kernel *_kernel;
};

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
template<ComputeMode mode,
    typename sse2Container
    DAAL_KERNEL_SSSE3_ONLY(typename ssse3Container)
    DAAL_KERNEL_SSE42_ONLY(typename sse42Container)
    DAAL_KERNEL_AVX_ONLY(typename avxContainer)
    DAAL_KERNEL_AVX2_ONLY(typename avx2Container)
    DAAL_KERNEL_AVX512_mic_ONLY(typename avx512_micContainer)
    DAAL_KERNEL_AVX512_ONLY(typename avx512Container)
>
class DAAL_EXPORT AlgorithmDispatchContainer : public AlgorithmContainerIface<mode>
{
public:
    /** Default constructor. Constructs empty container */
    AlgorithmDispatchContainer(daal::services::Environment::env *daalEnv);
    virtual ~AlgorithmDispatchContainer() { delete _cntr; }

    virtual services::Status compute() DAAL_C11_OVERRIDE
    {
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
        _cntr->setErrorCollection(this->_errors);
        return _cntr->setupCompute();
    }

    virtual services::Status resetCompute() DAAL_C11_OVERRIDE
    {
        return _cntr->resetCompute();
    }

protected:
    AlgorithmContainerIface<mode> *_cntr;
};

#define __DAAL_ALGORITHM_CONTAINER(Mode, ContainerTemplate, ...)    \
    AlgorithmDispatchContainer< Mode,                           \
        ContainerTemplate<__VA_ARGS__, sse2>                      \
        DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_AVX512_mic_CONTAINER(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>

/** @} */
} // namespace interface1
using interface1::Kernel;
using interface1::AlgorithmContainerIface;
using interface1::AlgorithmDispatchContainer;

}
}
#endif
