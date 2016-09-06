/* file: algorithm_base_common.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#ifndef __ALGORITHM_BASE_COMMON_H__
#define __ALGORITHM_BASE_COMMON_H__

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

    virtual ~Kernel () {};

    void setErrorCollection(const services::KernelErrorCollectionPtr& errors)
    {
        _errors = errors;
    }

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
     */
    void setErrorCollection(const services::ErrorCollectionPtr& errors)
    {
        _errors = errors;
        setKernelErrorCollection();
    }

    /**
     * Sets the collection of errors to kernels
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
    virtual void compute() = 0;

    /**
     * Computes final results of the algorithm using partial results in %online and %distributed modes.
     * This method behaves similarly to finalizeCompute method of the Algorithm class.
     */
    virtual void finalizeCompute() = 0;

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
 * \ref opt_notice
 *
 *
 * \tparam mode                 Computation mode of the algorithm, \ref ComputeMode
 * \tparam sse2Container        Implementation for Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2)
 * \tparam ssse3Container       Implementation for Intel(R) Supplemental Streaming SIMD Extensions 3 (Intel(R) SSSE3)
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

    virtual void compute() DAAL_C11_OVERRIDE
    {
        _cntr->setArguments(this->_in, this->_pres, this->_par);
        _cntr->setErrorCollection(this->_errors);
        _cntr->compute();
    }

    virtual void finalizeCompute() DAAL_C11_OVERRIDE
    {
        _cntr->setArguments(this->_in, this->_pres, this->_par);
        _cntr->setResult(this->_res);
        _cntr->setErrorCollection(this->_errors);
        _cntr->finalizeCompute();
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

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMIFACE"></a>
 *  \brief Abstract class which defines interface for the library component
 *         related to data processing involving execution of the algorithms
 *         for analysis, modeling, and prediction
 */
class AlgorithmIface
{
public:
    DAAL_NEW_DELETE();

    AlgorithmIface() : _enableChecks(true) {}

    virtual ~AlgorithmIface()
    {}

    /**
     * Validates parameters of the compute method
     */
    virtual void checkComputeParams() = 0;

    /**
     * Validates result parameters of the compute method
     */
    virtual void checkResult() = 0;

    /**
     * Sets flag of requiring parameters checks
     * \param enableChecksFlag True if checks are needed, false if no checks are required
     */
    void enableChecks(bool enableChecksFlag)
    {
        _enableChecks = enableChecksFlag;
    }

    /**
     * Returns flag of checking necessity
     * \return flag of checking necessity
     */
    bool isChecksEnabled() const
    {
        return _enableChecks;
    }

    /**
     * Returns errors during the computations
     * \return Errors during the computations
     */
    virtual services::SharedPtr<services::ErrorCollection> getErrors() = 0;

private:
    bool _enableChecks;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHM"></a>
 * \brief Implements the abstract interface AlgorithmIface. Algorithm is, in turn, the base class
 *         for the classes interfacing the major stages of data processing: Analysis, Training and Prediction.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template<ComputeMode mode> class Algorithm : public AlgorithmIface
{
public:
    /** Default constructor */
    Algorithm() : _ac(0), _in(0), _pres(0), _res(0), _par(0), _errors(new services::ErrorCollection())
    {
        getEnvironment();
    };

    virtual ~Algorithm()
    {
        if(_ac)
        {
            delete _ac;
        }
    }

    virtual void clean() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const = 0;

    /**
     * Validates result parameters of the finalizeCompute method
     */
    virtual void checkPartialResult() = 0;

    /**
     * Validates parameters of the finalizeCompute method
     */
    virtual void checkFinalizeComputeParams() = 0;

    services::SharedPtr<services::ErrorCollection> getErrors()
    {
        return _errors;
    }

protected:
    PartialResult *allocatePartialResultMemory()
    {
        if(_pres == 0)
        {
            allocatePartialResult();
        }

        return _pres;
    }

    virtual void setParameter() {}

    void allocateInputMemory()
    {
        allocateInput();
    }

    void allocateResultMemory()
    {
        if(_res == 0)
        {
            allocateResult();
        }
    }

    void initPartialResult()
    {
        initializePartialResult();
    }


    virtual void allocateInput() {};
    virtual void allocatePartialResult() = 0;
    virtual void allocateResult() = 0;

    virtual void initializePartialResult() = 0;
    virtual Algorithm<mode> *cloneImpl() const = 0;

    void getEnvironment()
    {
        int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();
        if(cpuid < 0)
        {
            _errors->add(services::ErrorCpuNotSupported);
        }
        _env.cpuid = cpuid;
    }

    void throwIfPossible()
    {
#if (!defined(DAAL_NOTHROW_EXCEPTIONS))
        throw services::Exception::getException(this->_errors->getDescription());
#endif
    }

    bool getInitFlag() { return _pres->getInitFlag(); }
    void setInitFlag(bool flag) { _pres->setInitFlag(flag); }

    AlgorithmContainerIface<mode> *_ac;
    daal::services::Environment::env    _env;

    Input         *_in;
    PartialResult *_pres;
    Result        *_res;
    Parameter     *_par;
    services::SharedPtr<services::ErrorCollection> _errors;
};
/** @} */
} // namespace interface1
using interface1::Kernel;
using interface1::AlgorithmContainerIface;
using interface1::AlgorithmDispatchContainer;
using interface1::AlgorithmIface;
using interface1::Algorithm;

}
}
#endif
