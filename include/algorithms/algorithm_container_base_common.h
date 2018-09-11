/* file: algorithm_container_base_common.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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
#include "algorithms/algorithm_kernel.h"

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
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINERIFACE"></a>
 * \brief Implements the abstract interface AlgorithmContainerIface. It is associated with the Algorithm class
 *        and supports the methods for computation and finalization of the algorithm results
 *        in the batch, distributed, and online modes.
 */
class AlgorithmContainerIface
{
public:
    DAAL_NEW_DELETE();

    virtual ~AlgorithmContainerIface() {}
};

/**
 * <a name="ALGORITHMS__ALGORITHMCONTAINERIFACEIMPL"></a>
 * \brief Implements the abstract interface AlgorithmContainerIfaceImpl. It is associated with the Algorithm class
 *        and supports the methods for computation and finalization of the algorithm results
 *        in the batch, distributed, and online modes.
 */
class AlgorithmContainerIfaceImpl : public AlgorithmContainerIface
{
public:
    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmContainerIfaceImpl(daal::services::Environment::env *daalEnv) : _env(daalEnv), _kernel(NULL) {}

    virtual ~AlgorithmContainerIfaceImpl() {}

    /**
     * Sets the information about the environment
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    void setEnvironment(daal::services::Environment::env *daalEnv)
    {
        _env = daalEnv;
    }

protected:
    daal::services::Environment::env     *_env;
    Kernel *_kernel;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINER"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms. It is associated with the Algorithm class
 *        and supports the methods for computation and finalization of the algorithm results
 *        in the batch, distributed, and online modes.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template<ComputeMode mode> class AlgorithmContainer : public AlgorithmContainerIfaceImpl
{
public:
    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmContainer(daal::services::Environment::env *daalEnv) : AlgorithmContainerIfaceImpl(daalEnv) {}

    virtual ~AlgorithmContainer() {}

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

    /**
     * Setups internal datastructures for compute function.
     */
    virtual services::Status setupCompute() = 0;

    /**
     * Resets internal datastructures for compute function.
     */
    virtual services::Status resetCompute() = 0;

    /**
     * Setups internal datastructures for finalizeCompute function.
     */
    virtual services::Status setupFinalizeCompute() = 0;

    /**
     * Resets internal datastructures for finalizeCompute function.
     */
    virtual services::Status resetFinalizeCompute() = 0;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMCONTAINERIMPL"></a>
 * \brief Abstract interface class that provides virtual methods to access and run implementations
 *        of the algorithms. It is associated with the Algorithm class
 *        and supports the methods for computation and finalization of the algorithm results
 *        in the batch, distributed, and online modes.
 *        The methods of the container are defined in derivative containers defined for each algorithm.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template<ComputeMode mode> class AlgorithmContainerImpl : public AlgorithmContainer<mode>
{
public:
    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
    AlgorithmContainerImpl(daal::services::Environment::env *daalEnv = 0) : AlgorithmContainer<mode>(daalEnv), _in(0), _pres(0), _res(0), _par(0) {}

    virtual ~AlgorithmContainerImpl() {}

    /**
     * Sets arguments of the algorithm
     * \param[in] in    Pointer to the input arguments of the algorithm
     * \param[in] pres  Pointer to the partial results of the algorithm
     * \param[in] par   Pointer to the parameters of the algorithm
     */
    void setArguments(Input *in, PartialResult *pres, Parameter *par)
    {
        _in   = in;
        _pres = pres;
        _par  = par;
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

    virtual services::Status setupCompute() DAAL_C11_OVERRIDE { return services::Status();  }

    virtual services::Status resetCompute() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status setupFinalizeCompute() DAAL_C11_OVERRIDE { return services::Status(); }

    virtual services::Status resetFinalizeCompute() DAAL_C11_OVERRIDE { return services::Status(); }

protected:
    Input                                *_in;
    PartialResult                        *_pres;
    Result                               *_res;
    Parameter                            *_par;
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
class DAAL_EXPORT AlgorithmDispatchContainer : public AlgorithmContainerImpl<mode>
{
public:
    /**
     * Default constructor
     * \param[in] daalEnv   Pointer to the structure that contains information about the environment
     */
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
        return _cntr->setupCompute();
    }

    virtual services::Status resetCompute() DAAL_C11_OVERRIDE
    {
        return _cntr->resetCompute();
    }

protected:
    AlgorithmContainerImpl<mode> *_cntr;
};

#define __DAAL_ALGORITHM_CONTAINER(Mode, ContainerTemplate, ...)         \
    AlgorithmDispatchContainer< Mode,                                    \
        ContainerTemplate<__VA_ARGS__, sse2>                             \
        DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__)      \
        DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__)      \
        DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, __VA_ARGS__)        \
        DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__)       \
        DAAL_KERNEL_AVX512_mic_CONTAINER(ContainerTemplate, __VA_ARGS__) \
        DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>

/** @} */
} // namespace interface1
using interface1::AlgorithmContainerImpl;
using interface1::AlgorithmDispatchContainer;

}
}
#endif
