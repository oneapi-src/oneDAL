/* file: algorithm_base_mode.h */
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

#ifndef __ALGORITHM_BASE_MODE_H__
#define __ALGORITHM_BASE_MODE_H__

#include "services/daal_memory.h"
#include "services/daal_kernel_defines.h"
#include "services/error_handling.h"
#include "services/env_detect.h"
#include "algorithms/algorithm_types.h"
#include "algorithms/algorithm_base_common.h"

namespace daal
{
namespace algorithms
{
namespace interface1
{

/**
 * @addtogroup base_algorithms
 * @{
 */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHM"></a>
 * \brief Implements the abstract interface AlgorithmIface. Algorithm is, in turn, the base class
 *         for the classes interfacing the major stages of data processing: Analysis, Training and Prediction.
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template<ComputeMode mode> class Algorithm : public AlgorithmIfaceImpl
{
public:
    /** Default constructor */
    Algorithm() : _ac(0), _in(0), _pres(0), _res(0), _par(0) { }

    virtual ~Algorithm()
    {
        if(_ac)
        {
            delete _ac;
        }
    }

    virtual void clean() {}

    /**
     * Validates result parameters of the finalizeCompute method
     */
    virtual services::Status checkPartialResult() = 0;

    /**
     * Validates parameters of the finalizeCompute method
     */
    virtual services::Status checkFinalizeComputeParams() = 0;

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

    services::Status allocateResultMemory()
    {
        if(_res == 0)
            return allocateResult();
        return services::Status();
    }

    services::Status initPartialResult()
    {
        return initializePartialResult();
    }

    virtual services::Status allocatePartialResult() = 0;
    virtual services::Status allocateResult() = 0;

    virtual services::Status initializePartialResult() = 0;
    virtual Algorithm<mode> *cloneImpl() const = 0;

    bool getInitFlag() { return _pres->getInitFlag(); }
    void setInitFlag(bool flag) { _pres->setInitFlag(flag); }

    AlgorithmContainerImpl<mode> *_ac;

    Input         *_in;
    PartialResult *_pres;
    Result        *_res;
    Parameter     *_par;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHM"></a>
 * \brief Implements the abstract interface AlgorithmIface. Algorithm<batch> is, in turn, the base class
 *        for the classes interfacing the major stages of data processing in %batch mode:
 *        Analysis<batch>, Training<batch> and Prediction.
 */
template<> class Algorithm<batch> : public AlgorithmIfaceImpl
{
public:
    /** Default constructor */
    Algorithm() : _ac(0), _par(0), _in(0), _res(0) { }

    virtual ~Algorithm()
    {
        if(_ac)
        {
            delete _ac;
        }
    }

    /**
     * Validates parameters of the compute method
     */
    virtual services::Status checkComputeParams() = 0;

    Parameter * getBaseParameter() { return _par; }

protected:
    services::Status allocateResultMemory()
    {
        if(_res == 0)
            return allocateResult();
        return services::Status();
    }

    virtual void setParameter() {}

    virtual services::Status allocateResult() = 0;

    virtual Algorithm<batch> *cloneImpl() const = 0;

    daal::algorithms::AlgorithmContainerImpl<batch> *_ac;

    Parameter *_par;
    Input     *_in;
    Result    *_res;
};

/** @} */
} // namespace interface1
using interface1::Algorithm;

}
}
#endif
