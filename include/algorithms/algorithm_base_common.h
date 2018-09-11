/* file: algorithm_base_common.h */
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
namespace interface1
{

/**
 * @addtogroup base_algorithms
 * @{
 */

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

    virtual ~AlgorithmIface() {}

    /**
     * Validates parameters of the compute method
     */
    virtual services::Status checkComputeParams() = 0;

    /**
     * Validates result parameters of the compute method
     */
    virtual services::Status checkResult() = 0;

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const = 0;

    /**
     * Returns errors during the computations
     * \return Errors during the computations
     * \DAAL_DEPRECATED
     */
    virtual services::SharedPtr<services::ErrorCollection> getErrors() = 0;
};

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMIFACEIMPL"></a>
 *  \brief Implements the abstract interface AlgorithmIface. AlgorithmIfaceImpl is, in turn, the base class
 *         for the classes interfacing the major compute modes: batch, online and distributed
 */
class AlgorithmIfaceImpl : public AlgorithmIface
{
public:
    /** Default constructor */
    AlgorithmIfaceImpl() : _enableChecks(true)
    {
        getEnvironment();
    }

    virtual ~AlgorithmIfaceImpl() {}

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
     * For backward compatibility. Returns error collection of the algorithm
     * \return Error collection of the algorithm
     * \DAAL_DEPRECATED
     */
    services::SharedPtr<services::ErrorCollection> getErrors()
    {
        return _status.getCollection();
    }

private:
    bool _enableChecks;

protected:
    services::Status getEnvironment()
    {
        int cpuid = (int)daal::services::Environment::getInstance()->getCpuId();
        if(cpuid < 0)
            return services::Status(services::ErrorCpuNotSupported);
        _env.cpuid = cpuid;
        _env.cpuid_init_flag = true;
        return services::Status();
    }

    daal::services::Environment::env    _env;
    services::Status _status;
};

/** @} */
} // namespace interface1
using interface1::AlgorithmIface;
using interface1::AlgorithmIfaceImpl;

}
}
#endif
