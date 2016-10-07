/* file: algorithm_base_mode_impl.h */
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

#ifndef __ALGORITHM_BASE_MODE_IMPL_H__
#define __ALGORITHM_BASE_MODE_IMPL_H__

#include "services/daal_defines.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/algorithm_base_mode_batch.h"

namespace daal
{
namespace algorithms
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * @addtogroup base_algorithms
 * @{
 */
/**
 * \brief Helper class that sets the status for the algorithm that specifies
 *        whether the algorithm can throw exceptions or not
 */
class CanThrowStatus
{
public:
    /**
    * Constructs object that handles "throw" or "not throw" status of the algorithm
    * \param[in] errors Error collection of the algorithm
    * \param[in] status Status of the algorithm. If true then the algorithm cat throw exceptions
    */
    CanThrowStatus(services::ErrorCollection *errors, bool status = false) :
        _errors(errors),
        _canThrowStatus(errors->setCanThrow(status))
    {}

    virtual ~CanThrowStatus()
    {
        _errors->setCanThrow(_canThrowStatus);
    }
protected:
    const bool _canThrowStatus;
    services::ErrorCollection *_errors;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMIMPL"></a>
 * \brief Provides implementations of the compute and finalizeCompute methods of the Algorithm class.
 *        The methods of the class support different computation modes: batch, distributed and online(see \ref ComputeMode)
 * \tparam mode Computation mode of the algorithm, \ref ComputeMode
 */
template<ComputeMode mode>
class AlgorithmImpl : public Algorithm<mode>
{
public:
    /** Deafult constructor */
    AlgorithmImpl() : wasSetup(false), resetFlag(true), wasFinalizeSetup(false), resetFinalizeFlag(true) {}

    virtual ~AlgorithmImpl()
    {
        resetCompute();
        resetFinalizeCompute();
    }

    /**
     * Computes final results of the algorithm in the %batch mode,
     * or partial results of the algorithm in %online and %distributed modes without possibility of throwing an exception.
     */
    void computeNoThrow()
    {
        CanThrowStatus noThrow(this->_errors.get());
        this->setParameter();

        this->_in->setErrorCollection(this->_errors);
        if(this->_par)
        {
            this->_par->setErrorCollection(this->_errors);
        }

        this->allocateInputMemory();
        if(this->_errors->size() != 0)
        {
            return;
        }

        if(this->isChecksEnabled())
        {
            this->checkComputeParams();
            if(this->_errors->size() != 0)
            {
                return;
            }
        }

        if(!this->allocatePartialResultMemory())
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }

        this->_ac->setArguments(this->_in,  this->_pres, this->_par);
        this->_ac->setErrorCollection(this->_errors);
        this->_pres->setErrorCollection(this->_errors);


        if(this->isChecksEnabled())
        {
            this->checkResult();
            if(this->_errors->size() != 0)
            {
                return;
            }
        }

        if(!this->getInitFlag())
        {
            this->initPartialResult();
            this->setInitFlag(true);
        }

        setupCompute();
        this->_ac->compute();
        resetCompute();
    }

    /**
     * Computes final results of the algorithm in the %batch mode,
     * or partial results of the algorithm in %online and %distributed modes.
     */
    void compute()
    {
        computeNoThrow();
        if(this->_errors->size() != 0)
        {
            this->throwIfPossible();
            return;
        }
    }

    /**
     * Computes final results of the algorithm using partial results in %online and %distributed modes.
     */
    void finalizeComputeNoThrow()
    {
        CanThrowStatus noThrow(this->_errors.get());
        if(this->isChecksEnabled())
        {
            this->checkPartialResult();
            if(this->_errors->size() != 0)
            {
                return;
            }
        }

        this->allocateResultMemory();
        if(this->_errors->size() != 0)
        {
            this->_errors->add(services::ErrorMemoryAllocationFailed);
            return;
        }

        this->_ac->setPartialResult(this->_pres);
        this->_ac->setResult(this->_res);
        this->_ac->setErrorCollection(this->_errors);

        if(this->_res)
        {
            this->_res->setErrorCollection(this->_errors);
        }

        if(this->isChecksEnabled())
        {
            this->checkFinalizeComputeParams();
            if(this->_errors->size() != 0)
            {
                return;
            }
        }

        setupFinalizeCompute();
        this->_ac->finalizeCompute();
        if(resetFinalizeFlag)
        {
            resetFinalizeCompute();
        }
    }

    /**
     * Computes final results of the algorithm using partial results in %online and %distributed modes.
     */
    void finalizeCompute()
    {
        finalizeComputeNoThrow();

        if(this->_errors->size() != 0)
        {
            this->throwIfPossible();
            return;
        }
    }

    /**
     * Validates parameters of the compute method
     */
    virtual void checkComputeParams() DAAL_C11_OVERRIDE
    {
        if (this->_par)
        {
            this->_par->check();
        }

        this->_in->check(this->_par, this->getMethod());

        if(this->_errors->size() != 0)
        {
            return;
        }
    }

    /**
     * Validates result parameters of the compute method
     */
    virtual void checkResult() DAAL_C11_OVERRIDE
    {
        if (this->_pres)
        {
            this->_pres->check(this->_in, this->_par, this->getMethod());
        }
        else
        {
            this->_errors->add(services::ErrorNullPartialResult);
        }

        if(this->_errors->size() != 0)
        {
            return;
        }
    }

    /**
     * Validates result parameters of the finalizeCompute method
     */
    virtual void checkPartialResult() DAAL_C11_OVERRIDE
    {
        if(this->_pres)
        {
            this->_pres->check(this->_par, this->getMethod());
        }
        else
        {
            this->_errors->add(services::ErrorNullPartialResult);
        }

        if(this->_errors->size() != 0)
        {
            return;
        }
    }

    /**
     * Validates parameters of the finalizeCompute method
     */
    virtual void checkFinalizeComputeParams() DAAL_C11_OVERRIDE
    {
        if(this->_res)
        {
            this->_res->check(this->_pres, this->_par, this->getMethod());
        }

        if(this->_errors->size() != 0)
        {
            return;
        }
    }

    void setupCompute()
    {
        if(!wasSetup)
        {
            this->_ac->setupCompute();
            wasSetup = true;
        }
    }

    void resetCompute()
    {
        if(wasSetup)
        {
            this->_ac->resetCompute();
            wasSetup = false;
        }
    }

    void enableResetOnCompute(bool flag)
    {
        resetFlag = flag;
    }

    void setupFinalizeCompute()
    {
        if(!wasFinalizeSetup)
        {
            this->_ac->setupFinalizeCompute();
            wasFinalizeSetup = true;
        }
    }

    void resetFinalizeCompute()
    {
        if(wasFinalizeSetup)
        {
            this->_ac->resetFinalizeCompute();
            wasFinalizeSetup = false;
        }
    }

    void enableResetOnFinalizeCompute(bool flag)
    {
        resetFinalizeFlag = flag;
    }

private:
    bool wasSetup;
    bool resetFlag;
    bool wasFinalizeSetup;
    bool resetFinalizeFlag;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHMIMPL_BATCH"></a>
 * \brief Provides implementations of the compute and checkComputeParams methods of the Algorithm<batch> class
 */
template<>
class DAAL_EXPORT AlgorithmImpl<batch> : public Algorithm<batch>
{
public:
    /** Deafult constructor */
    AlgorithmImpl() : wasSetup(false), resetFlag(true) {}

    virtual ~AlgorithmImpl()
    {
        resetCompute();
    }

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const = 0;

    /**
     * Computes final results of the algorithm in the %batch mode without possibility of throwing an exception.
     */
    void computeNoThrow();

    /**
     * Computes final results of the algorithm in the %batch mode.
     */
    void compute()
    {
        computeNoThrow();

        if(this->_errors->size() != 0)
        {
            this->throwIfPossible();
            return;
        }
    }
    /**
     * Validates parameters of the compute method
     */
    virtual void checkComputeParams() DAAL_C11_OVERRIDE
    {
        if (_par)
        {
            _par->check();

            if(this->_errors->size() != 0)
            {
                return;
            }
        }

        _in->check(_par, getMethod());

        if(this->_errors->size() != 0)
        {
            return;
        }
    }

    /**
     * Validates result parameters of the compute method
     */
    virtual void checkResult() DAAL_C11_OVERRIDE
    {
        if(_res)
        {
            _res->check(_in, _par, getMethod());
        }
        else
        {
            _errors->add(services::ErrorNullResult);
        }

        if(this->_errors->size() != 0)
        {
            return;
        }
    }

    void setupCompute()
    {
        if(!wasSetup)
        {
            this->_ac->setupCompute();
            wasSetup = true;
        }
    }

    void resetCompute()
    {
        if(wasSetup)
        {
            this->_ac->resetCompute();
            wasSetup = false;
        }
    }

    void enableResetOnCompute(bool flag)
    {
        resetFlag = flag;
    }

private:
    bool wasSetup;
    bool resetFlag;
};
/** @} */
} // namespace interface1
using interface1::AlgorithmImpl;

}
}
#endif
