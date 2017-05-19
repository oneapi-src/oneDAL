/** file algorithm_base_impl.cpp */
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
//
//--
*/

#include "algorithm_base.h"
#include "algorithm_base_mode_impl.h"

namespace daal
{
namespace algorithms
{
template<ComputeMode mode>
services::Status AlgorithmImpl<mode>::computeNoThrow()
{
    this->setParameter();

    services::Status s = this->allocateInputMemory();
    if(!s)
        return s;

    if(this->isChecksEnabled())
    {
        s = this->checkComputeParams();
        if(!s)
            return s;
    }

    if(!this->allocatePartialResultMemory())
        return services::Status(services::ErrorMemoryAllocationFailed);

    this->_ac->setArguments(this->_in, this->_pres, this->_par);
    this->_ac->setErrorCollection(this->_errors);

    if(this->isChecksEnabled())
    {
        s = this->checkResult();
        if(!s)
            return s;
    }

    if(!this->getInitFlag())
    {
        s = this->initPartialResult();
        if(!s)
            return s;
        this->setInitFlag(true);
    }

    s = setupCompute();
    if(s)
        s = this->_ac->compute();
    s |= resetCompute();
    return s;
}

/**
 * Computes final results of the algorithm in the %batch mode without possibility of throwing an exception.
 */
services::Status AlgorithmImpl<batch>::computeNoThrow()
{
    this->setParameter();

    if(this->isChecksEnabled())
    {
        services::Status _s = this->checkComputeParams();
        if(!_s)
            return _s;
    }

    services::Status s = this->allocateInputMemory();
    if(!s)
        return s;

    s = this->allocateResultMemory();
    if(!s)
        return s.add(services::ErrorMemoryAllocationFailed);

    this->_ac->setArguments(this->_in, this->_res, this->_par);
    this->_ac->setErrorCollection(this->_errors);

    if(this->isChecksEnabled())
    {
        s = this->checkResult();
        if(!s)
            return s;
    }

    s = setupCompute();
    if(s)
        s |= this->_ac->compute();
    if(resetFlag)
        s |= resetCompute();
    _res = this->_ac->getResult();
    return s;
}

template class interface1::AlgorithmImpl<online>;
template class interface1::AlgorithmImpl<distributed>;
} // namespace daal
} // namespace algorithms
