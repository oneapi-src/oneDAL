/** file algorithm_base_impl.cpp */
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

#include "algorithm_base_impl.h"

namespace daal
{
namespace algorithms
{
/**
 * Computes final results of the algorithm in the %batch mode without possibility of throwing an exception.
 */
void AlgorithmImpl<batch>::computeNoThrow()
{
    this->setParameter();

    this->_in->setErrorCollection(this->_errors);

    if(this->_par)
    {
        this->_par->setErrorCollection(this->_errors);
    }

    if(this->isChecksEnabled())
    {
        this->checkComputeParams();
        if(this->_errors->size() != 0)
        {
            return;
        }
    }

    this->allocateInputMemory();
    if(this->_errors->size() != 0)
    {
        return;
    }

    this->allocateResultMemory();
    if(this->_errors->size() != 0)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed);
        return;
    }

    this->_ac->setArguments(this->_in, this->_res, this->_par);
    this->_ac->setErrorCollection(this->_errors);

    this->_res->setErrorCollection(this->_errors);

    if(this->isChecksEnabled())
    {
        this->checkResult();
        if(this->_errors->size() != 0)
        {
            return;
        }
    }

    this->_ac->compute();

    _res = this->_ac->getResult();
}

} // namespace daal
} // namespace algorithms
