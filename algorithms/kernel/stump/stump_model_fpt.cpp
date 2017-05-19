/* file: stump_model_fpt.cpp */
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
//  Implementation of the decision stump model constructor.
//--
*/

#include "algorithms/stump/stump_model.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
/**
 * Constructs the decision stump model
 * \tparam modelFPType  Data type to store decision stump model data, double or float
 * \param[in] dummy     Dummy variable for the templated constructor
 */
template<typename modelFPType>
DAAL_EXPORT Model::Model(modelFPType dummy) :
    weak_learner::Model(),
    _values(new data_management::Matrix<double>(3, 1, data_management::NumericTable::doAllocate))
{}

template<typename modelFPType>
DAAL_EXPORT modelFPType Model::getSplitValue()
{
    return (*_values)[0][0];
}

template<typename modelFPType>
DAAL_EXPORT void Model::setSplitValue(modelFPType splitValue)
{
    (*_values)[0][0] = splitValue;
}

template<typename modelFPType>
DAAL_EXPORT modelFPType Model::getLeftSubsetAverage()
{
    return (*_values)[0][1];
}

template<typename modelFPType>
DAAL_EXPORT void Model::setLeftSubsetAverage(modelFPType leftSubsetAverage)
{
    (*_values)[0][1] = leftSubsetAverage;
}

template<typename modelFPType>
DAAL_EXPORT modelFPType Model::getRightSubsetAverage()
{
    return (*_values)[0][2];
}

template<typename modelFPType>
DAAL_EXPORT void Model::setRightSubsetAverage(modelFPType rightSubsetAverage)
{
    (*_values)[0][2] = rightSubsetAverage;
}

template DAAL_EXPORT Model::Model(DAAL_FPTYPE dummy);
template DAAL_EXPORT DAAL_FPTYPE Model::getSplitValue();
template DAAL_EXPORT void Model::setSplitValue(DAAL_FPTYPE splitValue);
template DAAL_EXPORT DAAL_FPTYPE Model::getLeftSubsetAverage();
template DAAL_EXPORT void Model::setLeftSubsetAverage(DAAL_FPTYPE leftSubsetAverage);
template DAAL_EXPORT DAAL_FPTYPE Model::getRightSubsetAverage();
template DAAL_EXPORT void Model::setRightSubsetAverage(DAAL_FPTYPE rightSubsetAverage);

}// namespace brownboost
}// namespace algorithms
}// namespace daal
