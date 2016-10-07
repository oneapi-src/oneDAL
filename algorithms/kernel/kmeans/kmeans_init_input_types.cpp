/* file: kmeans_init_input_types.cpp */
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
//  Implementation of kmeans classes.
//--
*/

#include "algorithms/kmeans/kmeans_init_types.h"
#include "daal_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface1
{

Input::Input() : InputIface(1) {}

/**
* Returns input objects for computing initial clusters for the K-Means algorithm
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
* Sets an input object for computing initial clusters for the K-Means algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the input object
*/
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns the number of features in the Input data table
* \return Number of features in the Input data table
*/
size_t Input::getNumberOfFeatures() const
{
    NumericTablePtr inTable = get(data);
    return inTable->getNumberOfColumns();
}

static bool isCSRMethod(int method)
{
    return (method == kmeans::init::deterministicCSR || method == kmeans::init::randomCSR
        || method == kmeans::init::plusPlusCSR || method == kmeans::init::parallelPlusCSR);
}

static bool isParallelPlusMethod(int method)
{
    return (method == kmeans::init::parallelPlusDense || method == kmeans::init::parallelPlusCSR);
}

/**
* Checks an input object for computing initial clusters for the K-Means algorithm
* \param[in] par     %Input object
* \param[in] method  Method of the algorithm
*/
void Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    if(isParallelPlusMethod(method))
    {
        //check parallel plus method parameters
        const daal::algorithms::kmeans::init::Parameter* prm = (const daal::algorithms::kmeans::init::Parameter*)(parameter);
        DAAL_CHECK_EX(prm->oversamplingFactor > 0, ErrorIncorrectParameter, ParameterName, oversamplingFactorStr());
        DAAL_CHECK_EX(prm->nRounds > 0, ErrorIncorrectParameter, ParameterName, nRoundsStr());
        size_t L(prm->oversamplingFactor*prm->nClusters);
        if(L*prm->nRounds <= prm->nClusters)
        {
            auto pError = services::Error::create(ErrorIncorrectParameter, ParameterName, nRoundsStr());
            pError->addStringDetail(ParameterName, oversamplingFactorStr());
            this->_errors->add(pError);
            return;
        }
    }

    if(isCSRMethod(method))
    {
        int expectedLayout = (int)NumericTableIface::csrArray;
        if (!checkNumericTable(get(data).get(), this->_errors.get(), dataStr(), 0, expectedLayout)) { return; }
    }
    else
    {
        if (!checkNumericTable(get(data).get(), this->_errors.get(), dataStr())) { return; }
    }
}

} // namespace interface1
} // namespace init
} // namespace kmeans
} // namespace algorithm
} // namespace daal
