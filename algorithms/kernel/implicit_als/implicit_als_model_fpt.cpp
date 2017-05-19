/* file: implicit_als_model_fpt.cpp */
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
//  Implementation of the class defining the implicit als model
//--
*/

#include "implicit_als_model.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{

template<typename modelFPType>
DAAL_EXPORT Model::Model(size_t nUsers, size_t nItems, const Parameter &parameter, modelFPType dummy)
{
    const size_t nFactors = parameter.nFactors;
    _usersFactors.reset(new data_management::HomogenNumericTable<modelFPType>(nFactors, nUsers, data_management::NumericTableIface::doAllocate, 0));
    _itemsFactors.reset(new data_management::HomogenNumericTable<modelFPType>(nFactors, nItems, data_management::NumericTableIface::doAllocate, 0));
}

template DAAL_EXPORT Model::Model(size_t nUsers, size_t nItems, const Parameter &par, DAAL_FPTYPE dummy);

}// namespace implicit_als
}// namespace algorithms
}// namespace daal
