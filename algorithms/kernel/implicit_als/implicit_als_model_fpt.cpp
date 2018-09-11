/* file: implicit_als_model_fpt.cpp */
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

template<typename modelFPType>
DAAL_EXPORT Model::Model(size_t nUsers, size_t nItems, const Parameter &parameter, modelFPType dummy, services::Status &st)
{
    using namespace daal::data_management;
    const size_t nFactors = parameter.nFactors;

    _usersFactors = HomogenNumericTable<modelFPType>::create(nFactors, nUsers, NumericTableIface::doAllocate, 0, &st);
    if (!st) { return; }

    _itemsFactors = HomogenNumericTable<modelFPType>::create(nFactors, nItems, NumericTableIface::doAllocate, 0, &st);
    if (!st) { return; }
}

/**
 * Constructs the implicit ALS model
 * \param[in]  nUsers    Number of users in the input data set
 * \param[in]  nItems    Number of items in the input data set
 * \param[in]  parameter Implicit ALS parameters
 * \param[out] stat      Status of the model construction
 */
template<typename modelFPType>
DAAL_EXPORT ModelPtr Model::create(size_t nUsers, size_t nItems,
                                   const Parameter &parameter,
                                   services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nUsers, nItems, parameter, (modelFPType)0 /* dummy */);
}

template DAAL_EXPORT Model::Model(size_t, size_t, const Parameter&, DAAL_FPTYPE);
template DAAL_EXPORT Model::Model(size_t, size_t, const Parameter&, DAAL_FPTYPE, services::Status&);
template DAAL_EXPORT ModelPtr Model::create<DAAL_FPTYPE>(size_t, size_t,
                                                         const Parameter&, services::Status*);

}// namespace implicit_als
}// namespace algorithms
}// namespace daal
