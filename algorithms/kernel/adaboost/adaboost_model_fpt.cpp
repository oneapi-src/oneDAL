/* file: adaboost_model_fpt.cpp */
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
//  Implementation of class defining Ada Boost model
//--
*/

#include "algorithms/boosting/adaboost_model.h"

namespace daal
{
namespace algorithms
{
namespace adaboost
{

/**
 * Constructs the AdaBoost model
 * \tparam modelFPType  Data type to store AdaBoost model data, double or float
 * \param[in] dummy     Dummy variable for the templated constructor
 * \DAAL_DEPRECATED_USE{ Model::create }
 */
template <typename modelFPType>
DAAL_EXPORT Model::Model(size_t nFeatures, modelFPType dummy) : boosting::Model(nFeatures)
{
    _alpha = data_management::NumericTablePtr(new data_management::HomogenNumericTable<modelFPType>(NULL, 1, 0));
}

template <typename modelFPType>
DAAL_EXPORT Model::Model(size_t nFeatures, modelFPType dummy, services::Status &st) : boosting::Model(nFeatures, st)
{
    if (!st) { return; }
    _alpha = data_management::HomogenNumericTable<modelFPType>::create(NULL, 1, 0, &st);
}

/**
 * Constructs the AdaBoost model
 * \tparam modelFPType   Data type to store AdaBoost model data, double or float
 * \param[in]  nFeatures Number of features in the dataset
 * \param[out] stat      Status of the model construction
 */
template<typename modelFPType>
DAAL_EXPORT ModelPtr Model::create(size_t nFeatures, services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nFeatures, (modelFPType)0 /* dummy */);
}

template DAAL_EXPORT Model::Model(size_t, DAAL_FPTYPE);
template DAAL_EXPORT Model::Model(size_t, DAAL_FPTYPE, services::Status&);
template DAAL_EXPORT ModelPtr Model::create<DAAL_FPTYPE>(size_t, services::Status*);

}// namespace adaboost
}// namespace algorithms
}// namespace daal
