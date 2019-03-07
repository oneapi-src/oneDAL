/* file: logitboost_model_fpt.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of class defining LogitBoost model.
//--
*/

#include "algorithms/boosting/logitboost_model.h"

namespace daal
{
namespace algorithms
{
namespace logitboost
{

/**
 *  Constructs the logitBoost %Model
 * \tparam modelFPType  Data type to store logitBoost model data, double or float
 * \param[in] dummy     Dummy variable for the templated constructor
 * \DAAL_DEPRECATED_USE{ Model::create }
 */
template <typename modelFPType>
DAAL_EXPORT Model::Model(size_t nFeatures, const Parameter *par, modelFPType dummy) :
    boosting::Model(nFeatures), _nIterations(par->maxIterations) {}

template DAAL_EXPORT Model::Model(size_t, const Parameter *, DAAL_FPTYPE);

}// namespace logitboost
}// namespace algorithms
}// namespace daal
