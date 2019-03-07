/* file: implicit_als_model.cpp */
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
//  Implementation of the class defining the implicit als model
//--
*/

#include "implicit_als_model.h"
#include "daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace implicit_als
{

Model::Model() { }

services::Status Parameter::check() const
{
    if(nFactors == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, nFactorsStr()));
    }
    if(maxIterations == 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, maxIterationsStr()));
    }
    if(alpha < 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, alphaStr()));
    }
    if(lambda < 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, lambdaStr()));
    }
    if(preferenceThreshold < 0)
    {
        return services::Status(services::Error::create(services::ErrorIncorrectParameter, services::ParameterName, preferenceThresholdStr()));
    }
    return services::Status();
}

PartialModel::PartialModel(const data_management::NumericTablePtr &factors,
                           const data_management::NumericTablePtr &indices,
                           services::Status &st) :
    _factors(factors), _indices(indices) { }

/**
 * Constructs a partial implicit ALS model from the indices and factors stored in the numeric tables
 * \param[in] factors   Pointer to the numeric table with factors stored in row-major order
 * \param[in] indices   Pointer to the numeric table with the indices of factors
 * \param[out] stat     Status of the model construction
 * \return Partial implicit ALS model with the specified indices and factors
 */
PartialModelPtr PartialModel::create(const data_management::NumericTablePtr &factors,
                                     const data_management::NumericTablePtr &indices,
                                     services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(PartialModel, factors, indices);
}


}// namespace implicit_als
}// namespace algorithms
}// namespace daal
