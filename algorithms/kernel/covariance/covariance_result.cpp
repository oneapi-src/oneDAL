/* file: covariance_result.cpp */
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
//  Implementation of covariance algorithm and types methods.
//--
*/

#include "covariance_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace covariance
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_COVARIANCE_RESULT_ID);
Result::Result() : daal::algorithms::Result(lastResultId + 1)
    {}

/**
    * Returns the final result of the correlation or variance-covariance matrix algorithm
    * \param[in] id   Identifier of the result, \ref ResultId
    * \return Final result that corresponds to the given identifier
    */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
    * Sets the result of the correlation or variance-covariance matrix algorithm
    * \param[in] id        Identifier of the result
    * \param[in] value     Pointer to the object
    */
void Result::set(ResultId id, const NumericTablePtr &value)
{
    Argument::set(id, value);
}

/**
    * Check correctness of the result
    * \param[in] partialResult     Pointer to the partial result arguments structure
    * \param[in] parameter         Pointer to the structure of the parameters of the algorithm
    * \param[in] method            Computation method
    */
services::Status Result::check(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter,
            int method) const
{
    const PartialResult *pres = static_cast<const PartialResult *>(partialResult);
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);
    size_t nFeatures = pres->getNumberOfFeatures();

    return checkImpl(nFeatures, algParameter->outputMatrixType);
}

/**
    * Check correctness of the result
    * \param[in] input     Pointer to the structure with input objects
    * \param[in] parameter Pointer to the structure of algorithm parameters
    * \param[in] method    Computation method
    */
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter,
            int method) const
{
    const Parameter *algParameter = static_cast<const Parameter *>(parameter);

    size_t nFeatures = (static_cast<const InputIface *>(input))->getNumberOfFeatures();

    return checkImpl(nFeatures, algParameter->outputMatrixType);
}

services::Status Result::checkImpl(size_t nFeatures, OutputMatrixType outputMatrixType) const
{
    int unexpectedLayouts = (int)NumericTableIface::csrArray |
                            (int)NumericTableIface::upperPackedTriangularMatrix |
                            (int)NumericTableIface::lowerPackedTriangularMatrix;
    services::Status s;
    if (outputMatrixType == covarianceMatrix)
    {
        /* Check covariance matrix */
        s |= checkNumericTable(get(covariance).get(),  covarianceStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
        if(!s) return s;
    }
    else if (outputMatrixType == correlationMatrix)
    {
        /* Check correlation matrix */
        s |= checkNumericTable(get(correlation).get(),  correlationStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
        if(!s) return s;
    }

    unexpectedLayouts |= (int)NumericTableIface::upperPackedSymmetricMatrix |
                            (int)NumericTableIface::lowerPackedSymmetricMatrix;

    /* Check mean vector */
    s |= checkNumericTable(get(mean).get(), meanStr(), unexpectedLayouts, 0, nFeatures, 1);
    return s;
}

}//namespace interface1

}//namespace covariance
}// namespace algorithms
}// namespace daal
