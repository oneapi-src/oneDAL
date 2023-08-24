/* file: svd_distr_step3_result.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
//  Implementation of svd classes.
//--
*/

#include "algorithms/svd/svd_types.h"
#include "src/services/serialization_utils.h"
#include "src/services/daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
__DAAL_REGISTER_SERIALIZATION_CLASS(DistributedPartialResultStep3, SERIALIZATION_SVD_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID);

/** Default constructor */
DistributedPartialResultStep3::DistributedPartialResultStep3() : daal::algorithms::PartialResult(lastDistributedPartialResultStep3Id + 1) {}

/**
 * Returns the result of the SVD algorithm with the matrix Q calculated
 * \param[in] id    Identifier of the result
 * \return          Result that corresponds to the given identifier
 */
ResultPtr DistributedPartialResultStep3::get(DistributedPartialResultStep3Id id) const
{
    return staticPointerCast<Result, SerializationIface>(Argument::get(id));
}

/**
 * Sets Result object to store the result of the SVD algorithm
 * \param[in] id    Identifier of the result
 * \param[in] value Pointer to the Result object
 */
void DistributedPartialResultStep3::set(DistributedPartialResultStep3Id id, const ResultPtr & value)
{
    Argument::set(id, staticPointerCast<SerializationIface, Result>(value));
}

/**
 * Checks partial results of the algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to parameters
 * \param[in] method Computation method
 */
Status DistributedPartialResultStep3::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    DistributedStep3Input * svdInput = static_cast<DistributedStep3Input *>(const_cast<daal::algorithms::Input *>(input));
    Parameter * svdPar               = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));
    int unexpectedLayouts            = (int)packed_mask;
    DataCollectionPtr qCollection    = svdInput->get(inputOfStep3FromStep1);
    size_t nFeatures                 = 0;
    size_t nVectors                  = 0;
    size_t qSize                     = qCollection->size();
    for (size_t i = 0; i < qSize; i++)
    {
        NumericTable * numTableInQCollection = static_cast<NumericTable *>((*qCollection)[i].get());
        nFeatures                            = numTableInQCollection->getNumberOfColumns();
        nVectors += numTableInQCollection->getNumberOfRows();
    }
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        if (get(finalResultFromStep3))
        {
            Status s = checkNumericTable(get(finalResultFromStep3)->get(leftSingularMatrix).get(), leftSingularMatrixStr(), unexpectedLayouts, 0,
                                         nFeatures, nVectors);
            if (!s)
            {
                return s;
            }
        }
    }
    return Status();
}

/**
 * Checks partial results of the algorithm
 * \param[in] parameter Pointer to parameters
 * \param[in] method Computation method
 */
Status DistributedPartialResultStep3::check(const daal::algorithms::Parameter * parameter, int method) const
{
    Parameter * svdPar = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));
    if (svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        if (get(finalResultFromStep3))
        {
            int unexpectedLayouts = (int)packed_mask;
            Status s = checkNumericTable(get(finalResultFromStep3)->get(leftSingularMatrix).get(), leftSingularMatrixStr(), unexpectedLayouts);
            if (!s)
            {
                return s;
            }
        }
    }
    return Status();
}

} // namespace svd
} // namespace algorithms
} // namespace daal
