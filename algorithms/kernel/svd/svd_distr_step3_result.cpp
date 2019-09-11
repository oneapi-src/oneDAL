/* file: svd_distr_step3_result.cpp */
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
//  Implementation of svd classes.
//--
*/

#include "algorithms/svd/svd_types.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
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
void DistributedPartialResultStep3::set(DistributedPartialResultStep3Id id, const ResultPtr &value)
{
    Argument::set(id, staticPointerCast<SerializationIface, Result>(value));
}

/**
 * Checks partial results of the algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to parameters
 * \param[in] method Computation method
 */
Status DistributedPartialResultStep3::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const
{
    DistributedStep3Input *svdInput   = static_cast<DistributedStep3Input *>(const_cast<daal::algorithms::Input *>(input  ));
    Parameter *svdPar   = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));
    int unexpectedLayouts = (int)packed_mask;
    DataCollectionPtr qCollection = svdInput->get(inputOfStep3FromStep1);
    size_t nFeatures = 0;
    size_t nVectors = 0;
    size_t qSize = qCollection->size();
    for(size_t i = 0 ; i < qSize ; i++)
    {
        NumericTable  *numTableInQCollection = static_cast<NumericTable *>((*qCollection)[i].get());
        nFeatures  = numTableInQCollection->getNumberOfColumns();
        nVectors += numTableInQCollection->getNumberOfRows();
    }
    if(svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        if(get(finalResultFromStep3))
        {
            Status s = checkNumericTable(get(finalResultFromStep3)->get(leftSingularMatrix).get(), leftSingularMatrixStr(), unexpectedLayouts, 0, nFeatures, nVectors);
            if(!s) { return s; }
        }
    }
    return Status();
}

/**
 * Checks partial results of the algorithm
 * \param[in] parameter Pointer to parameters
 * \param[in] method Computation method
 */
Status DistributedPartialResultStep3::check(const daal::algorithms::Parameter *parameter, int method) const
{
    Parameter *svdPar = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));
    if(svdPar->leftSingularMatrix == requiredInPackedForm)
    {
        if(get(finalResultFromStep3))
        {
            int unexpectedLayouts = (int)packed_mask;
            Status s = checkNumericTable(get(finalResultFromStep3)->get(leftSingularMatrix).get(), leftSingularMatrixStr(), unexpectedLayouts);
            if(!s) { return s; }
        }
    }
    return Status();
}

} // namespace interface1
} // namespace svd
} // namespace algorithm
} // namespace daal
