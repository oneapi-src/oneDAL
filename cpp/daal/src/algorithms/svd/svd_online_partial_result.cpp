/* file: svd_online_partial_result.cpp */
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
__DAAL_REGISTER_SERIALIZATION_CLASS(OnlinePartialResult, SERIALIZATION_SVD_ONLINE_PARTIAL_RESULT_ID);

/** Default constructor */
OnlinePartialResult::OnlinePartialResult() : daal::algorithms::PartialResult(lastPartialResultId + 1) {}

/**
 * Returns partial result of the SVD algorithm
 * \param[in] id    Identifier of the partial result
 * \return          Partial result that corresponds to the given identifier
 */
DataCollectionPtr OnlinePartialResult::get(PartialResultId id) const
{
    return staticPointerCast<DataCollection, SerializationIface>(Argument::get(id));
}

/**
 * Sets an input object for the SVD algorithm
 * \param[in] id    Identifier of partial result
 * \param[in] value Pointer to the partial result
 */
void OnlinePartialResult::set(PartialResultId id, const DataCollectionPtr & value)
{
    Argument::set(id, value);
}

/**
 * Checks parameters of the algorithm
 * \param[in] input Input of the algorithm
 * \param[in] parameter Pointer to the parameters
 * \param[in] method Computation method
 */
Status OnlinePartialResult::check(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, int method) const
{
    Input * svdInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t nFeatures, nRows;
    svdInput->getNumberOfColumns(&nFeatures);
    svdInput->getNumberOfRows(&nRows);
    return checkImpl(parameter, method, nFeatures, nRows);
}

/**
 * Checks parameters of the algorithm
 * \param[in] parameter Pointer to the parameters
 * \param[in] method Computation method
 */
Status OnlinePartialResult::check(const daal::algorithms::Parameter * parameter, int method) const
{
    return checkImpl(parameter, method, 0, 0);
}

/**
 * Returns the number of columns in the input data set
 * \return Number of columns in the input data set
 */
size_t OnlinePartialResult::getNumberOfColumns() const
{
    DataCollection * rCollection = get(outputOfStep1ForStep2).get();
    return static_cast<NumericTable *>((*rCollection)[0].get())->getNumberOfColumns();
}

/**
 * Returns the number of rows in the input data set
 * \return Number of rows in the input data set
 */
size_t OnlinePartialResult::getNumberOfRows() const
{
    DataCollection * qCollection = get(outputOfStep1ForStep3).get();
    size_t np                    = qCollection->size();
    size_t n                     = 0;

    /* we need V matrices */
    for (size_t i = 0; i < np; i++)
    {
        n += static_cast<NumericTable *>((*qCollection)[i].get())->getNumberOfRows();
    }

    return n;
}

Status OnlinePartialResult::checkImpl(const daal::algorithms::Parameter * parameter, int method, size_t nFeatures, size_t nVectors) const
{
    DataCollectionPtr qCollection = get(outputOfStep1ForStep3);
    DataCollectionPtr rCollection = get(outputOfStep1ForStep2);
    Parameter * svdPar            = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));
    if (svdPar->leftSingularMatrix != notRequired)
    {
        DAAL_CHECK_EX(qCollection, ErrorNullOutputDataCollection, ArgumentName, outputOfStep1ForStep3Str());
    }
    DAAL_CHECK_EX(rCollection, ErrorNullOutputDataCollection, ArgumentName, outputOfStep1ForStep2Str());

    size_t nodeSize = rCollection->size();
    if (nodeSize == 0)
    {
        DAAL_CHECK_EX(nVectors > 0, ErrorIncorrectNumberOfElementsInResultCollection, ArgumentName, outputOfStep1ForStep2Str());
        return Status();
    }

    if (svdPar->leftSingularMatrix != notRequired)
    {
        DAAL_CHECK_EX(nodeSize == qCollection->size(), ErrorIncorrectNumberOfElementsInResultCollection, ArgumentName, outputOfStep1ForStep3Str());
    }

    DAAL_CHECK_EX((*rCollection)[0], ErrorNullNumericTable, ArgumentName, rCollectionStr());

    NumericTablePtr firstNumTableInRCollection = NumericTable::cast((*rCollection)[0]);
    DAAL_CHECK_EX(firstNumTableInRCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, outputOfStep1ForStep2Str());

    Status s = checkNumericTable(firstNumTableInRCollection.get(), rCollectionStr());
    if (!s)
    {
        return s;
    }

    if (nFeatures == 0) nFeatures = firstNumTableInRCollection->getNumberOfColumns();
    for (size_t i = 0; i < nodeSize; i++)
    {
        DAAL_CHECK_EX((*rCollection)[i], ErrorNullNumericTable, ArgumentName, rCollectionStr());
        NumericTablePtr numTableInRCollection = NumericTable::cast((*rCollection)[i]);
        DAAL_CHECK_EX(numTableInRCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, outputOfStep1ForStep2Str());

        int unexpectedLayouts = (int)packed_mask;
        s |= checkNumericTable(numTableInRCollection.get(), rCollectionStr(), unexpectedLayouts, 0, nFeatures, nFeatures);
        if (!s)
        {
            return s;
        }

        if (svdPar->leftSingularMatrix != notRequired)
        {
            DAAL_CHECK_EX((*qCollection)[i], ErrorNullNumericTable, ArgumentName, qCollectionStr());
            NumericTablePtr numTableInQCollection = NumericTable::cast((*qCollection)[i]);
            DAAL_CHECK_EX(numTableInQCollection, ErrorIncorrectElementInNumericTableCollection, ArgumentName, outputOfStep1ForStep3Str());

            s |= checkNumericTable(numTableInQCollection.get(), qCollectionStr(), unexpectedLayouts, 0, nFeatures);
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
