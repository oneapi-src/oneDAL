/* file: dbscan_result.h */
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
//  Implementation of dbscan classes.
//--
*/

#ifndef __DBSCAN_RESULT__
#define __DBSCAN_RESULT__

#include "dbscan_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace dbscan
{

/**
 * Allocates memory to store the results of the DBSCAN algorithm
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Input * const algInput = static_cast<const Input *>(input);
    const Parameter *par = static_cast<const Parameter *>(parameter);

    const size_t nRows = algInput->get(data)->getNumberOfRows();
    const size_t nFeatures = algInput->get(data)->getNumberOfColumns();

    services::Status status;
    set(assignments, HomogenNumericTable<int>::create(1, nRows, NumericTable::doAllocate, &status));
    set(nClusters,   HomogenNumericTable<int>::create(1, 1, NumericTable::doAllocate, &status));

    if (par->resultsToCompute & computeCoreIndices)
    {
        set(coreIndices, HomogenNumericTable<int>::create(1, 0, NumericTable::notAllocate, &status));
    }

    if (par->resultsToCompute & computeCoreObservations)
    {
        set(coreObservations, HomogenNumericTable<algorithmFPType>::create(nFeatures, 0, NumericTable::notAllocate, &status));
    }

    return status;
}

} // namespace dbscan
} // namespace algorithms
} // namespace daal

#endif
