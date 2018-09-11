/* file: moments_online.h */
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
//  Implementation of LowOrderMoments algorithm and types methods
//--
*/
#ifndef __MOMENTS_ONLINE__
#define __MOMENTS_ONLINE__

#include "low_order_moments_types.h"
#include "service_numeric_table.h"

using namespace daal::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace low_order_moments
{

/**
 * Allocates memory to store partial results of the low order %moments algorithm
 * \param[in] input     Pointer to the structure with input objects
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    size_t nFeatures = 0;
    DAAL_CHECK_STATUS(s, static_cast<const InputIface *>(input)->getNumberOfColumns(nFeatures));

    set(nObservations, HomogenNumericTable<size_t>::create(1, 1, NumericTable::doAllocate, &s));
    for(size_t i = 1; i < lastPartialResultId + 1; i++)
    {
        Argument::set(i, HomogenNumericTable<algorithmFPType>::create(nFeatures, 1, NumericTable::doAllocate, &s));
    }
    return s;
}

template <typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult::initialize(const daal::algorithms::Input *_in, const daal::algorithms::Parameter *parameter, const int method)
{
    Input *input = const_cast<Input *>(static_cast<const Input *>(_in));

    services::Status s;
    DAAL_CHECK_STATUS(s, get(nObservations)->assign((algorithmFPType)0.0))
    DAAL_CHECK_STATUS(s, get(partialSum)->assign((algorithmFPType)0.0))
    DAAL_CHECK_STATUS(s, get(partialSumSquares)->assign((algorithmFPType)0.0))
    DAAL_CHECK_STATUS(s, get(partialSumSquaresCentered)->assign((algorithmFPType)0.0))

    ReadRows<algorithmFPType, sse2> dataBlock(input->get(data).get(), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(dataBlock)
    const algorithmFPType* firstRow = dataBlock.get();

    WriteOnlyRows<algorithmFPType, sse2> partialMinimumBlock(get(partialMinimum).get(), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(partialMinimumBlock)
    algorithmFPType* partialMinimumArray = partialMinimumBlock.get();

    WriteOnlyRows<algorithmFPType, sse2> partialMaximumBlock(get(partialMaximum).get(), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(partialMaximumBlock)
    algorithmFPType* partialMaximumArray = partialMaximumBlock.get();

    size_t nColumns = input->get(data)->getNumberOfColumns();

    for(size_t j = 0; j < nColumns; j++)
    {
        partialMinimumArray[j] = firstRow[j];
        partialMaximumArray[j] = firstRow[j];
    }
    return s;
}

}// namespace low_order_moments
}// namespace algorithms
}// namespace daal

#endif
