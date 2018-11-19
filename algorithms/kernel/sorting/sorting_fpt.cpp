/* file: sorting_fpt.cpp */
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
//  Implementation of sorting algorithm and types methods.
//--
*/

#include "sorting_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace sorting
{
namespace interface1
{
/**
 * Allocates memory to store final results of the sorting algorithms
 * \param[in] input     Input objects for the sorting algorithm
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const int method)
{
    const Input *in = static_cast<const Input *>(input);

    const size_t nFeatures = in->get(data)->getNumberOfColumns();
    const size_t nVectors = in->get(data)->getNumberOfRows();
    services::Status st;
    set(sortedData, HomogenNumericTable<algorithmFPType>::create(nFeatures, nVectors, NumericTable::doAllocate, &st));
    return st;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const int method);

}// namespace interface1
}// namespace sorting
}// namespace algorithms
}// namespace daal
