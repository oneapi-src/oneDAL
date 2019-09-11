/* file: pivoted_qr.h */
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
//  Definition of Pivoted QR common types.
//--
*/
#ifndef __PIVOTED_QR_BATCH__
#define __PIVOTED_QR_BATCH__

#include "pivoted_qr_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace pivoted_qr
{

/**
 * Allocates memory for storing final results of the pivoted QR algorithm
 * \param[in] input        Pointer to input object
 * \param[in] parameter    Pointer to parameter
 * \param[in] method       Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    size_t m = static_cast<const Input *>(input)->get(data)->getNumberOfColumns();
    size_t n = static_cast<const Input *>(input)->get(data)->getNumberOfRows();

    set(matrixQ, HomogenNumericTable<algorithmFPType>::create(m, n, NumericTable::doAllocate, &s));
    set(matrixR, HomogenNumericTable<algorithmFPType>::create(m, m, NumericTable::doAllocate, &s));
    set(permutationMatrix, HomogenNumericTable<size_t>::create(m, 1, NumericTable::doAllocate, 0, &s));
    return s;
}

}// namespace pivoted_qr
}// namespace algorithms
}// namespace daal

#endif
