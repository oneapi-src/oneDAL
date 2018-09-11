/* file: cholesky_batch.h */
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
//  Implementation of cholesky algorithm and types methods.
//--
*/
#ifndef __CHOLESKY_BATCH__
#define __CHOLESKY_BATCH__

#include "cholesky_types.h"

using namespace daal::data_management;
namespace daal
{
namespace algorithms
{
namespace cholesky
{
namespace interface1
{
/**
 * Allocates memory to store the results of Cholesky decomposition
 * \param[in] input  Pointer to the input structure
 * \param[in] par    Pointer to the parameter structure
 * \param[in] method Computation method of the algorithm
 */
template <typename algFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    Input *algInput = static_cast<Input *>(const_cast<daal::algorithms::Input *>(input));
    size_t nFeatures = algInput->get(data)->getNumberOfColumns();
    services::Status status;
    set(choleskyFactor, HomogenNumericTable<algFPType>::create(nFeatures, nFeatures, NumericTable::doAllocate, &status));
    return status;
}

}// namespace interface1
}// namespace cholesky
}// namespace algorithms
}// namespace daal

#endif
