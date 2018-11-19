/* file: svd_dense_default_batch.h */
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
//  Implementation of svd algorithm and types methods.
//--
*/
#ifndef __SVD_DENSE_DEFAULT_BATCH__
#define __SVD_DENSE_DEFAULT_BATCH__

#include "svd_types.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
{

/**
 * Allocates memory to store final results of the SVD algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Input *in = static_cast<const Input *>(input);
    return allocateImpl<algorithmFPType>(in->get(data)->getNumberOfColumns(), in->get(data)->getNumberOfRows());
}

/**
 * Allocates memory to store final results of the SVD algorithm
 * \param[in] partialResult  Pointer to the partial result
 * \param[in] parameter      Pointer to the parameter
 * \param[in] method         Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter, const int method)
{
    const OnlinePartialResult *in = static_cast<const OnlinePartialResult *>(partialResult);
    return allocateImpl<algorithmFPType>(in->getNumberOfColumns(), in->getNumberOfRows());
}

/**
 * Allocates memory to store final results of the SVD algorithm
 * \tparam     algorithmFPType  Data type to use for storage in the resulting HomogenNumericTable
 * \param[in]  m  Number of columns in the input data set
 * \param[in]  n  Number of rows in the input data set
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocateImpl(size_t m, size_t n)
{
    Status st;
    set(singularValues, HomogenNumericTable<algorithmFPType>::create(m, 1, NumericTable::doAllocate, &st));
    set(rightSingularMatrix, HomogenNumericTable<algorithmFPType>::create(m, m, NumericTable::doAllocate, &st));
    if(n != 0)
    {
        set(leftSingularMatrix, HomogenNumericTable<algorithmFPType>::create(m, n, NumericTable::doAllocate, &st));
    }
    return st;
}

}// namespace interface1
}// namespace svd
}// namespace algorithms
}// namespace daal

#endif
