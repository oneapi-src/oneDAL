/* file: qr_dense_default_batch.h */
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
//  Implementation of qr algorithm and types methods.
//--
*/
#ifndef __QR_DENSE_DEFAULT_BATCH__
#define __QR_DENSE_DEFAULT_BATCH__

#include "qr_types.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace interface1
{

/**
 * Allocates memory for storing final results of the QR decomposition algorithm
 * \param[in] input     Pointer to input object
 * \param[in] parameter Pointer to parameter
 * \param[in] method    Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Input *in = static_cast<const Input *>(input);
    return allocateImpl<algorithmFPType>(in->get(data)->getNumberOfColumns(), in->get(data)->getNumberOfRows());
}

/**
 * Allocates memory for storing final results of the QR decomposition algorithm
 * \param[in] partialResult  Pointer to partial result
 * \param[in] parameter      Pointer to the result
 * \param[in] method         Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter, const int method)
{
    const OnlinePartialResult *in = static_cast<const OnlinePartialResult *>(partialResult);
    return allocateImpl<algorithmFPType>(in->getNumberOfColumns(), in->getNumberOfRows());
}

/**
 * Allocates memory for storing final results of the QR decomposition algorithm
 * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
 * \param[in]  m  Number of columns in the input data set
 * \param[in]  n  Number of rows in the input data set
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocateImpl(size_t m, size_t n)
{
    Status s;
    if(n == 0)
    {
        Argument::set(matrixQ, data_management::SerializationIfacePtr());
    }
    else
    {
        Argument::set(matrixQ, data_management::HomogenNumericTable<algorithmFPType>::create(m, n, data_management::NumericTable::doAllocate, &s));
    }
    Argument::set(matrixR, data_management::HomogenNumericTable<algorithmFPType>::create(m, m, data_management::NumericTable::doAllocate, &s));
    return s;
}

}// namespace interface1
}// namespace qr
}// namespace algorithms
}// namespace daal

#endif
