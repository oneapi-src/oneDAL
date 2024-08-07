/* file: qr_dense_default_batch.h */
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
//  Implementation of qr algorithm and types methods.
//--
*/
#ifndef __QR_DENSE_DEFAULT_BATCH__
#define __QR_DENSE_DEFAULT_BATCH__

#include "algorithms/qr/qr_types.h"

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
Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter, const int method)
{
    const Input * in = static_cast<const Input *>(input);
    return allocateImpl<algorithmFPType>(in->get(data)->getNumberOfColumns(), in->get(data)->getNumberOfRows());
}

/**
 * Allocates memory for storing final results of the QR decomposition algorithm
 * \param[in] partialResult  Pointer to partial result
 * \param[in] parameter      Pointer to the result
 * \param[in] method         Algorithm method
 */
template <typename algorithmFPType>
Status Result::allocate(const daal::algorithms::PartialResult * partialResult, daal::algorithms::Parameter * parameter, const int method)
{
    const OnlinePartialResult * in = static_cast<const OnlinePartialResult *>(partialResult);
    return allocateImpl<algorithmFPType>(in->getNumberOfColumns(), in->getNumberOfRows());
}

/**
 * Allocates memory for storing final results of the QR decomposition algorithm
 * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
 * \param[in]  m  Number of columns in the input data set
 * \param[in]  n  Number of rows in the input data set
 */
template <typename algorithmFPType>
Status Result::allocateImpl(size_t m, size_t n)
{
    Status s;
    if (n == 0)
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

} // namespace interface1
} // namespace qr
} // namespace algorithms
} // namespace daal

#endif
