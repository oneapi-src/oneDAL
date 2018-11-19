/* file: qr_dense_default_online.h */
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
#ifndef __QR_DENSE_DEFAULT_ONLINE__
#define __QR_DENSE_DEFAULT_ONLINE__

#include "qr_types.h"
#include "daal_strings.h"

using namespace daal::services;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace interface1
{

/**
 * Allocates memory for storing partial results of the QR decomposition algorithm
 * \param[in] input     Pointer to input object
 * \param[in] parameter Pointer to parameter
 * \param[in] method    Algorithm method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status OnlinePartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    set(outputOfStep1ForStep3, DataCollectionPtr(new DataCollection()));
    set(outputOfStep1ForStep2, DataCollectionPtr(new DataCollection()));
    return Status();
}

template <typename algorithmFPType>
DAAL_EXPORT Status OnlinePartialResult::initialize(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    get(outputOfStep1ForStep3)->clear();
    get(outputOfStep1ForStep2)->clear();
    return Status();
}

/**
 * Allocates additional memory for storing partial results of the QR decomposition algorithm for each subsequent call to compute method
 * \tparam     algorithmFPType  Data type to be used for storage in resulting HomogenNumericTable
 * \param[in]  m  Number of columns in the input data set
 * \param[in]  n  Number of rows in the input data set
 */
template <typename algorithmFPType>
DAAL_EXPORT Status OnlinePartialResult::addPartialResultStorage(size_t m, size_t n)
{
    DataCollectionPtr qCollection = get(outputOfStep1ForStep3);
    DataCollectionPtr rCollection = get(outputOfStep1ForStep2);

    Status s;
    if(qCollection)
    {
        qCollection->push_back(HomogenNumericTable<algorithmFPType>::create(m, n, NumericTable::doAllocate, &s));
    }
    else
    {
        return Status(Error::create(ErrorNullOutputDataCollection, ArgumentName, outputOfStep1ForStep3Str()));
    }
    if(rCollection)
    {
        rCollection->push_back(HomogenNumericTable<algorithmFPType>::create(m, m, NumericTable::doAllocate, &s));
    }
    else
    {
        return Status(Error::create(ErrorNullOutputDataCollection, ArgumentName, outputOfStep1ForStep2Str()));
    }
    return s;
}

}// namespace interface1
}// namespace qr
}// namespace algorithms
}// namespace daal

#endif
