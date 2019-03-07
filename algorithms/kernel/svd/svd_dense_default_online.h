/* file: svd_dense_default_online.h */
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
//  Implementation of svd algorithm and types methods.
//--
*/
#ifndef __SVD_DENSE_DEFAULT_ONLINE__
#define __SVD_DENSE_DEFAULT_ONLINE__

#include "svd_types.h"
#include "daal_strings.h"

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

using namespace daal::services;

/**
 * Allocates memory to store final results of the SVD algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
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
 * Allocates additional memory to store partial results of the SVD algorithm for each subsequent compute() method
 * \tparam     algorithmFPType    Data type to use for storage in the resulting HomogenNumericTable
 * \param[in]  m    Number of columns in the input data set
 * \param[in]  n    Number of rows in the input data set
 * \param[in]  par  Reference to the object with the algorithm parameters
 */
template <typename algorithmFPType>
DAAL_EXPORT Status OnlinePartialResult::addPartialResultStorage(size_t m, size_t n, Parameter &par)
{
    DataCollectionPtr rCollection =
        staticPointerCast<DataCollection, SerializationIface>(Argument::get(outputOfStep1ForStep2));
    Status st;
    if(rCollection)
    {
        rCollection->push_back(HomogenNumericTable<algorithmFPType>::create(m, m, NumericTable::doAllocate, &st));
    }
    else
    {
        return Status(Error::create(ErrorNullOutputDataCollection, ArgumentName, outputOfStep1ForStep3Str()));
    }

    if(par.leftSingularMatrix != notRequired)
    {
        DataCollectionPtr qCollection =
            staticPointerCast<DataCollection,
            SerializationIface>(Argument::get(outputOfStep1ForStep3));
        if(qCollection)
        {
            qCollection->push_back(HomogenNumericTable<algorithmFPType>::create(m, n, NumericTable::doAllocate, &st));
        }
        else
        {
            return Status(Error::create(ErrorNullOutputDataCollection, ArgumentName, outputOfStep1ForStep3Str()));
        }
    }
    return st;
}

}// namespace interface1
}// namespace svd
}// namespace algorithms
}// namespace daal

#endif
