/* file: svd_dense_default_distr_step3.h */
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
#ifndef __SVD_DENSE_DEFAULT_DISTR_STEP3__
#define __SVD_DENSE_DEFAULT_DISTR_STEP3__

#include "svd_types.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
{

/**
 * Allocates memory to store partial results of the SVD algorithm
 * \param[in] input     Pointer to the input object
 * \param[in] parameter Pointer to the parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status DistributedPartialResultStep3::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    Argument::set(finalResultFromStep3, ResultPtr(new Result()));
    return Status();
}

/**
 * Allocates memory to store partial results of the SVD algorithm obtained in the third step in the distributed processing mode
 * \tparam     algorithmFPType            Data type to use for storage in the resulting HomogenNumericTable
 * \param[in]  qCollection  DataCollection of all partial results from step 1 of the SVD algorithm in the distributed processing mode
 */
template <typename algorithmFPType>
DAAL_EXPORT Status DistributedPartialResultStep3::setPartialResultStorage(data_management::DataCollection *qCollection)
{
    size_t qSize = qCollection->size();
    size_t m = 0;
    size_t n = 0;
    for(size_t i = 0 ; i < qSize ; i++)
    {
        data_management::NumericTable  *qNT = static_cast<data_management::NumericTable *>((*qCollection)[i].get());
        m  = qNT->getNumberOfColumns();
        n += qNT->getNumberOfRows();
    }
    ResultPtr result = services::staticPointerCast<Result, data_management::SerializationIface>(Argument::get(finalResultFromStep3));

    return result->allocateImpl<algorithmFPType>(m, n);
}

}// namespace interface1
}// namespace svd
}// namespace algorithms
}// namespace daal

#endif
