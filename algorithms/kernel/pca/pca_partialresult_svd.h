/* file: pca_partialresult_svd.h */
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
//  Implementation of PCA algorithm interface.
//--
*/

#ifndef __PCA_PARTIALRESULT_SVD_
#define __PCA_PARTIALRESULT_SVD_

#include "algorithms/pca/pca_types.h"

using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace pca
{

/**
 * Allocates memory for storing partial results of the PCA SVD algorithm
 * \param[in] input     Pointer to an object containing input data
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
template<typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult<svdDense>::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    set(nObservationsSVD, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTableIface::doAllocate, 0, &s));
    set(sumSquaresSVD, HomogenNumericTable<algorithmFPType>::create((static_cast<const InputIface *>(input))->getNFeatures(), 1, NumericTableIface::doAllocate, 0, &s));
    set(sumSVD, HomogenNumericTable<algorithmFPType>::create((static_cast<const InputIface *>(input))->getNFeatures(), 1, NumericTableIface::doAllocate, 0, &s));
    set(auxiliaryData, DataCollectionPtr(new DataCollection()));
    return s;
};

template<typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult<svdDense>::initialize(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, get(nObservationsSVD)->assign((algorithmFPType)0.0))
    DAAL_CHECK_STATUS(s, get(sumSquaresSVD)->assign((algorithmFPType)0.0))
    DAAL_CHECK_STATUS(s, get(sumSVD)->assign((algorithmFPType)0.0))
    get(auxiliaryData)->clear();
    return s;
};

} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
