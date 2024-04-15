/* file: pca_partialresult_svd.h */
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
template <typename algorithmFPType>
services::Status PartialResult<svdDense>::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                   const int method)
{
    services::Status s;
    set(nObservationsSVD, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTableIface::doAllocate, 0, &s));
    set(sumSquaresSVD, HomogenNumericTable<algorithmFPType>::create((static_cast<const InputIface *>(input))->getNFeatures(), 1,
                                                                    NumericTableIface::doAllocate, 0, &s));
    set(sumSVD, HomogenNumericTable<algorithmFPType>::create((static_cast<const InputIface *>(input))->getNFeatures(), 1,
                                                             NumericTableIface::doAllocate, 0, &s));
    set(auxiliaryData, DataCollectionPtr(new DataCollection()));
    return s;
};

template <typename algorithmFPType>
services::Status PartialResult<svdDense>::initialize(const daal::algorithms::Input * input, const daal::algorithms::Parameter * parameter,
                                                     const int method)
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
