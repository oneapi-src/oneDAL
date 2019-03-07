/* file: pca_partialresult_correlation.h */
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

#ifndef __PCA_PARTIALRESULT_CORRELATION_
#define __PCA_PARTIALRESULT_CORRELATION_

#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{

/**
 * Allocates memory to store partial results of the PCA  SVD algorithm
 * \param[in] input     Pointer to an object containing input data
 * \param[in] parameter Pointer to the structure of algorithm parameters
 * \param[in] method    Computation method
 */
template<typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult<correlationDense>::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    set(nObservationsCorrelation,
        data_management::HomogenNumericTable<algorithmFPType>::create(1, 1, data_management::NumericTableIface::doAllocate, 0, &s));
    set(sumCorrelation,
        data_management::HomogenNumericTable<algorithmFPType>::create((static_cast<const InputIface *>(input))->getNFeatures(), 1,
                                                             data_management::NumericTableIface::doAllocate, 0, &s));
    set(crossProductCorrelation,
        data_management::HomogenNumericTable<algorithmFPType>::create((static_cast<const InputIface *>(input))->getNFeatures(),
                                                             (static_cast<const InputIface *>(input))->getNFeatures(),
                                                             data_management::NumericTableIface::doAllocate, 0, &s));
    return s;
};

template<typename algorithmFPType>
DAAL_EXPORT services::Status PartialResult<correlationDense>::initialize(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    DAAL_CHECK_STATUS(s, get(nObservationsCorrelation)->assign((algorithmFPType)0.0))
    DAAL_CHECK_STATUS(s, get(sumCorrelation)->assign((algorithmFPType)0.0))
    DAAL_CHECK_STATUS(s, get(crossProductCorrelation)->assign((algorithmFPType)0.0))
    return s;
};

} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
