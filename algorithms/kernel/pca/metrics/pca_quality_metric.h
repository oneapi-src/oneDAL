/* file: pca_quality_metric.h */
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
//  Implementation of the class defining the pca explained variance quality metric
//--
*/

#ifndef __PCA_QUALITY_METRIC_
#define __PCA_QUALITY_METRIC_

#include "algorithms/pca/pca_explained_variance_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace quality_metric
{
namespace explained_variance
{

/**
* Allocates memory to store
* \param[in] input   %Input object
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Algorithm method
*/
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method)
{
    size_t nComponents = (static_cast<const Parameter *>(par))->nComponents;
    if (nComponents == 0)
    {
        const Input *in = static_cast<const Input *>(input);
        nComponents = in->get(eigenvalues)->getNumberOfColumns();
    }
    services::Status status;
    set(explainedVariances,
        data_management::HomogenNumericTable<algorithmFPType>::create(nComponents, 1, data_management::NumericTableIface::doAllocate, 0, &status));
    DAAL_CHECK_STATUS_VAR(status);
    set(explainedVariancesRatios,
        data_management::HomogenNumericTable<algorithmFPType>::create(nComponents, 1, data_management::NumericTableIface::doAllocate, 0, &status));
    DAAL_CHECK_STATUS_VAR(status);
    set(noiseVariance,
        data_management::HomogenNumericTable<algorithmFPType>::create(1, 1, data_management::NumericTableIface::doAllocate, 0, &status));
    DAAL_CHECK_STATUS_VAR(status);

    return status;
}

} // namespace explained_variance
} // namespace quality_metric
} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
