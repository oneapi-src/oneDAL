/* file: pca_quality_metric.h */
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
services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    size_t nComponents = (static_cast<const Parameter *>(par))->nComponents;
    if (nComponents == 0)
    {
        const Input * in = static_cast<const Input *>(input);
        nComponents      = in->get(eigenvalues)->getNumberOfColumns();
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
