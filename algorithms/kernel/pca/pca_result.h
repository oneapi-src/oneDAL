/* file: pca_result.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

namespace daal
{
namespace algorithms
{
namespace pca
{

/**
 * Allocates memory for storing partial results of the PCA algorithm
 * \param[in] input Pointer to an object containing input data
 * \param[in] parameter Algorithm parameter
 * \param[in] method Computation method
 */
template<typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::Input *input, daal::algorithms::Parameter *parameter, const Method method)
{
    const InputIface *in = static_cast<const InputIface *>(input);
    size_t nFeatures = in->getNFeatures();

    set(eigenvalues,
        data_management::NumericTablePtr(new data_management::HomogenNumericTable<algorithmFPType>
                                                           (nFeatures, 1, data_management::NumericTableIface::doAllocate, 0)));
    set(eigenvectors,
        data_management::NumericTablePtr(new data_management::HomogenNumericTable<algorithmFPType>
                                                           (nFeatures, nFeatures, data_management::NumericTableIface::doAllocate, 0)));
}

/**
 * Allocates memory for storing partial results of the PCA algorithm     * \param[in] partialResult Pointer to an object containing input data
 * \param[in] parameter Parameter of the algorithm
 * \param[in] method        Computation method
 */
template<typename algorithmFPType>
DAAL_EXPORT void Result::allocate(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter, const Method method)
{
    const PartialResultBase *partialRes = static_cast<const PartialResultBase *>(partialResult);
    size_t nFeatures = partialRes->getNFeatures();

    set(eigenvalues,
        data_management::NumericTablePtr(new data_management::HomogenNumericTable<algorithmFPType>
                                                           (nFeatures, 1,
                                                            data_management::NumericTableIface::doAllocate, 0)));
    set(eigenvectors,
        data_management::NumericTablePtr(new data_management::HomogenNumericTable<algorithmFPType>
                                                           (nFeatures,
                                                            nFeatures, data_management::NumericTableIface::doAllocate, 0)));
}

} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
