/* file: pca_batchparameter_svd_fpt.cpp */
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
//  Implementation of PCA algorithm interface.
//--
*/

#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface2
{

/** Constructs PCA parameters */
template<typename algorithmFPType>
DAAL_EXPORT BatchParameter<algorithmFPType, svdDense>::BatchParameter(
    const services::SharedPtr<normalization::zscore::BatchImpl> &normalization) :
    normalization(normalization) {};

template<typename algorithmFPType>
DAAL_EXPORT services::Status BatchParameter<algorithmFPType, svdDense>::check() const
{
    DAAL_CHECK(normalization, services::ErrorNullAuxiliaryAlgorithm);
    return services::Status();
}

template DAAL_EXPORT BatchParameter<DAAL_FPTYPE, svdDense>::BatchParameter
        (const services::SharedPtr<normalization::zscore::BatchImpl> &normalization);

template DAAL_EXPORT services::Status BatchParameter<DAAL_FPTYPE, svdDense>::check() const;

}
} // namespace pca
} // namespace algorithms
} // namespace daal
