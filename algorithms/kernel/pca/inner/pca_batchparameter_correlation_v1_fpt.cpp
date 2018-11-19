/* file: pca_batchparameter_correlation_v1_fpt.cpp */
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

#include "pca/inner/pca_types_v1.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{

/** Constructs PCA parameters */
template<typename algorithmFPType>
DAAL_EXPORT BatchParameter<algorithmFPType, correlationDense>::BatchParameter(const services::SharedPtr<covariance::BatchImpl> &covariance) : covariance(covariance) {};

template<typename algorithmFPType>
DAAL_EXPORT services::Status BatchParameter<algorithmFPType, correlationDense>::check() const
{
    DAAL_CHECK(covariance, services::ErrorNullAuxiliaryAlgorithm);
    return services::Status();
}

template DAAL_EXPORT BatchParameter<DAAL_FPTYPE, correlationDense>::BatchParameter(const services::SharedPtr<covariance::BatchImpl> &covariance);
template DAAL_EXPORT services::Status BatchParameter<DAAL_FPTYPE, correlationDense>::check() const;

}// namespace interface1
}// namespace pca
}// namespace algorithms
}// namespace daal
