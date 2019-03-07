/* file: pca_distributedparameter_correlation_fpt.cpp */
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

#include "pca_distributedparameter_correlation.h"

namespace daal
{
namespace algorithms
{
namespace pca
{

template DAAL_EXPORT DistributedParameter<step2Master, DAAL_FPTYPE, correlationDense>::DistributedParameter(const services::SharedPtr<covariance::DistributedIface<step2Master> > &covariance);
template DAAL_EXPORT services::Status DistributedParameter<step2Master, DAAL_FPTYPE, correlationDense>::check() const;

}// namespace pca
}// namespace algorithms
}// namespace daal
