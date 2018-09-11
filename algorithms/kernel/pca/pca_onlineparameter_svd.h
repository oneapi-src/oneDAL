/* file: pca_onlineparameter_svd.h */
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

#ifndef __PCA_ONLINEPARAMETER_
#define __PCA_ONLINEPARAMETER_

#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{

/** Constructs PCA parameters */
template<typename algorithmFPType>
DAAL_EXPORT OnlineParameter<algorithmFPType, svdDense>::OnlineParameter() {};

template<typename algorithmFPType>
DAAL_EXPORT services::Status OnlineParameter<algorithmFPType, svdDense>::check() const
{
    return services::Status();
}

} // namespace pca
} // namespace algorithms
} // namespace daal

#endif
