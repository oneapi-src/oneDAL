/* file: pca_baseparameter_fpt.cpp */
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
#include "algorithms/pca/pca_types.h"

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace interface1
{

/** Constructs PCA parameters */
template<typename algorithmFPType, Method method>
DAAL_EXPORT BaseParameter<algorithmFPType, method>::BaseParameter() {};

template DAAL_EXPORT BaseParameter<DAAL_FPTYPE, correlationDense>::BaseParameter();
template DAAL_EXPORT BaseParameter<DAAL_FPTYPE, svdDense>::BaseParameter();

}// namespace interface1
} // namespace pca
} // namespace algorithms
} // namespace daal
