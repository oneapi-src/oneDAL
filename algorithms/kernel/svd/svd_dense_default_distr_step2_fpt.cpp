/* file: svd_dense_default_distr_step2_fpt.cpp */
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
//  Implementation of svd algorithm and types methods.
//--
*/

#include "svd_dense_default_distr_step2.h"
namespace daal
{
namespace algorithms
{
namespace svd
{
namespace interface1
{

template DAAL_EXPORT services::Status DistributedPartialResult::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status DistributedPartialResult::setPartialResultStorage<DAAL_FPTYPE>(data_management::KeyValueDataCollection *inCollection, size_t &nBlocks);

}// namespace interface1
}// namespace svd
}// namespace algorithms
}// namespace daal
