/* file: qr_dense_default_batch_fpt.cpp */
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
//  Implementation of qr algorithm and types methods.
//--
*/

#include "qr_dense_default_batch.h"
namespace daal
{
namespace algorithms
{
namespace qr
{
namespace interface1
{

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT services::Status Result::allocateImpl<DAAL_FPTYPE>(size_t m, size_t n);

}// namespace interface1
}// namespace qr
}// namespace algorithms
}// namespace daal
