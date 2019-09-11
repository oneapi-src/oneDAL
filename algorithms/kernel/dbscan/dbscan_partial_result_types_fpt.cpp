/* file: dbscan_partial_result_types_fpt.cpp */
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
//  Implementation of DBSCAN algorithm and types methods.
//--
*/

#include "dbscan_partial_result.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace dbscan
{
namespace interface1
{

template DAAL_EXPORT Status DistributedPartialResultStep1::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep2::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep3::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep4::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep5::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep6::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep7::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep8::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedResultStep9::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep9::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep10::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep11::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep12::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedResultStep13::allocate<DAAL_FPTYPE>(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep13::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface1
}// namespace dbscan
}// namespace algorithms
}// namespace daal
