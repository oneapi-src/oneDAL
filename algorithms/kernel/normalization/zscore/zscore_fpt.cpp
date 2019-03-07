/* file: zscore_fpt.cpp */
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
//  Implementation of zscore algorithm and types methods.
//--
*/

#include "inner/zscore_types_v1.h"
#include "zscore_result.h"
using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{

namespace interface2
{
/**
 * Allocates memory to store final results of the z-score normalization algorithms
 * \param[in] input     Input objects for the z-score normalization algorithm
 * \param[in] parameter Pointer to algorithm parameter
 * \param[in] method    Algorithm computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    auto impl = ResultImpl::cast(getStorage(*this));
    DAAL_CHECK(impl, ErrorNullPtr);
    return impl->allocate<algorithmFPType>(input, parameter);
}

/**
* Allocates memory to store final results of the z-score normalization algorithms for interface1 API compartibility
* \param[in] input     Input objects for the z-score normalization algorithm
* \param[in] method    Algorithm computation method
*/
template <typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const int method)
{
    return allocate<algorithmFPType>(input, NULL, method);
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const int method);
template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace interface2
}// namespace zscore
}// namespace normalization
}// namespace algorithms
}// namespace daal
