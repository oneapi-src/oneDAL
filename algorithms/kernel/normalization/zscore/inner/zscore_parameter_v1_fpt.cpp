/* file: zscore_parameter_v1_fpt.cpp */
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
//  Implementation of zscore algorithm and types methods.
//--
*/

#include "zscore_types_v1.h"

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
namespace interface1
{
/** Constructs z-score normalization parameters */
template<typename algorithmFPType>
DAAL_EXPORT Parameter<algorithmFPType, defaultDense>::Parameter(const SharedPtr<low_order_moments::BatchImpl> &moments) : moments(moments) {};
/**
 * Check the correctness of the %Parameter object
 */
template<typename algorithmFPType>
DAAL_EXPORT Status Parameter<algorithmFPType, defaultDense>::check() const
{
    DAAL_CHECK(moments.get() != 0, ErrorNullParameterNotSupported);
    return Status();
}

template DAAL_EXPORT Parameter<DAAL_FPTYPE, defaultDense>::Parameter(const SharedPtr<low_order_moments::BatchImpl> &moments);
template DAAL_EXPORT Status Parameter<DAAL_FPTYPE, defaultDense>::check() const;

}// namespace interface1

}// namespace zscore
}// namespace normalization
}// namespace algorithms
}// namespace daal
