/* file: minmax_parameter_fpt.cpp */
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
//  Implementation of minmax algorithm and types methods.
//--
*/

#include "minmax_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace minmax
{
namespace interface1
{

typedef SharedPtr<low_order_moments::BatchImpl> LowOrderMomentsPtr;

/** Constructs min-max normalization parameters with default low order algorithm */
template<typename algorithmFPType>
DAAL_EXPORT Parameter<algorithmFPType>::Parameter(double lowerBound, double upperBound) :
    ParameterBase(lowerBound, upperBound, LowOrderMomentsPtr(new low_order_moments::Batch<algorithmFPType>())) { }

/** Constructs min-max normalization parameters */
template<typename algorithmFPType>
DAAL_EXPORT Parameter<algorithmFPType>::Parameter(
    double lowerBound, double upperBound, const LowOrderMomentsPtr &moments) :
    ParameterBase(lowerBound, upperBound, moments) { }

template DAAL_EXPORT Parameter<DAAL_FPTYPE>::Parameter(double lowerBound, double upperBound);

template DAAL_EXPORT Parameter<DAAL_FPTYPE>::Parameter(double lowerBound, double upperBound,
                                                       const LowOrderMomentsPtr &moments);

}// namespace interface1
}// namespace minmax
}// namespace normalization
}// namespace algorithms
}// namespace daal
