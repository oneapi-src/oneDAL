/* file: zscore_parameter_fpt.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of zscore algorithm and types methods.
//--
*/

#include "algorithms/normalization/zscore_types.h"

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
namespace interface3
{
/** Constructs z-score normalization parameters */
template <typename algorithmFPType>
Parameter<algorithmFPType, defaultDense>::Parameter(const SharedPtr<low_order_moments::BatchImpl> & moments, const bool doScale)
    : BaseParameter(doScale), moments(moments) {};

/** Constructs z-score normalization parameters */
template <typename algorithmFPType>
Parameter<algorithmFPType, sumDense>::Parameter(const bool doScale) : BaseParameter(doScale) {};

/**
    * Check the correctness of the %Parameter object
    */
template <typename algorithmFPType>
Status Parameter<algorithmFPType, defaultDense>::check() const
{
    DAAL_CHECK(moments.get() != 0, ErrorNullParameterNotSupported);
    return Status();
}

template Parameter<DAAL_FPTYPE, defaultDense>::Parameter(const SharedPtr<low_order_moments::BatchImpl> & moments, const bool doScale);
template Parameter<DAAL_FPTYPE, sumDense>::Parameter(const bool doScale);
template Status Parameter<DAAL_FPTYPE, defaultDense>::check() const;

} // namespace interface3

} // namespace zscore
} // namespace normalization
} // namespace algorithms
} // namespace daal
