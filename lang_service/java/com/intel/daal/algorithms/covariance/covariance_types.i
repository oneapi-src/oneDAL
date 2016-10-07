/* file: covariance_types.i */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include "daal.h"

using namespace daal::algorithms;

#include "covariance/JMethod.h"

/* Covariance computation methods */
#define DefaultDense    com_intel_daal_algorithms_covariance_Method_DefaultDense
#define SinglePassDense com_intel_daal_algorithms_covariance_Method_SinglePassDense
#define SumDense        com_intel_daal_algorithms_covariance_Method_SumDense
#define FastCSR         com_intel_daal_algorithms_covariance_Method_FastCSR
#define SinglePassCSR   com_intel_daal_algorithms_covariance_Method_SinglePassCSR
#define SumCSR          com_intel_daal_algorithms_covariance_Method_SumCSR
