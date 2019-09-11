/* file: daal_defines.h */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation.
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
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

/*
!  Content:
!    Auxiliary functions used in C++ neural networks samples
!******************************************************************************/

#ifndef _DAAL_DEFINES_H
#define _DAAL_DEFINES_H

#include <daal.h>

using namespace daal;
using namespace daal::services;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;

typedef initializers::uniform::Batch<> UniformInitializer;
typedef SharedPtr<UniformInitializer> UniformInitializerPtr;
typedef initializers::xavier::Batch<> XavierInitializer;
typedef SharedPtr<XavierInitializer> XavierInitializerPtr;
typedef initializers::gaussian::Batch<> GaussianInitializer;
typedef SharedPtr<GaussianInitializer> GaussianInitializerPtr;

#endif
