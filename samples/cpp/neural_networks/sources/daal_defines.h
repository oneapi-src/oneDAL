/* file: daal_defines.h */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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
