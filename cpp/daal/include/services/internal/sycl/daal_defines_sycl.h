/* file: daal_defines_sycl.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  Common definitions.
//--
*/

#ifndef __DAAL_DEFINES_SYCL_H__
#define __DAAL_DEFINES_SYCL_H__

/** \file daal_defines_sycl.h */

#include "services/daal_defines.h"
#include "services/internal/sycl/types.h"

#define DAAL_ASSERT_UNIVERSAL_BUFFER(buffer, bufferType, bufferSize)             \
    {                                                                            \
        DAAL_ASSERT((buffer).type() == TypeIds::id<bufferType>());               \
        DAAL_ASSERT((buffer).template get<bufferType>().size() == (bufferSize)); \
    }

#endif
