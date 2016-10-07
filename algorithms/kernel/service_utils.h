/* file: service_utils.h */
/*******************************************************************************
* Copyright 2015-2016 Intel Corporation
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
//  Declaration of service utilities
//--
*/
#ifndef __SERVICE_UTILS_H__
#define __SERVICE_UTILS_H__

#include "env_detect.h"

namespace daal
{

template<class T, CpuType cpu>
inline void swap(T& x, T& y)
{
    T tmp = x;
    x = y;
    y = tmp;
}

} // namespace daal

#endif
