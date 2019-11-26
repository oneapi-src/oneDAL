/** file daal_strings.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  String variables
//--
*/

#include "daal_strings.h"

namespace daal
{
const char * s_stringConsts[] = {
#define DECLARE_DAAL_STRING_CONST(arg1) #arg1,
    DAAL_STRINGS_LIST() "" //last
};
#undef DECLARE_DAAL_STRING_CONST

#define DECLARE_DAAL_STRING_CONST(arg1) \
    const char * arg1##Str() { return s_stringConsts[arg1##EStr]; }
DAAL_STRINGS_LIST()
#undef DECLARE_DAAL_STRING_CONST

const char * getStr(EStringConst eStr)
{
    return s_stringConsts[eStr];
}

} // namespace daal
