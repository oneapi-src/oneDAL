/** file daal_strings.cpp */
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
//  String variables
//--
*/

#include "daal_strings.h"

namespace daal
{

const char* s_stringConsts[] = {
#define DECLARE_DAAL_STRING_CONST(arg1) #arg1,
    DAAL_STRINGS_LIST()
    ""//last
};
#undef DECLARE_DAAL_STRING_CONST

#define DECLARE_DAAL_STRING_CONST(arg1) const char * arg1##Str() { return s_stringConsts[arg1##EStr]; }
DAAL_STRINGS_LIST()
#undef DECLARE_DAAL_STRING_CONST

const char *getStr(EStringConst eStr)
{
    return s_stringConsts[eStr];
}

}
