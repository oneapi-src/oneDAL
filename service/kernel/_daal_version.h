/* file: _daal_version.h */
/*******************************************************************************
* Copyright 2015-2018 Intel Corporation.
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

#ifndef ___DAAL_VERSION_H__
#define ___DAAL_VERSION_H__

#include "_daal_version_data.h"

#if PRODUCT_STATUS == 'A'
    #define PRODUCTSTATUS "Alpha"
    #define PRODUCTSTATUSDLL " Alpha"
    #define SUBBUILD 0
#endif
#if PRODUCT_STATUS == 'B'
    #define PRODUCTSTATUS "Beta"
    #define PRODUCTSTATUSDLL " Beta"
    #define SUBBUILD 0
#endif
#if PRODUCT_STATUS == 'P'
    #define PRODUCTSTATUS "Product"
    #define PRODUCTSTATUSDLL ""
    #define SUBBUILD 1
#endif

/* Intermediate defines */
#define FILE_VERSION1(a,b,c,d) FILE_VERSION0(a,b,c,d)
#define FILE_VERSION0(a,b,c,d) #a "." #b "." #c "." #d

#define PRODUCT_VERSION1(a,b) PRODUCT_VERSION0(a, b)
#define PRODUCT_VERSION0(a,b) #a "." #b PRODUCTSTATUSDLL

/* The next 3 defines need to use in *.rc files */
/* instead of symbolic constants like "10.0.2.0" */

#define FILE_VERSION MAJORVERSION, MINORVERSION, UPDATEVERSION, SUBBUILD
#define FILE_VERSION_STR FILE_VERSION1(MAJORVERSION,MINORVERSION,UPDATEVERSION,SUBBUILD)
#define PRODUCT_VERSION_STR PRODUCT_VERSION1(MAJORVERSION,MINORVERSION)

#define PRODUCT_NAME_STR "Intel(R) Data Analytics Acceleration Library\0"

#endif
