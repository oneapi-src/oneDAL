/* file: data_source.cpp */
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

#include "JDataSource.h"

#include "numeric_table.h"
#include "data_source.h"

using namespace daal;
using namespace daal::data_management;

#define NO_PARAMS_FUNCTION_MAP_0ARG(jType,jFunc,cFunc)                                         \
    JNIEXPORT jType JNICALL Java_com_intel_daal_data_1management_data_1source_DataSource_##jFunc \
    (JNIEnv *env, jobject obj, jlong ptr)                                                      \
    {                                                                                          \
        return(jType)((DataSource *)ptr)->cFunc();                                             \
    }                                                                                          \

#define NO_PARAMS_FUNCTION_MAP_1ARG(jType,jFunc,cFunc,arg1,jArg1type,cArg1type)                \
    JNIEXPORT jType JNICALL Java_com_intel_daal_data_1management_data_1source_DataSource_##jFunc \
    (JNIEnv *env, jobject obj, jlong ptr, jArg1type arg1)                                      \
    {                                                                                          \
        return(jType)((DataSource *)ptr)->cFunc((cArg1type)arg1);                              \
    }                                                                                          \

#define NO_PARAMS_FUNCTION_MAP_2ARG(jType,jFunc,cFunc,arg1,jArg1type,cArg1type,arg2,jArg2type,cArg2type)   \
    JNIEXPORT jType JNICALL Java_com_intel_daal_data_1management_data_1source_DataSource_##jFunc             \
    (JNIEnv *env, jobject obj, jlong ptr, jArg1type arg1, jArg2type arg2)                                  \
    {                                                                                                      \
        return(jType)((DataSource *)ptr)->cFunc((cArg1type)arg1, (cArg2type)arg2);                         \
    }                                                                                                      \

NO_PARAMS_FUNCTION_MAP_0ARG(void, cCreateDictionaryFromContext, createDictionaryFromContext);
NO_PARAMS_FUNCTION_MAP_0ARG(jlong, cGetNumberOfColumns,          getNumberOfColumns         );
NO_PARAMS_FUNCTION_MAP_0ARG(jlong, cGetNumberOfAvailableRows,    getNumberOfAvailableRows   );
NO_PARAMS_FUNCTION_MAP_0ARG(void, cAllocateNumericTable,        allocateNumericTable       );
//NO_PARAMS_FUNCTION_MAP_0ARG(jlong,cGetNumericTable,             getNumericTable            );
NO_PARAMS_FUNCTION_MAP_0ARG(void, cFreeNumericTable,            freeNumericTable           );

NO_PARAMS_FUNCTION_MAP_0ARG(jlong, cLoadDataBlock0Inputs,      loadDataBlock);
NO_PARAMS_FUNCTION_MAP_1ARG(jlong, cLoadDataBlock,      loadDataBlock,    maxRows, jlong, size_t);

/*
 * Class:     com_intel_daal_data_1management_data_1source_DataSource
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_1source_DataSource_cDispose
(JNIEnv *env, jobject obj, jlong ptr)
{
    delete(DataSource *)ptr;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_DataSource_cGetNumericTable(JNIEnv *env, jobject obj, jlong ptr)
{
    NumericTablePtr *spnt = new NumericTablePtr();
    *spnt = ((DataSource *)ptr)->getNumericTable();

    return (jlong)((SerializationIfacePtr *)spnt);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_DataSource_cLoadDataBlock3Inputs
(JNIEnv *env, jobject obj, jlong ptr, jlong maxRows, jlong offset, jlong fullRows)
{
    return(jlong)((DataSource *)ptr)->loadDataBlock((size_t)maxRows, (size_t)offset, (size_t)fullRows);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_DataSource_cLoadDataBlockNt
(JNIEnv *env, jobject obj, jlong ptr, jlong maxRows, jlong ntAddr)
{
    NumericTable *tbl = ((NumericTablePtr *)ntAddr)->get();
    return(jlong)((DataSource *)ptr)->loadDataBlock((size_t)maxRows, tbl);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_1source_DataSource_cLoadDataBlockNt1Input
(JNIEnv *env, jobject obj, jlong ptr, jlong ntAddr)
{
    NumericTable *tbl = ((NumericTablePtr *)ntAddr)->get();
    return(jlong)((DataSource *)ptr)->loadDataBlock(tbl);
}
