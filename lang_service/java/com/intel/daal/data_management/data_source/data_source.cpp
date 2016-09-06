/* file: data_source.cpp */
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
