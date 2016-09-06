/* file: java_csr_numeric_table.h */
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

/*
//++
//  Implementation of the class that connects Java and C++ CSR Numeric Table
//--
*/

#ifndef __JAVA_CSR_NUMERIC_TABLE_H__
#define __JAVA_CSR_NUMERIC_TABLE_H__

#include <jni.h>
#include <tbb/tbb.h>

#include "java_numeric_table.h"
#include "csr_numeric_table.h"

using namespace daal::data_management;

namespace daal
{

/**
 *  \brief Class that implements C++ to Java "connector" for CSR Numeric Table.
 *  Getters and Setters of this class are callbacks
 *  to the corresponding methods of user-defined Java class.
 */
class JavaCSRNumericTable : public JavaNumericTable, virtual public CSRNumericTableIface
{
public:
    JavaCSRNumericTable(): JavaNumericTable(true) {}
    /**
     *  Constructor
     *
     *  \param featnum[in]       Number of features
     *  \param obsnum[in]        Number of observations
     *  \param _jvm[in]          Java VM interface function table
     *  \param _JavaNumTable[in] Java object associated with this C++ object
     */
    JavaCSRNumericTable(size_t featnum, size_t obsnum, JavaVM *_jvm, jobject _JavaNumTable) :
        JavaNumericTable(featnum, obsnum, _jvm, _JavaNumTable, StorageLayout::csrArray, true) {}

    virtual ~JavaCSRNumericTable() {}

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_JAVANIOCSR_NT_ID;
    }

    void getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        getSparseTBlock<double>(vector_idx, vector_num, rwflag, block, "getDoubleSparseBlock");
    }
    void getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        getSparseTBlock<float>(vector_idx, vector_num, rwflag, block, "getFloatSparseBlock");
    }
    void getSparseBlock(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, CSRBlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        getSparseTBlock<int>(vector_idx, vector_num, rwflag, block, "getIntSparseBlock");
    }

    void releaseSparseBlock(CSRBlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseSparseTBlock<double>(block);
    }
    void releaseSparseBlock(CSRBlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseSparseTBlock<float>(block);
    }
    void releaseSparseBlock(CSRBlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseSparseTBlock<int>(block);
    }

    void releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<double>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<float>(block);
    }
    void releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTBlock<int>(block);
    }

    void releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<double>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<float>(block);
    }
    void releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        releaseTFeature<int>(block);
    }

    size_t getDataSize() DAAL_C11_OVERRIDE
    {
        jint status = JNI_OK;
        _tls local_tls = tls.local();

        /* Get JNI interface pointer for current thread */
        status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
        if(status != JNI_OK)
        {
            return 0;
        }
        else
        {
            local_tls.is_attached = true;
        }

        /* Get class associated with Java object */
        local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
        if(local_tls.jcls == NULL)
        {
            return 0;
        }

        /* Get ID of the 'getBlockOfRows' method of the Java class */
        jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, "getSparseBlockSize", "(JJ)J");
        if(jmeth == NULL)
        {
            return 0;
        }

        size_t bufferSize = (size_t)((local_tls.jenv)->CallObjectMethod(
                                         jJavaNumTable, jmeth, (jlong)0, (jlong)getNumberOfRows()));

        if(!local_tls.is_main_thread)
        {
            status = jvm->DetachCurrentThread();
            if(status != JNI_OK)
            {
                return 0;
            }
        }
        return bufferSize;
    }

protected:

    template <typename T>
    void getSparseTBlock( size_t idx, size_t nrows, int rwFlag, CSRBlockDescriptor<T>& block, const char *javaMethodName )
    {
        jint status = JNI_OK;
        _tls local_tls = tls.local();

        /* Get JNI interface pointer for current thread */
        status = jvm->AttachCurrentThread((void **)(&(local_tls.jenv)), NULL);
        if(status != JNI_OK) { return; }

        local_tls.is_attached = true;

        /* Get class associated with Java object */
        local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
        if(local_tls.jcls == NULL) { return; }

        /* Get ID of the 'getBlockOfRows' method of the Java class */
        jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, "getSparseBlockSize", "(JJ)J");
        if(jmeth == NULL) { return; }
        size_t nValues = (size_t)((local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth, (jlong)idx, (jlong)nrows));

        size_t ncols = _ddict->getNumberOfFeatures();

        block.setDetails( ncols, idx, rwFlag );

        size_t valuesSize     = nValues * sizeof(T);
        size_t colindicesSize = nValues * sizeof(size_t);
        size_t rowOffsetsSize = (nrows + 1) * sizeof(size_t);

        size_t colIndicesSizeJava = nValues * sizeof(__int64);
        size_t rowOffsetsSizeJava = (nrows + 1) * sizeof(__int64);

        T       *valuesBuf     = (T      *)daal::services::daal_malloc(valuesSize);
        size_t  *colIndicesBuf = (size_t *)daal::services::daal_malloc(colindicesSize);
        size_t  *rowOffsetsBuf = (size_t *)daal::services::daal_malloc(rowOffsetsSize);
        __int64 *colIndicesBufJava = (__int64 *)colIndicesBuf;
        __int64 *rowOffsetsBufJava = (__int64 *)rowOffsetsBuf;

        if(sizeof(size_t) != sizeof(__int64))
        {
            rowOffsetsBufJava = (__int64 *)daal::services::daal_malloc(rowOffsetsSizeJava);
            colIndicesBufJava = (__int64 *)daal::services::daal_malloc(colIndicesSizeJava);
        }

        jobject jdata       = (local_tls.jenv)->NewDirectByteBuffer(valuesBuf, valuesSize);
        jobject jColIndices = (local_tls.jenv)->NewDirectByteBuffer(colIndicesBufJava, colIndicesSizeJava);
        jobject jRowOffsets = (local_tls.jenv)->NewDirectByteBuffer(rowOffsetsBufJava, rowOffsetsSizeJava);

        jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls,
                                              javaMethodName, "(JJLjava/nio/ByteBuffer;Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;)J");
        if(jmeth == NULL) { return; }
        size_t nRows = (size_t)((local_tls.jenv)->CallObjectMethod(
                                    jJavaNumTable, jmeth, (jlong)idx, (jlong)nrows, jdata, jColIndices, jRowOffsets));

        if(nRows == 0) { return; }

        if(sizeof(size_t) != sizeof(__int64))
        {
            for(size_t i = 0; i < nValues; i++)
            {
                colIndicesBuf[i] = (size_t)colIndicesBufJava[i];
            }
            for(size_t i = 0; i < nrows + 1; i++)
            {
                rowOffsetsBuf[i] = (size_t)rowOffsetsBufJava[i];
            }
            daal::services::daal_free(rowOffsetsBufJava);
            daal::services::daal_free(colIndicesBufJava);
        }

        block.setValuesPtr(valuesBuf, nValues);
        block.setColumnIndicesPtr(colIndicesBuf, nValues);
        block.setRowIndicesPtr(rowOffsetsBuf, nrows);

        if(!local_tls.is_main_thread)
        {
            status = jvm->DetachCurrentThread();
            if(status != JNI_OK) { return; }
        }
    }

    template <typename T>
    void releaseSparseTBlock( CSRBlockDescriptor<T>& block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            this->_errors->add(services::ErrorMethodNotSupported);
            return;
        }
        void *valuesBuf     = block.getBlockValuesPtr();
        void *colIndicesBuf = block.getBlockColumnIndicesPtr();
        void *rowOffsetsBuf = block.getBlockRowIndicesPtr();

        if(valuesBuf    ) daal::services::daal_free(valuesBuf);
        if(colIndicesBuf) daal::services::daal_free(colIndicesBuf);
        if(rowOffsetsBuf) daal::services::daal_free(rowOffsetsBuf);

        block.setValuesPtr(0, 0);
        block.setColumnIndicesPtr(0, 0);
        block.setRowIndicesPtr(0, 0);

        block.setDetails( 0, 0, 0 );
    }

    template <typename T>
    void releaseTBlock( BlockDescriptor<T>& block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            this->_errors->add(services::ErrorMethodNotSupported);
            return;
        }
        block.setDetails( 0, 0, 0 );
    }

    template <typename T>
    void releaseTFeature( BlockDescriptor<T>& block )
    {
        if (block.getRWFlag() & (int)writeOnly)
        {
            this->_errors->add(services::ErrorMethodNotSupported);
        }
        block.setDetails( 0, 0, 0 );
    }

}; // class JavaCSRNumericTable

} // namespace daal

#endif
