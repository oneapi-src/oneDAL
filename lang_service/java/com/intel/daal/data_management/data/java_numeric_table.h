/* file: java_numeric_table.h */
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
//  Implementation of the class that connects Java and C++ Numeric Tables
//--
*/

#ifndef __JAVA_NUMERIC_TABLE_H__
#define __JAVA_NUMERIC_TABLE_H__

#include <jni.h>
#include <tbb/tbb.h>

#include "services/daal_defines.h"
#include "data_management/data/data_serialize.h"
#include "numeric_table.h"
#include "java_threading_helper.h"

using namespace daal::data_management;

namespace daal
{
class JavaNumericTableBase
{
public:
    virtual ~JavaNumericTableBase() {}

    static void setJavaVM(JavaVM * jvm);

    static JavaVM * getJavaVM();

    static void setDaalContext(jobject context);

    static jobject getDaalContext();

    virtual jobject getJavaObject() const = 0;

private:
    static JavaVM * globalJavaVM;
    static tbb::enumerable_thread_specific<jobject> globalDaalContext;
};

/**
 *  \brief Class that implements C++ to Java "connector".
 *  Getters and Setters of this class are callbacks
 *  to the corresponding methods of user-defined Java class.
 */
template <int Tag>
class DAAL_EXPORT JavaNumericTable : public NumericTable, public JavaNumericTableBase
{
public:
    DECLARE_SERIALIZABLE_TAG();

    explicit JavaNumericTable(DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual)
        : NumericTable(0, 0, featuresEqual), jvm(NULL), jJavaNumTable(NULL)
    {}

    /**
     *  Constructor
     *
     *  \param featnum[in]       Number of features
     *  \param obsnum[in]        Number of observations
     *  \param _jvm[in]          Java VM interface function table
     *  \param _JavaNumTable[in] Java object associated with this C++ object
     */
    JavaNumericTable(size_t featnum, size_t obsnum, JavaVM * _jvm, jobject _JavaNumTable, StorageLayout layout = layout_unknown,
                     DictionaryIface::FeaturesEqual featuresEqual = DictionaryIface::notEqual)
        : NumericTable(featnum, obsnum, featuresEqual), jvm(_jvm)
    {
        _layout       = layout;
        _memStatus    = userAllocated;
        jJavaNumTable = NULL;

        daal::internal::_java_tls local_tls = tls.local();

        /* mark current thread as 'main' in order not to detach it further */
        local_tls.is_main_thread = true;

        services::Status status = daal::internal::attachCurrentThread(jvm, local_tls);
        if (!status)
        {
            this->_status |= status;
        }
        else
        {
            jJavaNumTable = (local_tls.jenv)->NewGlobalRef(_JavaNumTable);
            if (jJavaNumTable == NULL)
            {
                this->_status.add(services::ErrorCouldntCreateGlobalReferenceToJavaObject);
            }
        }

        tls.local() = local_tls;
    }

    /**
     *  Destructor
     */
    virtual ~JavaNumericTable()
    {
        if (jvm != NULL)
        {
            daal::internal::_java_tls local_tls = tls.local();
            this->_status |= daal::internal::attachCurrentThread(jvm, local_tls);
            if (this->_status && local_tls.is_attached)
            {
                if (jJavaNumTable != NULL)
                {
                    (local_tls.jenv)->DeleteGlobalRef(jJavaNumTable);
                }

                this->_status |= daal::internal::detachCurrentThread(jvm, local_tls, true);
            }
        }
    }

    services::Status serializeImpl(InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block, "getDoubleBlock", "(JJLjava/nio/ByteBuffer;)Ljava/nio/DoubleBuffer;");
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block, "getFloatBlock", "(JJLjava/nio/ByteBuffer;)Ljava/nio/FloatBuffer;");
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block, "getIntBlock", "(JJLjava/nio/ByteBuffer;)Ljava/nio/IntBuffer;");
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<double>(block, "releaseDoubleBlock");
    }
    services::Status releaseBlockOfRows(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE { return releaseTBlock<float>(block, "releaseFloatBlock"); }
    services::Status releaseBlockOfRows(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE { return releaseTBlock<int>(block, "releaseIntBlock"); }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<double>(feature_idx, vector_idx, value_num, rwflag, block, "getDoubleFeature",
                                   "(JJJLjava/nio/ByteBuffer;)Ljava/nio/DoubleBuffer;");
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<float>(feature_idx, vector_idx, value_num, rwflag, block, "getFloatFeature",
                                  "(JJJLjava/nio/ByteBuffer;)Ljava/nio/FloatBuffer;");
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                            BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return getTFeature<int>(feature_idx, vector_idx, value_num, rwflag, block, "getIntFeature", "(JJJLjava/nio/ByteBuffer;)Ljava/nio/IntBuffer;");
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double> & block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<double>(block, "releaseDoubleFeature");
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float> & block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<float>(block, "releaseFloatFeature");
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int> & block) DAAL_C11_OVERRIDE
    {
        return releaseTFeature<int>(block, "releaseIntFeature");
    }

    template <typename T>
    services::Status getTBlock(size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T> & block, const char * javaMethodName,
                               const char * javaMethodSignature)
    {
        services::Status status;
        daal::internal::_java_tls local_tls = tls.local();

        block.setDetails(0, idx, rwFlag);

        /* Get JNI interface pointer for current thread */
        DAAL_CHECK_STATUS(status, daal::internal::attachCurrentThread(jvm, local_tls))

        size_t ncols      = _ddict->getNumberOfFeatures();
        size_t bufferSize = nrows * ncols * sizeof(T);

        if (!block.resizeBuffer(ncols, nrows))
        {
            return services::Status();
        }

        void * buf = block.getBlockPtr();

        /* Get class associated with Java object */
        local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
        if (local_tls.jcls == NULL)
        {
            return services::Status(services::ErrorCouldntFindClassForJavaObject);
        }

        /* Get ID of the 'getBlockOfRows' method of the Java class */
        jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName, javaMethodSignature);
        if (jmeth == NULL)
        {
            return services::Status(
                services::Error::create(services::ErrorCouldntFindJavaMethod, services::Method, services::String(javaMethodName)));
        }

        /* Call 'getBlockOfRows' Java method */
        local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer(buf, bufferSize);
        local_tls.jbuf = (local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth, (jlong)idx, (jlong)nrows, local_tls.jbuf);

        buf = (local_tls.jenv)->GetDirectBufferAddress(local_tls.jbuf);

        block.setPtr((T *)buf, ncols, nrows);

        DAAL_CHECK_STATUS(status, daal::internal::detachCurrentThread(jvm, local_tls))

        tls.local() = local_tls;

        return status;
    }

    template <typename T>
    services::Status releaseTBlock(BlockDescriptor<T> & block, const char * javaMethodName)
    {
        services::Status status;
        daal::internal::_java_tls local_tls = tls.local();
        if (block.getRWFlag() == writeOnly)
        {
            /* Get JNI interface pointer for current thread */
            DAAL_CHECK_STATUS(status, daal::internal::attachCurrentThread(jvm, local_tls))

            /* Get class associated with Java object */
            local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
            if (local_tls.jcls == NULL)
            {
                return services::Status(services::ErrorCouldntFindClassForJavaObject);
            }

            /* Get ID of the 'releaseBlockOfRows' method of the Java class */
            jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName, "(JJLjava/nio/ByteBuffer;)V");
            if (jmeth == NULL)
            {
                return services::Status(
                    services::Error::create(services::ErrorCouldntFindJavaMethod, services::Method, services::String(javaMethodName)));
            }

            size_t idx        = block.getRowsOffset();
            size_t nrows      = block.getNumberOfRows();
            size_t ncols      = _ddict->getNumberOfFeatures();
            size_t bufferSize = nrows * ncols * sizeof(T);
            local_tls.jbuf    = (local_tls.jenv)->NewDirectByteBuffer(block.getBlockPtr(), bufferSize);

            /* Call 'releaseBlockOfRows' Java method */
            (local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth, (jlong)idx, (jlong)nrows, local_tls.jbuf, block.getRWFlag());

            DAAL_CHECK_STATUS(status, daal::internal::detachCurrentThread(jvm, local_tls))
        }

        tls.local() = local_tls;
        block.reset();
        return status;
    }

    template <typename T>
    services::Status getTFeature(size_t feature_idx, size_t idx, size_t nrows, ReadWriteMode rwFlag, BlockDescriptor<T> & block,
                                 const char * javaMethodName, const char * javaMethodSignature)
    {
        services::Status status;
        daal::internal::_java_tls local_tls = tls.local();

        block.setDetails(feature_idx, idx, rwFlag);

        /* Get JNI interface pointer for current thread */
        DAAL_CHECK_STATUS(status, daal::internal::attachCurrentThread(jvm, local_tls))

        size_t bufferSize = nrows * sizeof(T);

        if (!block.resizeBuffer(1, nrows))
        {
            return services::Status();
        }

        void * buf = block.getBlockPtr();

        /* Get class associated with Java object */
        local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
        if (local_tls.jcls == NULL)
        {
            return services::Status(services::ErrorCouldntFindClassForJavaObject);
        }

        /* Get ID of the 'getBlockOfRows' method of the Java class */
        jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName, javaMethodSignature);
        if (jmeth == NULL)
        {
            return services::Status(
                services::Error::create(services::ErrorCouldntFindJavaMethod, services::Method, services::String(javaMethodName)));
        }

        /* Call 'getBlockOfRows' Java method */
        local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer(buf, bufferSize);

        local_tls.jbuf = (local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth, (jlong)feature_idx, (jlong)idx, (jlong)nrows, local_tls.jbuf);

        buf = (local_tls.jenv)->GetDirectBufferAddress(local_tls.jbuf);

        DAAL_CHECK_STATUS(status, daal::internal::detachCurrentThread(jvm, local_tls))

        tls.local() = local_tls;

        block.setPtr((T *)buf, 1, nrows);
        return status;
    }

    template <typename T>
    services::Status releaseTFeature(BlockDescriptor<T> & block, const char * javaMethodName)
    {
        services::Status status;
        if (block.getRWFlag() == writeOnly)
        {
            daal::internal::_java_tls local_tls = tls.local();
            /* Get JNI interface pointer for current thread */
            DAAL_CHECK_STATUS(status, daal::internal::attachCurrentThread(jvm, local_tls))

            /* Get class associated with Java object */
            local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
            if (local_tls.jcls == NULL)
            {
                return services::Status(services::ErrorCouldntFindClassForJavaObject);
            }

            /* Get ID of the 'releaseBlockOfRows' method of the Java class */
            jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName, "(JJJLjava/nio/ByteBuffer;)V");
            if (jmeth == NULL)
            {
                return services::Status(
                    services::Error::create(services::ErrorCouldntFindJavaMethod, services::Method, services::String(javaMethodName)));
            }

            size_t idx         = block.getRowsOffset();
            size_t nrows       = block.getNumberOfRows();
            size_t feature_idx = block.getColumnsOffset();

            size_t bufferSize = nrows * sizeof(T);
            local_tls.jbuf    = (local_tls.jenv)->NewDirectByteBuffer(block.getBlockPtr(), bufferSize);

            /* Call 'releaseBlockOfRows' Java method */
            (local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth, (jlong)feature_idx, (jlong)idx, (jlong)nrows, local_tls.jbuf, block.getRWFlag());

            DAAL_CHECK_STATUS(status, daal::internal::detachCurrentThread(jvm, local_tls))

            tls.local() = local_tls;
        }
        block.reset();
        return status;
    }

    virtual jobject getJavaObject() const DAAL_C11_OVERRIDE { return jJavaNumTable; }

protected:
    tbb::enumerable_thread_specific<daal::internal::_java_tls> tls; /**< Thread local storage */
    jobject jJavaNumTable;                                          /**< Java object associated with this C++ object */
    JavaVM * jvm;                                                   /**< Java VM interface function table */

    template <typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        services::Status status;
        NumericTable::serialImpl<Archive, onDeserialize>(arch);

        arch->set(_memStatus);

        if (!onDeserialize)
        {
            daal::internal::_java_tls local_tls = tls.local();

            if (jvm == NULL)
            {
                jvm                      = getJavaVM();
                local_tls.is_main_thread = true;
            }

            DAAL_CHECK_STATUS(status, daal::internal::attachCurrentThread(jvm, local_tls))
            if (!status)
            {
                tls.local() = local_tls;
                return status;
            }

            local_tls.jcls  = (local_tls.jenv)->GetObjectClass(jJavaNumTable);
            jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, "packNative", "()V");
            (local_tls.jenv)->CallObjectMethod(jJavaNumTable, jmeth);

            jclass clsByteArrayOutputStream            = (local_tls.jenv)->FindClass("java/io/ByteArrayOutputStream");
            jmethodID byteArrayOutputStreamConstructor = (local_tls.jenv)->GetMethodID(clsByteArrayOutputStream, "<init>", "()V");
            jobject jOutputByteStream                  = (local_tls.jenv)->NewObject(clsByteArrayOutputStream, byteArrayOutputStreamConstructor);

            jclass clsObjectOutputStream            = (local_tls.jenv)->FindClass("java/io/ObjectOutputStream");
            jmethodID objectOutputStreamConstructor = (local_tls.jenv)->GetMethodID(clsObjectOutputStream, "<init>", "(Ljava/io/OutputStream;)V");
            jobject jOutputStream = (local_tls.jenv)->NewObject(clsObjectOutputStream, objectOutputStreamConstructor, jOutputByteStream);

            jmeth = (local_tls.jenv)->GetMethodID(clsObjectOutputStream, "writeObject", "(Ljava/lang/Object;)V");
            (local_tls.jenv)->CallObjectMethod(jOutputStream, jmeth, jJavaNumTable);

            jmeth           = (local_tls.jenv)->GetMethodID(clsByteArrayOutputStream, "toByteArray", "()[B");
            jbyteArray jbuf = static_cast<jbyteArray>((local_tls.jenv)->CallObjectMethod(jOutputByteStream, jmeth));

            jbyte * ptr = (local_tls.jenv)->GetByteArrayElements(jbuf, 0);
            size_t size = (local_tls.jenv)->GetArrayLength(jbuf);

            arch->set(_layout);

            arch->set(size);

            arch->set(ptr, size);

            (local_tls.jenv)->ReleaseByteArrayElements(jbuf, ptr, 0);

            DAAL_CHECK_STATUS(status, daal::internal::detachCurrentThread(jvm, local_tls))

            tls.local() = local_tls;
        }
        else
        {
            daal::internal::_java_tls local_tls = tls.local();

            if (jvm == NULL)
            {
                jvm                      = getJavaVM();
                local_tls.is_main_thread = true;
            }

            DAAL_CHECK_STATUS(status, daal::internal::attachCurrentThread(jvm, local_tls))
            if (!status)
            {
                tls.local() = local_tls;
                return status;
            }

            jbyte * ptr;
            size_t size;

            arch->set(_layout);

            arch->set(size);

            ptr = (jbyte *)daal::services::daal_malloc(size * sizeof(jbyte));
            arch->set(ptr, size);
            jbyteArray jbuf = (local_tls.jenv)->NewByteArray(size);
            (local_tls.jenv)->SetByteArrayRegion(jbuf, 0, size, ptr);

            jclass clsByteArrayInputStream            = (local_tls.jenv)->FindClass("java/io/ByteArrayInputStream");
            jmethodID byteArrayInputStreamConstructor = (local_tls.jenv)->GetMethodID(clsByteArrayInputStream, "<init>", "([B)V");
            jobject jInputByteStream                  = (local_tls.jenv)->NewObject(clsByteArrayInputStream, byteArrayInputStreamConstructor, jbuf);

            jclass clsObjectInputStream            = (local_tls.jenv)->FindClass("java/io/ObjectInputStream");
            jmethodID objectInputStreamConstructor = (local_tls.jenv)->GetMethodID(clsObjectInputStream, "<init>", "(Ljava/io/InputStream;)V");
            jobject jInputStream = (local_tls.jenv)->NewObject(clsObjectInputStream, objectInputStreamConstructor, jInputByteStream);

            jmethodID jmeth      = (local_tls.jenv)->GetMethodID(clsObjectInputStream, "readObject", "()Ljava/lang/Object;");
            jobject javaNumTable = (local_tls.jenv)->CallObjectMethod(jInputStream, jmeth);

            jclass jNumericTable = (local_tls.jenv)->GetObjectClass(javaNumTable);

            jmethodID numericTableUnpack = (local_tls.jenv)->GetMethodID(jNumericTable, "unpackNative", "(Lcom/intel/daal/services/DaalContext;J)V");

            SerializationIfacePtr * thisPtr = new SerializationIfacePtr(this, services::EmptyDeleter());

            jobject context = getDaalContext();
            (local_tls.jenv)->CallObjectMethod(javaNumTable, numericTableUnpack, context, (jlong)thisPtr);
            setDaalContext(context);

            jJavaNumTable = (local_tls.jenv)->NewGlobalRef(javaNumTable);
            if (jJavaNumTable == NULL)
            {
                this->_status.add(services::ErrorCouldntCreateGlobalReferenceToJavaObject);
            }
            daal::services::daal_free(ptr);

            if (_memStatus != notAllocated)
            {
                _memStatus = internallyAllocated;
            }

            DAAL_CHECK_STATUS(status, daal::internal::detachCurrentThread(jvm, local_tls))

            tls.local() = local_tls;
        }

        return status;
    }

    virtual services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE { return services::Status(); }

    virtual void freeDataMemoryImpl() DAAL_C11_OVERRIDE {}
};

} // namespace daal

#endif
