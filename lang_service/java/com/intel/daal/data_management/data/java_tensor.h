/* file: java_tensor.h */
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
//  Implementation of the class that connects Java and C++ Tensors
//--
*/

#ifndef __JAVA_TENSOR_H__
#define __JAVA_TENSOR_H__

#include <jni.h>
#include <tbb/tbb.h>

#include "daal.h"
#include "services/daal_defines.h"
#include "data_management/data/data_serialize.h"
#include "java_threading_helper.h"

namespace daal
{

using namespace daal::services;
using namespace daal::data_management;

class JavaTensorBase
{
public:
    virtual ~JavaTensorBase() {}

    static void setJavaVM(JavaVM *jvm);

    static JavaVM *getJavaVM();

    static void setDaalContext(jobject context);

    static jobject getDaalContext();

    virtual jobject getJavaObject() const = 0;

private:
    static JavaVM *globalJavaVM;
    static tbb::enumerable_thread_specific<jobject> globalDaalContext;
};

/**
 *  \brief Class that implements C++ to Java "connector".
 *  Getters and Setters of this class are callbacks
 *  to the corresponding methods of user-defined Java class.
 */
template<int Tag>
class DAAL_EXPORT JavaTensor : public Tensor, public JavaTensorBase
{
public:
    DECLARE_SERIALIZABLE_TAG();

    explicit JavaTensor(): Tensor(&_layout), jvm(NULL), jJavaTensor(NULL), _layout(services::Collection<size_t>()) {}

    /**
     *  Constructor
     *  \param[in] dims          Collection with tensor dimension sizes
     *  \param[in] _jvm          Java VM interface function table
     *  \param[in] _JavaTensor Java object associated with this C++ object
     */
    JavaTensor(Collection<size_t> &dims, JavaVM *_jvm, jobject _JavaTensor):
        Tensor(&_layout), jvm(_jvm), _layout(dims), jJavaTensor(NULL)
    {
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
            jJavaTensor = (local_tls.jenv)->NewGlobalRef(_JavaTensor);
            if(jJavaTensor == NULL)
            {
                this->_status.add(services::ErrorCouldntCreateGlobalReferenceToJavaObject);
            }
        }

        tls.local() = local_tls;
    }

    /**
     *  Destructor
     */
    virtual ~JavaTensor()
    {
        if(jvm != NULL)
        {
            daal::internal::_java_tls local_tls = tls.local();
            this->_status |= daal::internal::attachCurrentThread(jvm, local_tls);
            if (this->_status && local_tls.is_attached)
            {
                if (jJavaTensor != NULL)
                {
                    (local_tls.jenv)->DeleteGlobalRef(jJavaTensor);
                }

                this->_status |= daal::internal::detachCurrentThread(jvm, local_tls, true);
            }
        }
    }

    virtual services::Status setDimensions(size_t nDim, const size_t *dimSizes) DAAL_C11_OVERRIDE
    {
        if(!dimSizes)
        {
            return services::Status(services::ErrorNullParameterNotSupported);
        }

        _layout = TensorOffsetLayout(services::Collection<size_t>(nDim, dimSizes));
        return services::Status();
    }

    virtual services::Status setDimensions(const services::Collection<size_t>& dimensions) DAAL_C11_OVERRIDE
    {
        if(!dimensions.size())
        {
            return services::Status(services::ErrorNullParameterNotSupported);
        }

        _layout = TensorOffsetLayout(dimensions);
        return services::Status();
    }

    services::Status serializeImpl(InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        //Tensor::serialImpl<Archive, onDeserialize>(arch);
        return services::Status();
    }

    services::Status getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, ReadWriteMode rwflag,
        SubtensorDescriptor<double>& subtensor, const TensorOffsetLayout& layout ) DAAL_C11_OVERRIDE
    {
        return getTSubtensor<double>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, layout,
        "getDoubleSubtensor", "([JJJLjava/nio/ByteBuffer;)Ljava/nio/DoubleBuffer;");
    }
    services::Status getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, ReadWriteMode rwflag,
        SubtensorDescriptor<float>& subtensor, const TensorOffsetLayout& layout ) DAAL_C11_OVERRIDE
    {
        return getTSubtensor<float>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, layout,
        "getFloatSubtensor", "([JJJLjava/nio/ByteBuffer;)Ljava/nio/FloatBuffer;");
    }
    services::Status getSubtensorEx(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, ReadWriteMode rwflag,
        SubtensorDescriptor<int>& subtensor, const TensorOffsetLayout& layout ) DAAL_C11_OVERRIDE
    {
        return getTSubtensor<int>(fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwflag, subtensor, layout,
        "getIntSubtensor", "([JJJLjava/nio/ByteBuffer;)Ljava/nio/IntBuffer;");
    }


    services::Status releaseSubtensor(SubtensorDescriptor<double> &subtensor) DAAL_C11_OVERRIDE
    {
        return releaseTSubtensor<double>(subtensor, "releaseDoubleSubtensor");
    }
    services::Status releaseSubtensor(SubtensorDescriptor<float> &subtensor) DAAL_C11_OVERRIDE
    {
        return releaseTSubtensor<float>(subtensor, "releaseFloatSubtensor");
    }
    services::Status releaseSubtensor(SubtensorDescriptor<int> &subtensor) DAAL_C11_OVERRIDE
    {
        return releaseTSubtensor<int>(subtensor, "releaseIntSubtensor");
    }

    template<typename T>
    services::Status getTSubtensor(size_t fixedDims, const size_t* fixedDimNums, size_t rangeDimIdx, size_t rangeDimNum, ReadWriteMode rwFlag,
        SubtensorDescriptor<T>& subtensor, const TensorOffsetLayout& layout , const char *javaMethodName, const char *javaMethodSignature)
    {
        services::Status status;
        daal::internal::_java_tls local_tls = tls.local();

        size_t  nDim     = layout.getDimensions().size();
        const size_t* dimSizes = &((layout.getDimensions())[0]);
        size_t blockSize = subtensor.setDetails( nDim, dimSizes, fixedDims, fixedDimNums, rangeDimIdx, rangeDimNum, rwFlag );
        subtensor.saveOffsetLayoutCopy(layout);

        size_t* _dimOffsets = (size_t*)daal::services::daal_malloc(nDim * sizeof(size_t));

        if(!dimSizes) { return services::Status(services::ErrorNullParameterNotSupported); }

        for(size_t i=0; i<nDim; i++)
        {
            _dimOffsets[i] = 1;
        }

        for(size_t i=0; i<nDim; i++)
        {
            for(size_t j=0; j<i; j++)
            {
                _dimOffsets[j] *= dimSizes[i];
            }
        }

        size_t shift = 0;
        for( size_t i=0; i<fixedDims; i++ )
        {
            shift += fixedDimNums[i] * _dimOffsets[i];
        }
        if( fixedDims != nDim )
        {
            shift += rangeDimIdx * _dimOffsets[fixedDims];
        }

        /* Get JNI interface pointer for current thread */
        DAAL_CHECK_STATUS(status, daal::internal::attachCurrentThread(jvm, local_tls))

        size_t bufferSize = blockSize * sizeof(T);

        if( !subtensor.resizeBuffer() ) { return services::Status(); }

        void *buf = subtensor.getPtr();

        /* Get class associated with Java object */
        local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaTensor);
        if(local_tls.jcls == NULL)
        {
            return services::Status(services::ErrorCouldntFindClassForJavaObject);
        }

        /* Get ID of the 'getSubtensor' method of the Java class */
        jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName,
                                                        javaMethodSignature);
        if(jmeth == NULL)
        {
            return services::Status(Error::create(services::ErrorCouldntFindJavaMethod, services::Method, services::String(javaMethodName)));
        }

        jlongArray jFixedDimNums = (local_tls.jenv)->NewLongArray(fixedDims);
        if (jFixedDimNums == NULL) {
            return services::Status(services::UnknownError); /* out of memory error thrown */
        }
        (local_tls.jenv)->SetLongArrayRegion(jFixedDimNums, 0, fixedDims, (jlong*)fixedDimNums);


        /* Call 'getSubtensor' Java method */
        local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer( buf, bufferSize);
        local_tls.jbuf = (local_tls.jenv)->CallObjectMethod(jJavaTensor, jmeth, jFixedDimNums,
                                                            (jlong)rangeDimIdx, (jlong)rangeDimNum, local_tls.jbuf);

        buf = (local_tls.jenv)->GetDirectBufferAddress(local_tls.jbuf);

        subtensor.setPtr( (T *)buf );

        DAAL_CHECK_STATUS(status, daal::internal::detachCurrentThread(jvm, local_tls))

        tls.local() = local_tls;

        daal::services::daal_free( _dimOffsets );
        return status;
    }

    template<typename T>
    services::Status releaseTSubtensor(SubtensorDescriptor<T> &subtensor, const char *javaMethodName)
    {
        services::Status status;
        daal::internal::_java_tls local_tls = tls.local();
        if(subtensor.getRWFlag() == writeOnly)
        {
            /* Get JNI interface pointer for current thread */
            DAAL_CHECK_STATUS(status, daal::internal::detachCurrentThread(jvm, local_tls))

            /* Get class associated with Java object */
            local_tls.jcls = (local_tls.jenv)->GetObjectClass(jJavaTensor);
            if(local_tls.jcls == NULL)
            {
                return services::Status(services::ErrorCouldntFindClassForJavaObject);
            }

            /* Get ID of the 'releaseSubtensor' method of the Java class */
            jmethodID jmeth = (local_tls.jenv)->GetMethodID(local_tls.jcls, javaMethodName, "([JJJLjava/nio/ByteBuffer;)V");
            if(jmeth == NULL)
            {
                return services::Status(Error::create(services::ErrorCouldntFindJavaMethod, services::Method, services::String(javaMethodName)));
            }

            size_t nDim = getNumberOfDimensions();

            size_t blockSize = subtensor.getSize();
            const size_t* dimSizes = &((subtensor.getLayout()->getDimensions())[0]);

            size_t fixedDims     = subtensor.getFixedDims();
            size_t *fixedDimNums = subtensor.getFixedDimNums();
            size_t rangeDimIdx   = subtensor.getRangeDimIdx();
            size_t rangeDimNum   = subtensor.getRangeDimNum();

            size_t* _dimOffsets = (size_t*)daal::services::daal_malloc(nDim * sizeof(size_t));

            for(size_t i=0; i<nDim; i++)
            {
                _dimOffsets[i] = 1;
            }

            for(size_t i=0; i<nDim; i++)
            {
                for(size_t j=0; j<i; j++)
                {
                    _dimOffsets[j] *= dimSizes[i];
                }
            }

            size_t shift = 0;
            for( size_t i=0; i<fixedDims; i++ )
            {
                shift += fixedDimNums[i] * _dimOffsets[i];
            }
            if( fixedDims != nDim )
            {
                shift += rangeDimIdx * _dimOffsets[fixedDims];
            }

            jlongArray jFixedDimNums = (local_tls.jenv)->NewLongArray(fixedDims);
            if (jFixedDimNums == NULL) {
                return services::Status(services::UnknownError); /* out of memory error thrown */
            }
            (local_tls.jenv)->SetLongArrayRegion(jFixedDimNums, 0, fixedDims, (jlong*)fixedDimNums);

            size_t bufferSize = blockSize * sizeof(T);
            local_tls.jbuf = (local_tls.jenv)->NewDirectByteBuffer( subtensor.getPtr(), bufferSize );

            /* Call 'releaseSubtensor' Java method */
            (local_tls.jenv)->CallObjectMethod(
                jJavaTensor, jmeth, jFixedDimNums, (jlong)rangeDimIdx, (jlong)rangeDimNum, local_tls.jbuf);

            DAAL_CHECK_STATUS(status, daal::internal::detachCurrentThread(jvm, local_tls))

            daal::services::daal_free( _dimOffsets );
        }
        subtensor.reset();

        tls.local() = local_tls;
        return status;
    }

    DAAL_DEPRECATED_VIRTUAL virtual TensorPtr getSampleTensor(size_t firstDimIndex) DAAL_C11_OVERRIDE
    {
        return TensorPtr();
    }

    virtual services::Status check(const char *description) const DAAL_C11_OVERRIDE
    {
        return services::Status();
    }

    virtual TensorOffsetLayout createDefaultSubtensorLayout() const DAAL_C11_OVERRIDE
    {
        return TensorOffsetLayout(_layout);
    }

    virtual TensorOffsetLayout createRawSubtensorLayout() const DAAL_C11_OVERRIDE
    {
        TensorOffsetLayout layout(_layout);
        layout.sortOffsets();
        return layout;
    }

    virtual jobject getJavaObject() const DAAL_C11_OVERRIDE
    {
        return jJavaTensor;
    }

protected:
    virtual services::Status allocateDataMemoryImpl(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        return services::Status();
    }

    virtual services::Status freeDataMemoryImpl() DAAL_C11_OVERRIDE { return services::Status(); }

    tbb::enumerable_thread_specific<daal::internal::_java_tls> tls;  /**< Thread local storage */
    jobject jJavaTensor;                                             /**< Java object associated with this C++ object */
    JavaVM *jvm;                                                     /**< Java VM interface function table */

private:
    TensorOffsetLayout _layout;
};

} // namespace daal

#endif
