/* file: fullyconnected_layer_forward_impl.i */
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
//  Implementation of fullyconnected algorithm
//--
*/

#ifndef __FULLYCONNECTED_LAYER_FORWARD_IMPL_I__
#define __FULLYCONNECTED_LAYER_FORWARD_IMPL_I__

#include "service_blas.h"
#include "threading.h"
#include "service_memory.h"

#define _DEFAULT_BLOCKSIZE 256
#define _SMALL_BLOCKSIZE   128

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace fullyconnected
{
namespace forward
{
namespace internal
{

using namespace daal::internal;
using namespace daal::services;

/* Common data structure */
template<typename algorithmFPType, CpuType cpu>
struct common_fullyconnected_data_t
{
    Status status;

    Tensor &inputTensor;
    Tensor &wTensor;
    Tensor &bTensor;
    Tensor &resultTensor;

    SubtensorDescriptor<algorithmFPType> inputBlock;
    SubtensorDescriptor<algorithmFPType> wBlock;
    SubtensorDescriptor<algorithmFPType> bBlock;
    SubtensorDescriptor<algorithmFPType> resultBlock;

    algorithmFPType *iArray;
    algorithmFPType *wArray;
    algorithmFPType *bArray;
    algorithmFPType *rArray;

    size_t blocknum      = 0;
    size_t blocksize     = 0;
    size_t lastblocksize = 0;

    size_t outs_num;
    size_t nDims;
    size_t batch_size;
    size_t data_size;
    size_t full_size;
    bool do_parallel;

    /* Common data constructor */
    common_fullyconnected_data_t( const Tensor &_inputTensor,
                                  const Tensor &_wTensor,
                                  const Tensor &_bTensor,
                                  Tensor &_resultTensor,
                                  const fullyconnected::Parameter &parameter ) :
                                                                                inputTensor(const_cast<Tensor &>(_inputTensor)),
                                                                                wTensor(const_cast<Tensor &>(_wTensor)),
                                                                                bTensor(const_cast<Tensor &>(_bTensor)),
                                                                                resultTensor(const_cast<Tensor &>(_resultTensor))
    {
        outs_num = parameter.nOutputs;

        const services::Collection<size_t>& inDims = inputTensor.getDimensions();
        const services::Collection<size_t>& wDims  = wTensor.getDimensions();

        nDims = inDims.size();

        TensorOffsetLayout inputLayout = inputTensor.createRawSubtensorLayout();

        status |= inputTensor.getSubtensor(0, 0, 0, inDims[0], readOnly, inputBlock); if(!status) return;
        status |= wTensor.getSubtensor(0, 0, 0, wDims[0], readOnly, wBlock); if(!status) return;
        status |= bTensor.getSubtensor(0, 0, 0, outs_num, readOnly, bBlock); if(!status) return;
        status |= resultTensor.getSubtensor(0, 0, 0, inDims[0], writeOnly, resultBlock); if(!status) return;

        iArray = inputBlock.getPtr();
        wArray = wBlock.getPtr();
        bArray = bBlock.getPtr();
        rArray = resultBlock.getPtr();

        batch_size = inDims[0];
        data_size  = 1;
        for(size_t i=1; i<nDims; i++)
        {
            data_size *= inDims[i];
        }
        full_size  = batch_size * data_size;

        /* Filtering dimensions where manual threading is slower */
        if(
            ( (batch_size == 1) && (outs_num >= 200) )
            || ( (batch_size * outs_num) > (100*data_size) )
#if __CPUID__(DAAL_CPU) >= __avx512_mic__
            || ( (batch_size >= 256) && (data_size <= 7000) && (outs_num <= 1000) )
#else
            || ( data_size <= 7000 )
            || ( (data_size <= 200000) && (outs_num > 256) )
#endif
          )
        {
            do_parallel   = 0;
        }
        else
        {
            do_parallel   = 1;

            /* Block size and amount calculation */
            blocksize     = (data_size > 10000)?_DEFAULT_BLOCKSIZE:_SMALL_BLOCKSIZE;
            blocksize     = (blocksize < data_size)? blocksize:data_size;
            blocknum      = data_size / blocksize;
            lastblocksize = data_size - blocknum*blocksize;
            if( lastblocksize != 0 ) { blocknum++; }
            else                     { lastblocksize = blocksize; }
        }
    }

    /* Common data destructor */
    ~common_fullyconnected_data_t()
    {
        if(iArray)inputTensor.releaseSubtensor(inputBlock);
        if(wArray)wTensor.releaseSubtensor(wBlock);
        if(bArray)bTensor.releaseSubtensor(bBlock);
        if(rArray)resultTensor.releaseSubtensor(resultBlock);
    }
};

/* TLS data structure */
template<typename algorithmFPType, CpuType cpu>
struct tls_fullyconnected_data_t
{
    Status status;
    algorithmFPType* result;

    /* TLS data constructor */
    tls_fullyconnected_data_t(size_t m, size_t n)
    {
        result = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>( m * n );
        if(!result) status = Status(ErrorMemoryAllocationFailed);
    }

    /* TLS data destructor */
    ~tls_fullyconnected_data_t()
    {
        if(result){ daal::services::internal::service_scalable_free<algorithmFPType,cpu>( result ); result = 0; }
    }
};

/* Main computation function for fully connected layer */
template<typename algorithmFPType, Method method, CpuType cpu>
Status FullyconnectedKernel<algorithmFPType, method, cpu>::compute( const Tensor &inputTensor,
                                                                  const Tensor &wTensor,
                                                                  const Tensor &bTensor,
                                                                  Tensor &resultTensor,
                                                                  const fullyconnected::Parameter &parameter )
{
    typedef typename Blas<algorithmFPType, cpu>::SizeType BlasSize;

    Status s;

    /* Allocate memory for common data and compute sizes */
    common_fullyconnected_data_t<algorithmFPType, cpu> _cd( inputTensor, wTensor, bTensor, resultTensor, parameter);
    DAAL_CHECK_STATUS(s, _cd.status)

    /* Copy biases array to result array */
    if( _cd.batch_size > 10 && _cd.outs_num > 100 )
    {
        daal::threader_for( _cd.batch_size, _cd.batch_size, [&](int j)
        {
          PRAGMA_IVDEP
          PRAGMA_VECTOR_ALWAYS
            for(size_t i=0; i<_cd.outs_num; i++)
            {
                _cd.rArray[j*_cd.outs_num + i] = _cd.bArray[i];
            }
        } );
    }
    else
    {
        for(size_t j=0; j < _cd.batch_size; j++)
        {
          PRAGMA_IVDEP
          PRAGMA_VECTOR_ALWAYS
            for(size_t i=0; i<_cd.outs_num; i++)
            {
                _cd.rArray[j*_cd.outs_num + i] = _cd.bArray[i];
            }
        }
    }

    /* Batch size = 1 case - vector by matrix multiplication */
    if( _cd.batch_size == 1 )
    {
        char trans            = 't';
        BlasSize _m           = _cd.data_size;
        BlasSize _n           = _cd.outs_num;
        BlasSize lda          = _cd.full_size;
        BlasSize incx         = 1;
        BlasSize incy         = 1;
        algorithmFPType alpha = 1.0;
        algorithmFPType beta  = 1.0;

        /* Manual threading branch */
        if( _cd.do_parallel )
        {
            /* Allocate TLS data */
            daal::tls<tls_fullyconnected_data_t<algorithmFPType, cpu> *> tls_data([ & ]()
            {
                return new tls_fullyconnected_data_t<algorithmFPType, cpu>( _cd.outs_num, _cd.batch_size );
            });

            SafeStatus safeStat;

            /* Run gemv split to blocks */
            daal::threader_for( _cd.blocknum, _cd.blocknum, [ & ](size_t b)
            {
                struct tls_fullyconnected_data_t<algorithmFPType,cpu> * _td = tls_data.local();
                safeStat |= _td->status;
                if(!_td->status) return;

                BlasSize cursize_local = (b < (_cd.blocknum-1))?_cd.blocksize:_cd.lastblocksize;

                Blas<algorithmFPType, cpu>::xxgemv
                (
                    &trans, &cursize_local, &_n, &alpha,
                    _cd.wArray + b*_cd.blocksize, &lda,
                    _cd.iArray + b*_cd.blocksize,
                    &incx, &beta, _td->result, &incy
                );
            } ); /* daal::threader_for */

            /* Sum all TLS arrays and release TLS memory */
            tls_data.reduce( [ & ]( tls_fullyconnected_data_t<algorithmFPType,cpu>* _td )
            {
                if(!safeStat.ok())
                {
                    delete _td;
                    return;
                }

              PRAGMA_IVDEP
              PRAGMA_VECTOR_ALWAYS
                for(size_t i = 0; i < (_cd.outs_num); i++)
                {
                    _cd.rArray[i] += _td->result[i];
                }

                delete _td;
            } ); /* tls_data.reduce */
            DAAL_CHECK_SAFE_STATUS();
        }
        else
        {
            Blas<algorithmFPType, cpu>::xgemv
            (
                &trans, &_m, &_n, &alpha,
                _cd.wArray, &lda,
                _cd.iArray,
                &incx, &beta, _cd.rArray, &incy
            );
        } /* if( _cd.do_parallel ) */
    }
    /* Batch size > 1 case - matrix by matrix multiplication */
    else
    {
        char transa           = 't';
        char transb           = 'n';
        BlasSize _m           = _cd.outs_num;
        BlasSize _n           = _cd.batch_size;
        BlasSize _k           = _cd.data_size;
        BlasSize lda          = _k;
        BlasSize ldb          = _k;
        BlasSize ldc          = _cd.outs_num;
        algorithmFPType alpha = 1.0;
        algorithmFPType beta  = 1.0;

        /* Manual threading branch */
        if( _cd.do_parallel )
        {
            /* Allocate TLS data */
            daal::tls<tls_fullyconnected_data_t<algorithmFPType, cpu> *> tls_data([ & ]()
            {
                return new tls_fullyconnected_data_t<algorithmFPType, cpu>( _cd.outs_num, _cd.batch_size );
            });

            SafeStatus safeStat;

            /* Run gemm split to blocks */
            daal::threader_for( _cd.blocknum, _cd.blocknum, [ & ](size_t b)
            {
                struct tls_fullyconnected_data_t<algorithmFPType,cpu> * _td = tls_data.local();
                safeStat |= _td->status;
                if(!_td->status) return;

                BlasSize cursize_local = (b < (_cd.blocknum-1))?_cd.blocksize:_cd.lastblocksize;

                Blas<algorithmFPType, cpu>::xxgemm
                (
                    &transa, &transb, &_m, &_n,
                    &cursize_local, &alpha,
                    _cd.wArray + b*_cd.blocksize, &lda,
                    _cd.iArray + b*_cd.blocksize, &ldb,
                    &beta,
                    _td->result, &ldc
                );
            } ); /* daal::threader_for */

            /* Sum all TLS arrays and release TLS memory */
            tls_data.reduce( [ & ]( tls_fullyconnected_data_t<algorithmFPType,cpu>* _td )
            {
                if(!safeStat.ok())
                {
                    delete _td;
                    return;
                }

                /* Additional threading for big size arrays */
                if( _cd.batch_size > 10 && _cd.outs_num > 100 )
                {
                    daal::threader_for( _cd.batch_size, _cd.batch_size, [&](int j)
                    {
                      PRAGMA_IVDEP
                      PRAGMA_VECTOR_ALWAYS
                        for(size_t i = 0; i < _cd.outs_num; i++)
                        {
                            _cd.rArray[j*_cd.outs_num + i] += _td->result[j*_cd.outs_num + i];
                        }
                    } );
                }
                /* Sequential reduction */
                else
                {
                  PRAGMA_IVDEP
                  PRAGMA_VECTOR_ALWAYS
                    for(size_t i = 0; i < (_cd.outs_num * _cd.batch_size); i++)
                    {
                        _cd.rArray[i] += _td->result[i];
                    }
                }

                delete _td;
            } ); /* tls_data.reduce */
            DAAL_CHECK_SAFE_STATUS();
        }
        else
        {
            Blas<algorithmFPType, cpu>::xgemm
            (
                &transa, &transb, &_m, &_n,
                &_k, &alpha,
                _cd.wArray, &lda,
                _cd.iArray, &ldb,
                &beta,
                _cd.rArray, &ldc
            );
        }  /* if( _cd.do_parallel ) */
    } /* if( _cd.batch_size == 1 ) */
    return Status();
} /* void FullyconnectedKernel<algorithmFPType, method, cpu>::compute */

} // internal
} // forward
} // namespace fullyconnected
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
