/* file: low_order_moments_estimates_batch.i */
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

using namespace daal::services;
/* Common structure with arrays and variables */
template<typename algorithmFPType, CpuType cpu>
struct common_moments_data_t
{
    common_moments_data_t( NumericTable *dataTable, Result *r )  : dataTable(dataTable)
    {
            malloc_errors = 0;

            nVectors  = dataTable->getNumberOfRows();
            nFeatures = dataTable->getNumberOfColumns();

            dataTable->getBlockOfRows(0, 1 /*nVectors*/, readOnly, firstRowBD);
            firstRow = firstRowBD.getBlockPtr();
            if(!firstRow)
            {
                malloc_errors++;
                return;
            }

            for (size_t i = 0; i < lastResultId + 1; i++)
            {
                resultTable[i] = r->get((ResultId)i);
                resultTable[i]->getBlockOfRows(0, 1, writeOnly, resultBD[i]);
                resultArray[i] = resultBD[i].getBlockPtr();
                if(!(resultArray[i]))
                {
                    malloc_errors++;
                    return;
                }
            }

#ifdef _MEAN_ENABLE_
            daal::services::internal::service_memset<algorithmFPType,cpu>(resultArray[(int)mean], 0, nFeatures);
#endif
#ifdef _SORM_ENABLE_
            daal::services::internal::service_memset<algorithmFPType,cpu>(resultArray[(int)secondOrderRawMoment], 0, nFeatures);
#endif
#if defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_
            daal::services::internal::service_memset<algorithmFPType,cpu>(resultArray[(int)variance], 0, nFeatures);
#endif
#ifdef _SUM_ENABLE_
            daal::services::internal::service_memset<algorithmFPType,cpu>(resultArray[(int)sum], 0, nFeatures);
#endif
#ifdef _SUM2_ENABLE_
            daal::services::internal::service_memset<algorithmFPType,cpu>(resultArray[(int)sumSquares], 0, nFeatures);
#endif

            /* Initialize min and max arrays by first input row */
#if (defined _MIN_ENABLE_ || defined _MAX_ENABLE_)
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for(int j = 0; j < nFeatures; j++)
            {
#ifdef _MIN_ENABLE_
                (resultArray[(int)minimum])[j] = firstRow[j];
#endif
#ifdef _MAX_ENABLE_
                (resultArray[(int)maximum])[j] = firstRow[j];
#endif
            }
#endif /* #if (defined _MIN_ENABLE_ || defined _MAX_ENABLE_) */
    return;
    }

    ~common_moments_data_t()
    {
            dataTable->releaseBlockOfRows(firstRowBD);
            for (size_t i = 0; i < lastResultId + 1; i++)
            {
                resultTable[i]->releaseBlockOfRows(resultBD[i]);
            }
    }

    int malloc_errors;

    size_t nVectors;
    size_t nFeatures;

    NumericTable *dataTable;
    NumericTablePtr resultTable[lastResultId + 1];

    BlockDescriptor<algorithmFPType> firstRowBD;
    BlockDescriptor<algorithmFPType> resultBD[lastResultId + 1];

    algorithmFPType *firstRow;
    algorithmFPType *resultArray[lastResultId + 1];

};


/* TLS structure with local arrays and variables */
template<typename algorithmFPType, CpuType cpu>
struct tls_moments_data_t
{
    int malloc_errors;
    algorithmFPType  nvectors;

#ifdef _MEAN_ENABLE_
    algorithmFPType* mean;
#endif
#ifdef _SORM_ENABLE_
    algorithmFPType* sorm;
#endif
#ifdef _SUM_ENABLE_
    algorithmFPType* sum;
#endif
#ifdef _SUM2_ENABLE_
    algorithmFPType* sum2;
#endif
#if defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
    algorithmFPType* varc;
#endif
#ifdef _MIN_ENABLE_
    algorithmFPType* min;
#endif
#ifdef _MAX_ENABLE_
    algorithmFPType* max;
#endif

    tls_moments_data_t(size_t nFeatures, algorithmFPType* input)
    {
        malloc_errors = 0;
        nvectors = 0;

#ifdef _MEAN_ENABLE_
        mean = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#ifdef _SORM_ENABLE_
        sorm = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#ifdef _SUM_ENABLE_
        sum  = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#ifdef _SUM2_ENABLE_
        sum2 = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#if defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
        varc = daal::services::internal::service_scalable_calloc<algorithmFPType, cpu>(nFeatures);
#endif
#ifdef _MIN_ENABLE_
        min  = daal::services::internal::service_scalable_malloc<algorithmFPType, cpu>(nFeatures);
#endif
#ifdef _MAX_ENABLE_
        max  = daal::services::internal::service_scalable_malloc<algorithmFPType, cpu>(nFeatures);
#endif

        if (
#ifdef _MEAN_ENABLE_
             (!mean) ||
#endif
#ifdef _SORM_ENABLE_
             (!sorm) ||
#endif
#ifdef _SUM_ENABLE_
             (!sum) ||
#endif
#ifdef _SUM2_ENABLE_
             (!sum2) ||
#endif
#if defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
             (!varc) ||
#endif
#ifdef _MIN_ENABLE_
             (!min) ||
#endif
#ifdef _MAX_ENABLE_
             (!max) ||
#endif
             false
             ) { malloc_errors++; return; }

        /* Initialize min and max arrays by first input row */
#if (defined _MIN_ENABLE_ || defined _MAX_ENABLE_)
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for(int j = 0; j < nFeatures; j++)
        {
#ifdef _MIN_ENABLE_
            min[j] = input[j];
#endif
#ifdef _MAX_ENABLE_
            max[j] = input[j];
#endif
        }
#endif /* #if (defined _MIN_ENABLE_ || defined _MAX_ENABLE_) */
    }

    ~tls_moments_data_t()
    {
#ifdef _MEAN_ENABLE_
        if(mean){ daal::services::internal::service_scalable_free<algorithmFPType,cpu>( mean ); mean = 0; }
#endif
#ifdef _SORM_ENABLE_
        if(sorm){ daal::services::internal::service_scalable_free<algorithmFPType,cpu>( sorm ); sorm = 0; }
#endif
#ifdef _SUM_ENABLE_
        if(sum) { daal::services::internal::service_scalable_free<algorithmFPType,cpu>( sum );  sum = 0; }
#endif
#ifdef _SUM2_ENABLE_
        if(sum2){ daal::services::internal::service_scalable_free<algorithmFPType,cpu>( sum2 ); sum2 = 0; }
#endif
#if defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
        if(varc){ daal::services::internal::service_scalable_free<algorithmFPType,cpu>( varc ); varc = 0; }
#endif
#ifdef _MIN_ENABLE_
        if(min) { daal::services::internal::service_scalable_free<algorithmFPType,cpu>( min );  min = 0; }
#endif
#ifdef _MAX_ENABLE_
        if(max) { daal::services::internal::service_scalable_free<algorithmFPType,cpu>( max );  max = 0; }
#endif
    }
};

template<typename algorithmFPType, CpuType cpu>
Status compute_estimates(NumericTable *dataTable, Result *result)
{
    /* Common data structure */
    common_moments_data_t<algorithmFPType,cpu> _cd(dataTable,result);
    if(_cd.malloc_errors)
    return Status(daal::services::ErrorMemoryAllocationFailed);

    /* "Short names" for result arrays */
    algorithmFPType* _min   = _cd.resultArray[(int)minimum];
    algorithmFPType* _max   = _cd.resultArray[(int)maximum];
    algorithmFPType* _sum   = _cd.resultArray[(int)sum];
    algorithmFPType* _sum2  = _cd.resultArray[(int)sumSquares];
    algorithmFPType* _sum2c = _cd.resultArray[(int)sumSquaresCentered];
    algorithmFPType* _mean  = _cd.resultArray[(int)mean];
    algorithmFPType* _sorm  = _cd.resultArray[(int)secondOrderRawMoment];
    algorithmFPType* _varc  = _cd.resultArray[(int)variance];
    algorithmFPType* _stdev = _cd.resultArray[(int)standardDeviation];
    algorithmFPType* _vart  = _cd.resultArray[(int)variation];

    /* Rows and features splitting by blocks */
    const size_t blockSize          = getDefaultBatchModeBlockSize<cpu>(_cd.nVectors);
    const size_t numRowsInBlock     = (_cd.nVectors > blockSize) ? blockSize : _cd.nVectors;
    const size_t numRowsBlocks      = _cd.nVectors / numRowsInBlock;
    const size_t numRowsInLastBlock = numRowsInBlock + ( _cd.nVectors - numRowsBlocks * numRowsInBlock);

#undef _FEATURE_BLOCK_SIZE_
#define _FEATURE_BLOCK_SIZE_ 32
#undef _THREAD_REDUCTION_MIN_SIZE_
#define _THREAD_REDUCTION_MIN_SIZE_ 128
    size_t numFeaturesInBlock = (_cd.nFeatures > _FEATURE_BLOCK_SIZE_)?_FEATURE_BLOCK_SIZE_:_cd.nFeatures;
    size_t numFeatureBlocks   = _cd.nFeatures / numFeaturesInBlock;
    size_t numFeaturesInLastBlock = numFeaturesInBlock + ( _cd.nFeatures - numFeatureBlocks * numFeaturesInBlock);

    /* TLS buffers initialization */
    daal::tls<tls_moments_data_t<algorithmFPType, cpu> *> tls_data([ & ]()
    {
        return new tls_moments_data_t<algorithmFPType, cpu>( _cd.nFeatures, _cd.firstRow );
    });

    SafeStatus safeStat;
    /* Compute partial results for each TLS buffer */
    daal::threader_for( numRowsBlocks, numRowsBlocks, [ & ](int iBlock)
    {
        struct tls_moments_data_t<algorithmFPType,cpu> * _td = tls_data.local();
        if(_td->malloc_errors)
        {
            return;
        }

        size_t _startRow = iBlock * numRowsInBlock;
        size_t _nRows    = (iBlock < (numRowsBlocks-1))?numRowsInBlock:numRowsInLastBlock;

        daal::internal::ReadRows<algorithmFPType, cpu, NumericTable> dataTableBD(dataTable, _startRow, _nRows);
        DAAL_CHECK_BLOCK_STATUS_THR(dataTableBD);
        const algorithmFPType* _dataArray_block = dataTableBD.get();

        for(int i = 0; i < _nRows; i++)
        {
            /* loop invariants */
#if defined _MEAN_ENABLE_  || defined _SORM_ENABLE_
            algorithmFPType _invN = algorithmFPType(1.0) / algorithmFPType(_td->nvectors+1);
#endif
           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for(int j = 0; j < _cd.nFeatures; j++)
            {
                algorithmFPType arg  = _dataArray_block[i*_cd.nFeatures + j];

#if defined _SUM2_ENABLE_ || defined _SORM_ENABLE_
                algorithmFPType arg2   = arg * arg;
#endif
#if defined _MEAN_ENABLE_ || defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                algorithmFPType delta  = arg  - _td->mean[j];
#endif
#ifdef _SORM_ENABLE_
                algorithmFPType delta2 = arg2 - _td->sorm[j];
#endif

#ifdef _MIN_ENABLE_
                if(arg < _td->min[j]) _td->min[j] = arg;
#endif
#ifdef _MAX_ENABLE_
                if(arg > _td->max[j]) _td->max[j] = arg;
#endif
#ifdef _SUM_ENABLE_
                _td->sum[j]  += arg;
#endif
#ifdef _SUM2_ENABLE_
                _td->sum2[j] += arg2;
#endif
#ifdef _MEAN_ENABLE_
                _td->mean[j] += delta  * _invN;
#endif
#ifdef _SORM_ENABLE_
                _td->sorm[j] += delta2 * _invN;
#endif
#if defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                _td->varc[j] += delta * ( arg - _td->mean[j] );
#endif
            }

            _td->nvectors++;
        }
    } );

    /* Number of already merged values */
    algorithmFPType n_current = 0;

    bool bMemoryAllocationFailed = false;
    /* Merge results by TLS buffers */
    tls_data.reduce( [ & ]( tls_moments_data_t<algorithmFPType,cpu>* _td )
    {
        if(_td->malloc_errors)
        {
            bMemoryAllocationFailed = true;
            delete _td;
            return;
        }
        if(!safeStat)
        {
            delete _td;
            return;
        }
        /* loop invariants */
        algorithmFPType n1_p_n2        = n_current + _td->nvectors;
        algorithmFPType n1_m_n2        = n_current * _td->nvectors;
        algorithmFPType delta_scale    = n1_m_n2 / n1_p_n2;
        algorithmFPType mean_scale     = algorithmFPType(1.0) / (n1_p_n2);
        algorithmFPType variance_scale = algorithmFPType(1.0) / (n1_p_n2 - algorithmFPType(1.0));

        /* Make reduction stage threaded for wide matrices */
        if(_cd.nFeatures >= _THREAD_REDUCTION_MIN_SIZE_)
        {
            daal::threader_for( numFeatureBlocks, numFeatureBlocks, [ & ](int iFeatureBlock)
            {
                size_t _jstart = iFeatureBlock * numFeaturesInBlock;
                size_t _jend   = _jstart + ((iFeatureBlock < (numFeatureBlocks-1))?numFeaturesInBlock:numFeaturesInLastBlock);

               PRAGMA_IVDEP
               PRAGMA_VECTOR_ALWAYS
                for( int j = _jstart; j < _jend; j++ )
                {
#if defined _MEAN_ENABLE_ || defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                    algorithmFPType delta  = _td->mean[j] - _mean[j];
#endif
#ifdef _MIN_ENABLE_
                    if(_td->min[j] < _min[j]) _min[j] = _td->min[j]; /* merging min */
#endif
#ifdef _MAX_ENABLE_
                    if(_td->max[j] > _max[j]) _max[j] = _td->max[j]; /* merging max */
#endif
#ifdef _SUM_ENABLE_
                    _sum[j]  += _td->sum[j]; /* merging sums */
#endif
#ifdef _SUM2_ENABLE_
                    _sum2[j] += _td->sum2[j]; /* merging sum2 */
#endif
#if defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                    _sum2c[j] = ( _td->varc[j] + _varc[j]*(n_current-1) + delta*delta*delta_scale ); /* merging _sum2c */
#endif
#if defined _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                    _varc[j]  =  _sum2c[j]*variance_scale; /* merging variances */
#endif
#ifdef _MEAN_ENABLE_
                    _mean[j]  = ( _mean[j]*n_current + _td->mean[j]*_td->nvectors )*mean_scale; /* merging means */
#endif
#ifdef _SORM_ENABLE_
                    _sorm[j]  = ( _sorm[j]*n_current + _td->sorm[j]*_td->nvectors )*mean_scale; /* merging sorms */
#endif
                }
             });
        } /* if(_cd.nFeatures >= _THREAD_REDUCTION_MIN_SIZE_) */
        else /* if(_cd.nFeatures < _THREAD_REDUCTION_MIN_SIZE_) */
        {
           PRAGMA_IVDEP
           PRAGMA_VECTOR_ALWAYS
            for( int j = 0; j < _cd.nFeatures; j++ )
            {
#if defined _MEAN_ENABLE_ || defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                algorithmFPType delta  = _td->mean[j] - _mean[j];
#endif
#ifdef _MIN_ENABLE_
                if(_td->min[j] < _min[j]) _min[j] = _td->min[j]; /* merging min */
#endif
#ifdef _MAX_ENABLE_
                if(_td->max[j] > _max[j]) _max[j] = _td->max[j]; /* merging max */
#endif
#ifdef _SUM_ENABLE_
                _sum[j]  += _td->sum[j]; /* merging sums */
#endif
#ifdef _SUM2_ENABLE_
                _sum2[j] += _td->sum2[j]; /* merging sum2 */
#endif
#if defined _SUM2C_ENABLE_ || defined  _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                _sum2c[j] = ( _td->varc[j] + _varc[j]*(n_current-1) + delta*delta*delta_scale ); /* merging _sum2c */
#endif
#if defined _VARC_ENABLE_  || defined _STDEV_ENABLE_ || defined _VART_ENABLE_
                _varc[j]  =  _sum2c[j]*variance_scale; /* merging variances */
#endif
#ifdef _MEAN_ENABLE_
                _mean[j]  = ( _mean[j]*n_current + _td->mean[j]*_td->nvectors )*mean_scale; /* merging means */
#endif
#ifdef _SORM_ENABLE_
                _sorm[j]  = ( _sorm[j]*n_current + _td->sorm[j]*_td->nvectors )*mean_scale; /* merging sorms */
#endif
            }
        } /* if(_cd.nFeatures < _THREAD_REDUCTION_MIN_SIZE_) */

        /* Increase number of already merged values */
        n_current += _td->nvectors;

        delete _td;
    } );

    if(bMemoryAllocationFailed)
        return Status(daal::services::ErrorMemoryAllocationFailed);

    DAAL_CHECK_SAFE_STATUS();

    /* Final loop for std deviations and variations */
#if (defined _STDEV_ENABLE_  || defined _VART_ENABLE_)
    daal::internal::Math<algorithmFPType, cpu>::vSqrt( _cd.nFeatures, &_varc[0], &_stdev[0] );
#ifdef _VART_ENABLE_
   PRAGMA_IVDEP
   PRAGMA_VECTOR_ALWAYS
    for(int j = 0; j < _cd.nFeatures; j++)
    {
            _vart[j]  = _stdev[j] / _mean[j];
    }
#endif
#endif /* #if (defined _STDEV_ENABLE_  || defined _VART_ENABLE_) */

    return Status();
} /* compute_estimates */
