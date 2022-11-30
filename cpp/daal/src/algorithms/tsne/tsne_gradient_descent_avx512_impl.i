/* file: tsne_gradient_descent_avx512_impl.i */
/*******************************************************************************
* Copyright 2022 Intel Corporation
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
//  Parts of tSNE algorithm oprtimized for AVX512.
//--
*/

#ifndef __TSNE_GRADIENT_DESCENT_AVX512_IMPL_I__
#define __TSNE_GRADIENT_DESCENT_AVX512_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace internal
{

/* Partial template specialization of attractive kernel for single precision data and AVX512 ISA */
template <bool DivComp, typename IdxType>
struct AttractiveKernel<DivComp, IdxType, float, avx512>
{
    static services::Status impl(const float * val, const IdxType * col, const size_t * row, MemoryCtxType<IdxType, xyType<float>, avx512> & mem,
                                 float & zNorm, float & divergence, const IdxType N, const IdxType nnz, const IdxType nElements,
                                 const float exaggeration)
    {
        DAAL_CHECK_MALLOC(val);
        DAAL_CHECK_MALLOC(col);
        DAAL_CHECK_MALLOC(row);

        const float multiplier = exaggeration * float(zNorm);
        divergence             = 0.;

        const IdxType prefetch_dist = 32;

        daal::TlsSum<float, avx512> divTlsData(1);
        daal::tls<float *> logTlsData([=]() { return services::internal::service_scalable_calloc<float, avx512>(nElements); });

        const IdxType nThreads    = threader_get_threads_number();
        const IdxType sizeOfBlock = services::internal::min<avx512, size_t>(256, N / nThreads + 1);
        const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

        daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
            const IdxType iStart = iBlock * sizeOfBlock;
            const IdxType iEnd   = services::internal::min<avx512, IdxType>(N, iStart + sizeOfBlock);
            float * logLocal     = logTlsData.local();
            float * divLocal     = divTlsData.local();

            xyType<float> row_point;
            IdxType iCol, prefetch_index;
            float y1d, y2d, sqDist, PQ;

            for (IdxType iRow = iStart; iRow < iEnd; ++iRow)
            {
                size_t iSize     = 0;
                mem._attr[iRow].x = 0.0;
                mem._attr[iRow].y = 0.0;
                row_point        = mem._pos[iRow];

                if (!DivComp)
                {
                    IdxType start_index = row[iRow] - 1;
                    IdxType range       = row[iRow + 1] - row[iRow];

                    __m512 vec_1   = _mm512_set1_ps(1.0);
                    __m512i vec_1i = _mm512_set1_epi32(1);

                    __m512 vec_point_x = _mm512_set1_ps(row_point.x);
                    __m512 vec_point_y = _mm512_set1_ps(row_point.y);

                    __m512 vec_point_xs = _mm512_setzero_ps();
                    __m512 vec_point_ys = _mm512_setzero_ps();

                    for (IdxType i = 0; i < (range / 16) * 16; i += 16)
                    {
                        prefetch_index = start_index + i + prefetch_dist;
                        if (prefetch_index < nnz) _mm_prefetch(&mem._pos[col[prefetch_index] - 1], _MM_HINT_T0);

                        __m512i vec_iCol = _mm512_sub_epi32(_mm512_loadu_epi32((__m512i *)&col[start_index + i]), vec_1i);

                        __m512 vec_point_xd = _mm512_sub_ps(vec_point_x, _mm512_i32gather_ps(vec_iCol, &mem._pos[0].x, 8));
                        __m512 vec_point_yd = _mm512_sub_ps(vec_point_y, _mm512_i32gather_ps(vec_iCol, &mem._pos[0].y, 8));

                        __m512 vec_pq = _mm512_div_ps(
                            _mm512_loadu_ps((__m512 *)&val[start_index + i]),
                            _mm512_add_ps(_mm512_fmadd_ps(vec_point_xd, vec_point_xd, _mm512_mul_ps(vec_point_yd, vec_point_yd)), vec_1));

                        vec_point_xs = _mm512_fmadd_ps(vec_point_xd, vec_pq, vec_point_xs);
                        vec_point_ys = _mm512_fmadd_ps(vec_point_yd, vec_pq, vec_point_ys);
                    }

                    mem._attr[iRow].x += _mm512_reduce_add_ps(vec_point_xs);
                    mem._attr[iRow].y += _mm512_reduce_add_ps(vec_point_ys);

                    for (IdxType i = (range / 16) * 16; i < range; ++i)
                    {
                        prefetch_index = start_index + i + prefetch_dist;
                        if (prefetch_index < nnz) _mm_prefetch(&mem._pos[col[prefetch_index] - 1], _MM_HINT_T0);

                        iCol = col[start_index + i] - 1;

                        y1d = row_point.x - mem._pos[iCol].x;
                        y2d = row_point.y - mem._pos[iCol].y;

                        sqDist = 1.0 + y1d * y1d + y2d * y2d;
                        PQ     = val[start_index + i] / sqDist;

                        mem._attr[iRow].x += PQ * y1d;
                        mem._attr[iRow].y += PQ * y2d;
                    }
                }
                else
                {
                    for (size_t index = row[iRow] - 1; index < row[iRow + 1] - 1; ++index)
                    {
                        prefetch_index = index + prefetch_dist;
                        if (prefetch_index < nnz) _mm_prefetch(&mem._pos[col[prefetch_index] - 1], _MM_HINT_T0);
                        iCol = col[index] - 1;

                        y1d    = row_point.x - mem._pos[iCol].x;
                        y2d    = row_point.y - mem._pos[iCol].y;
                        sqDist = services::internal::max<avx512, float>(float(0), y1d * y1d + y2d * y2d);
                        PQ     = val[index] / (sqDist + 1.);

                        // Apply forces
                        mem._attr[iRow].x += PQ * y1d;
                        mem._attr[iRow].y += PQ * y2d;

                        logLocal[iSize++] = val[index] * multiplier * (1. + sqDist);
                    }

                    Math<float, avx512>::vLog(iSize, logLocal, logLocal);
                    IdxType start = row[iRow] - 1;
                    for (IdxType index = 0; index < iSize; ++index)
                    {
                        divLocal[0] += val[start + index] * logLocal[index]; // 2*NNZ Flop
                    }
                }
            }
        });

        divTlsData.reduceTo(&divergence, 1);
        divergence *= exaggeration;
        logTlsData.reduce([&](float * buf) { services::internal::service_scalable_free<float, avx512>(buf); });

        // Find_Normalization
        zNorm = float(1) / zNorm;

        return services::Status();
    }
};

/* Partial template specialization of attractive kernel for double precision data and AVX512 ISA */
template <bool DivComp, typename IdxType>
struct AttractiveKernel<DivComp, IdxType, double, avx512>
{
    static services::Status impl(const double * val, const IdxType * col, const size_t * row, MemoryCtxType<IdxType, xyType<double>, avx512> & mem,
                                 double & zNorm, double & divergence, const IdxType N, const IdxType nnz, const IdxType nElements,
                                 const double exaggeration)
    {
        DAAL_CHECK_MALLOC(val);
        DAAL_CHECK_MALLOC(col);
        DAAL_CHECK_MALLOC(row);

        const double multiplier = exaggeration * double(zNorm);
        divergence              = 0.;

        const IdxType prefetch_dist = 32;

        daal::TlsSum<double, avx512> divTlsData(1);
        daal::tls<double *> logTlsData([=]() { return services::internal::service_scalable_calloc<double, avx512>(nElements); });

        const IdxType nThreads    = threader_get_threads_number();
        const IdxType sizeOfBlock = services::internal::min<avx512, size_t>(256, N / nThreads + 1);
        const IdxType nBlocks     = N / sizeOfBlock + !!(N % sizeOfBlock);

        daal::threader_for(nBlocks, nBlocks, [&](IdxType iBlock) {
            const IdxType iStart = iBlock * sizeOfBlock;
            const IdxType iEnd   = services::internal::min<avx512, IdxType>(N, iStart + sizeOfBlock);
            double * logLocal    = logTlsData.local();
            double * divLocal    = divTlsData.local();

            xyType<double> row_point;
            IdxType iCol, prefetch_index;
            double y1d, y2d, sqDist, PQ;

            for (IdxType iRow = iStart; iRow < iEnd; ++iRow)
            {
                size_t iSize     = 0;
                mem._attr[iRow].x = 0.0;
                mem._attr[iRow].y = 0.0;
                row_point        = mem._pos[iRow];

                if (!DivComp)
                {
                    IdxType start_index = row[iRow] - 1;
                    IdxType range       = row[iRow + 1] - row[iRow];

                    __m512d vec_1  = _mm512_set1_pd(1.0);
                    __m256i vec_1i = _mm256_set1_epi32(1);

                    __m512d vec_point_x = _mm512_set1_pd(row_point.x);
                    __m512d vec_point_y = _mm512_set1_pd(row_point.y);

                    __m512d vec_point_xs = _mm512_setzero_pd();
                    __m512d vec_point_ys = _mm512_setzero_pd();

                    for (IdxType i = 0; i < (range / 8) * 8; i += 8)
                    {
                        prefetch_index = start_index + i + prefetch_dist;
                        if (prefetch_index < nnz) _mm_prefetch(&mem._pos[col[prefetch_index] - 1], _MM_HINT_T0);

                        __m256i vec_iCol = _mm256_slli_epi32(_mm256_sub_epi32(_mm256_loadu_epi32((__m256i *)&col[start_index + i]), vec_1i), 4);

                        __m512d vec_point_xd = _mm512_sub_pd(vec_point_x, _mm512_i32gather_pd(vec_iCol, &mem._pos[0].x, 1));
                        __m512d vec_point_yd = _mm512_sub_pd(vec_point_y, _mm512_i32gather_pd(vec_iCol, &mem._pos[0].y, 1));

                        __m512d vec_pq = _mm512_div_pd(
                            _mm512_loadu_pd((__m512d *)&val[start_index + i]),
                            _mm512_add_pd(_mm512_fmadd_pd(vec_point_xd, vec_point_xd, _mm512_mul_pd(vec_point_yd, vec_point_yd)), vec_1));

                        vec_point_xs = _mm512_fmadd_pd(vec_point_xd, vec_pq, vec_point_xs);
                        vec_point_ys = _mm512_fmadd_pd(vec_point_yd, vec_pq, vec_point_ys);
                    }

                    mem._attr[iRow].x += _mm512_reduce_add_pd(vec_point_xs);
                    mem._attr[iRow].y += _mm512_reduce_add_pd(vec_point_ys);

                    for (IdxType i = (range / 8) * 8; i < range; ++i)
                    {
                        prefetch_index = start_index + i + prefetch_dist;
                        if (prefetch_index < nnz) _mm_prefetch(&mem._pos[col[prefetch_index] - 1], _MM_HINT_T0);

                        iCol = col[start_index + i] - 1;

                        y1d = row_point.x - mem._pos[iCol].x;
                        y2d = row_point.y - mem._pos[iCol].y;

                        sqDist = 1.0 + y1d * y1d + y2d * y2d;
                        PQ     = val[start_index + i] / sqDist;

                        mem._attr[iRow].x += PQ * y1d;
                        mem._attr[iRow].y += PQ * y2d;
                    }
                }
                else
                {
                    for (size_t index = row[iRow] - 1; index < row[iRow + 1] - 1; ++index)
                    {
                        prefetch_index = index + prefetch_dist;
                        if (prefetch_index < nnz) _mm_prefetch(&mem._pos[col[prefetch_index] - 1], _MM_HINT_T0);

                        iCol = col[index] - 1;

                        y1d    = row_point.x - mem._pos[iCol].x;
                        y2d    = row_point.y - mem._pos[iCol].y;
                        sqDist = services::internal::max<avx512, double>(double(0), y1d * y1d + y2d * y2d);
                        PQ     = val[index] / (sqDist + 1.);

                        // Apply forces
                        mem._attr[iRow].x += PQ * y1d;
                        mem._attr[iRow].y += PQ * y2d;

                        logLocal[iSize++] = val[index] * multiplier * (1. + sqDist);
                    }

                    Math<double, avx512>::vLog(iSize, logLocal, logLocal);
                    IdxType start = row[iRow] - 1;
                    for (IdxType index = 0; index < iSize; ++index)
                    {
                        divLocal[0] += val[start + index] * logLocal[index]; // 2*NNZ Flop
                    }
                }
            }
        });

        divTlsData.reduceTo(&divergence, 1);
        divergence *= exaggeration;
        logTlsData.reduce([&](double * buf) { services::internal::service_scalable_free<double, avx512>(buf); });

        //Find_Normalization
        zNorm = double(1) / zNorm;
        // zNorm = double(1) / (zNorm - double(N));  // old code

        return services::Status();
    }
};

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif // __TSNE_GRADIENT_DESCENT_AVX512_IMPL_I__
