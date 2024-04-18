

#ifndef __FINITENESS_CHECKER_AVX2_IMPL_I__
#define __FINITENESS_CHECKER_AVX2_IMPL_I__


template <typename DataType>
DataType sumWithAVX2(size_t n, const DataType * dataPtr)
{
    constexpr size_t avx2RegisterLength = 256;
    constexpr size_t numberOfBitsInByte = 8;
    constexpr size_t nPerInstr = avx2RegisterLength / (numberOfBitsInByte * sizeof(DataType));
    DataType sum;
    if (sizeof(DataType) == 4)
    {
        __m256 sums     = _mm256_set1_ps(0);
        __m256 * ptr256 = (__m256 *)dataPtr;
        for (size_t i = 0; i < n / nPerInstr; i++) sums = _mm256_add_ps(sums, ptr256[i]);

        // AVX2 doesn't have reduce_add_ps, so finer-grained used of intrinsics necessary
        __m128 s0    = _mm256_castps256_ps128(sums);
        __m128 s1    = _mm256_extractf128_ps(sums, 1);
        s0           = _mm_add_ps(s0, s1);
        __m128 stemp = _mm_movehl_ps(s0, s0);
        __m128 sD    = _mm_add_ps(s0, stemp);
        __m128 hi    = _mm_shuffle_ps(sD, sD, 0x1);

        sum = _mm_cvtss_f32(_mm_add_ss(sD, hi));
    }
    else
    {
        __m256d sums     = _mm256_set1_pd(0);
        __m256d * ptr256 = (__m256d *)dataPtr;
        for (size_t i = 0; i < n / nPerInstr; i++) sums = _mm256_add_pd(sums, ptr256[i]);

        // AVX2 doesn't have reduce_add_pd, so finer-grained used of intrinsics necessary
        __m128d s0    = _mm256_castpd256_pd128(sums);
        __m128d s1    = _mm256_extractf128_pd(sums, 1);
        s0            = _mm_add_pd(s0, s1);
        __m128d stemp = _mm_unpackhi_pd(s0, s0);

        sum = _mm_cvtsd_f64(_mm_add_sd(s0, stemp));
    }
    for (size_t i = (n / nPerInstr) * nPerInstr; i < n; ++i) sum += dataPtr[i];

    return sum;
}

template <typename DataType>
DataType computeSumAVX2Impl(size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs)
{
    size_t nBlocksPerPtr = nElementsPerPtr / BLOCK_SIZE;
    if (nBlocksPerPtr == 0) nBlocksPerPtr = 1;
    size_t nElements    = nDataPtrs * nElementsPerPtr;
    bool inParallel     = !(nElements < THREADING_BORDER);
    size_t nPerBlock    = nElementsPerPtr / nBlocksPerPtr;
    size_t nSurplus     = nElementsPerPtr % nBlocksPerPtr;
    size_t nTotalBlocks = nBlocksPerPtr * nDataPtrs;

    daal::services::internal::TArray<DataType, avx2> partialSumsArr(nTotalBlocks);
    DataType * pSums = partialSumsArr.get();
    if (!pSums) return getInf<DataType>();
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) pSums[iBlock] = 0;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;

        pSums[iBlock] = sumWithAVX2<DataType>(end - start, dataPtrs[ptrIdx] + start);
    });

    return sumWithAVX2<DataType>(nTotalBlocks, pSums);
}

template <>
float computeSum<float, avx2>(size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs)
{
    return computeSumAVX2Impl<float>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

template <>
double computeSum<double, avx2>(size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs)
{
    return computeSumAVX2Impl<double>(nDataPtrs, nElementsPerPtr, dataPtrs);
}

double computeSumSOAAVX2Impl(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    SafeStatus safeStat;
    double sum                                  = 0;
    bool breakFlag                              = false;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    daal::TlsMem<double, avx2, services::internal::ScalableCalloc<double, avx2> > tlsSum(1);
    daal::TlsMem<bool, avx2, services::internal::ScalableCalloc<bool, avx2> > tlsNotFinite(1);

    daal::threader_for_break(nCols, nCols, [&](size_t i, bool & needBreak) {
        double * localSum     = tlsSum.local();
        bool * localNotFinite = tlsNotFinite.local();
        DAAL_CHECK_MALLOC_THR(localSum);
        DAAL_CHECK_MALLOC_THR(localNotFinite);

        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, avx2> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const float * colPtr = colBlock.get();
            *localSum += static_cast<double>(computeSum<float, avx2>(1, nRows, &colPtr));
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, avx2> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const double * colPtr = colBlock.get();
            *localSum += computeSum<double, avx2>(1, nRows, &colPtr);
            break;
        }
        default: break;
        }

        *localNotFinite |= valuesAreNotFinite(localSum, 1, false);
        if (*localNotFinite)
        {
            needBreak = true;
            breakFlag = true;
        }
    });

    st |= safeStat.detach();
    if (!st)
    {
        return 0;
    }

    if (breakFlag)
    {
        sum         = getInf<double>();
        sumIsFinite = false;
    }
    else
    {
        tlsSum.reduce([&](double * localSum) { sum += *localSum; });
        tlsNotFinite.reduce([&](bool * localNotFinite) { sumIsFinite &= !*localNotFinite; });
    }

    return sum;
}

template <>
double computeSumSOA<avx2>(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    return computeSumSOAAVX2Impl(table, sumIsFinite, st);
}

services::Status checkFinitenessInBlocks256(const float ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                            size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    constexpr size_t nPerInstr = 8;
    services::internal::TArray<bool, avx2> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        // create masks for exponent and fraction parts of FP type and zero register
        __m256i exp256Mask  = _mm256_set1_epi32(floatExpMask);
        __m256i frac256Mask = _mm256_set1_epi32(floatFracMask);
        __m256i zero256     = _mm256_setzero_si256();
        __m256i endMask     = _mm256_setzero_si256();

        __m256i notAllowNaNMask = allowNaN ? _mm256_setzero_si256() : _mm256_set1_epi64x(-1);

        __m256i * ptr256i = (__m256i *)(dataPtrs[ptrIdx] + start);

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            // apply masks
            __m256i expBits  = _mm256_and_si256(exp256Mask, ptr256i[i]);
            __m256i fracBits = _mm256_and_si256(frac256Mask, ptr256i[i]);

            __m256i expAreOnes   = _mm256_cmpeq_epi32(exp256Mask, expBits);
            __m256i fracAreZeros = _mm256_cmpeq_epi32(zero256, fracBits);

            // "values aren't finite" = "exponent bits are ones" AND ( "fraction bits are zeros" OR NOT "NaN is allowed" )
            __m256i orMask    = _mm256_or_si256(fracAreZeros, notAllowNaNMask);
            __m256i finalMask = _mm256_and_si256(expAreOnes, orMask);
            endMask           = _mm256_or_si256(endMask, finalMask); // collect ones for final check
        }
        if (_mm256_testz_si256(endMask, endMask) != 1) notFinitePtr[iBlock] = true;
        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        notFinitePtr[iBlock] |= valuesAreNotFinite(dataPtrs[ptrIdx] + offset, end - offset, allowNaN);
    });

    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock)
        if (notFinitePtr[iBlock])
        {
            finiteness = false;
            return s;
        }
    finiteness = true;
    return s;
}

services::Status checkFinitenessInBlocks256(const double ** dataPtrs, bool inParallel, size_t nTotalBlocks, size_t nBlocksPerPtr, size_t nPerBlock,
                                            size_t nSurplus, bool allowNaN, bool & finiteness)
{
    services::Status s;
    const size_t nPerInstr = 4;
    services::internal::TArray<bool, avx2> notFiniteArr(nTotalBlocks);
    bool * notFinitePtr = notFiniteArr.get();
    DAAL_CHECK_MALLOC(notFinitePtr);
    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock) notFinitePtr[iBlock] = false;

    daal::conditional_threader_for(inParallel, nTotalBlocks, [&](size_t iBlock) {
        size_t ptrIdx        = iBlock / nBlocksPerPtr;
        size_t blockIdxInPtr = iBlock - nBlocksPerPtr * ptrIdx;
        size_t start         = blockIdxInPtr * nPerBlock;
        size_t end           = blockIdxInPtr == nBlocksPerPtr - 1 ? start + nPerBlock + nSurplus : start + nPerBlock;
        size_t lcSize        = end - start;

        // create masks for exponent and fraction parts of FP type and zero register
        __m256i exp256Mask  = _mm256_set1_epi64x(doubleExpMask);
        __m256i frac256Mask = _mm256_set1_epi64x(doubleFracMask);
        __m256i zero256     = _mm256_setzero_si256();
        __m256i endMask     = _mm256_setzero_si256();

        __m256i notAllowNaNMask = allowNaN ? _mm256_setzero_si256() : _mm256_set1_epi64x(-1);

        __m256i * ptr256i = (__m256i *)(dataPtrs[ptrIdx] + start);

        for (size_t i = 0; i < lcSize / nPerInstr; ++i)
        {
            // apply masks
            __m256i expBits  = _mm256_and_si256(exp256Mask, ptr256i[i]);
            __m256i fracBits = _mm256_and_si256(frac256Mask, ptr256i[i]);

            __m256i expAreOnes   = _mm256_cmpeq_epi64(exp256Mask, expBits);
            __m256i fracAreZeros = _mm256_cmpeq_epi64(zero256, fracBits);

            // "values aren't finite" = "exponent bits are ones" AND ( "fraction bits are zeros" OR NOT "NaN is allowed" )
            __m256i orMask    = _mm256_or_si256(fracAreZeros, notAllowNaNMask);
            __m256i finalMask = _mm256_and_si256(expAreOnes, orMask);
            endMask           = _mm256_or_si256(endMask, finalMask); // collect ones for final check
        }
        if (_mm256_testz_si256(endMask, endMask) != 1) notFinitePtr[iBlock] = true;

        size_t offset = start + (lcSize / nPerInstr) * nPerInstr;
        notFinitePtr[iBlock] |= valuesAreNotFinite(dataPtrs[ptrIdx] + offset, end - offset, allowNaN);
    });

    for (size_t iBlock = 0; iBlock < nTotalBlocks; ++iBlock)
        if (notFinitePtr[iBlock])
        {
            finiteness = false;
            return s;
        }
    finiteness = true;
    return s;
}

template <typename DataType>
bool checkFinitenessAVX2Impl(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs, bool allowNaN)
{
    size_t nBlocksPerPtr = nElementsPerPtr / BLOCK_SIZE;
    if (nBlocksPerPtr == 0) nBlocksPerPtr = 1;
    bool inParallel     = !(nElements < THREADING_BORDER);
    size_t nPerBlock    = nElementsPerPtr / nBlocksPerPtr;
    size_t nSurplus     = nElementsPerPtr % nBlocksPerPtr;
    size_t nTotalBlocks = nBlocksPerPtr * nDataPtrs;

    bool finiteness;
    checkFinitenessInBlocks256(dataPtrs, inParallel, nTotalBlocks, nBlocksPerPtr, nPerBlock, nSurplus, allowNaN, finiteness);
    return finiteness;
}

template <>
bool checkFiniteness<float, avx2>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const float ** dataPtrs, bool allowNaN)
{
    return checkFinitenessAVX2Impl<float>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

template <>
bool checkFiniteness<double, avx2>(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const double ** dataPtrs, bool allowNaN)
{
    return checkFinitenessAVX2Impl<double>(nElements, nDataPtrs, nElementsPerPtr, dataPtrs, allowNaN);
}

bool checkFinitenessSOAAVX2Impl(NumericTable & table, bool allowNaN, services::Status & st)
{
    SafeStatus safeStat;
    bool valuesAreFinite                        = true;
    bool breakFlag                              = false;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    daal::TlsMem<bool, avx2, services::internal::ScalableCalloc<bool, avx2> > tlsNotFinite(1);

    daal::threader_for_break(nCols, nCols, [&](size_t i, bool & needBreak) {
        bool * localNotFinite = tlsNotFinite.local();
        DAAL_CHECK_MALLOC_THR(localNotFinite);

        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, avx2> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const float * colPtr = colBlock.get();
            *localNotFinite |= !checkFiniteness<float, avx2>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, avx2> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS_THR(colBlock);
            const double * colPtr = colBlock.get();
            *localNotFinite |= !checkFiniteness<double, avx2>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        default: break;
        }

        if (*localNotFinite)
        {
            needBreak = true;
            breakFlag = true;
        }
    });

    st |= safeStat.detach();
    if (!st)
    {
        return false;
    }

    if (breakFlag)
    {
        valuesAreFinite = false;
    }
    else
    {
        tlsNotFinite.reduce([&](bool * localNotFinite) { valuesAreFinite &= !*localNotFinite; });
    }

    return valuesAreFinite;
}

template <>
bool checkFinitenessSOA<avx2>(NumericTable & table, bool allowNaN, services::Status & st)
{
    return checkFinitenessSOAAVX2Impl(table, allowNaN, st);
}

#endif __FINITENESS_CHECKER_AVX2_IMPL_I__