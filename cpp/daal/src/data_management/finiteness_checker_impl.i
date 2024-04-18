#ifndef __FINITENESS_CHECKER_IMPL_I__
#define __FINITENESS_CHECKER_IMPL_I__

template <typename DataType, daal::CpuType cpu>
DataType computeSum(size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs)
{
    DataType sum = 0;
    for (size_t ptrIdx = 0; ptrIdx < nDataPtrs; ++ptrIdx)
        for (size_t i = 0; i < nElementsPerPtr; ++i) sum += dataPtrs[ptrIdx][i];

    return sum;
}

template <daal::CpuType cpu>
double computeSumSOA(NumericTable & table, bool & sumIsFinite, services::Status & st)
{
    double sum                                  = 0;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    for (size_t i = 0; (i < nCols) && sumIsFinite; ++i)
    {
        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const float * colPtr = colBlock.get();
            sum += static_cast<double>(computeSum<float, cpu>(1, nRows, &colPtr));
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const double * colPtr = colBlock.get();
            sum += computeSum<double, cpu>(1, nRows, &colPtr);
            break;
        }
        default: break;
        }
        sumIsFinite &= !valuesAreNotFinite(&sum, 1, false);
    }

    return sum;
}

template <typename DataType, daal::CpuType cpu>
bool checkFiniteness(const size_t nElements, size_t nDataPtrs, size_t nElementsPerPtr, const DataType ** dataPtrs, bool allowNaN)
{
    bool notFinite = false;
    for (size_t ptrIdx = 0; ptrIdx < nDataPtrs; ++ptrIdx) notFinite = notFinite || valuesAreNotFinite(dataPtrs[ptrIdx], nElementsPerPtr, allowNaN);

    return !notFinite;
}

template <daal::CpuType cpu>
bool checkFinitenessSOA(NumericTable & table, bool allowNaN, services::Status & st)
{
    bool valuesAreFinite                        = true;
    const size_t nRows                          = table.getNumberOfRows();
    const size_t nCols                          = table.getNumberOfColumns();
    NumericTableDictionaryPtr tableFeaturesDict = table.getDictionarySharedPtr();

    for (size_t i = 0; (i < nCols) && valuesAreFinite; ++i)
    {
        switch ((*tableFeaturesDict)[i].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
        {
            ReadColumns<float, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const float * colPtr = colBlock.get();
            valuesAreFinite &= checkFiniteness<float, cpu>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
        {
            ReadColumns<double, cpu> colBlock(table, i, 0, nRows);
            DAAL_CHECK_BLOCK_STATUS(colBlock);
            const double * colPtr = colBlock.get();
            valuesAreFinite &= checkFiniteness<double, cpu>(nRows, 1, nRows, &colPtr, allowNaN);
            break;
        }
        default: break;
        }
    }

    return valuesAreFinite;
}
#endif // __FINITENESS_CHECKER_IMPL_I__