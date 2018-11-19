/* file: kmeans_plusplus_init_impl.i */
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
//  Implementation of K-means++ initialization algorithm.
//--
*/

#include "algorithm.h"
#include "numeric_table.h"
#include "threading.h"
#include "daal_defines.h"
#include "service_memory.h"
#include "service_error_handling.h"
#include "service_numeric_table.h"
#include "service_blas.h"
#include "service_spblas.h"
#include "uniform_kernel.h"
#include "service_data_utils.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace internal
{

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;
using namespace daal::algorithms::distributions::uniform::internal;

const size_t _nRowsInBlock = 512;

//BlockHelperXXX template class is a helper class for a block of rows.
//It provides multiplication of the block's rows to the given matrix of centers
//and the result access according to the data type.
//Single interface hides data-specific manipulations.

//BlockHelperDense is the helper class for the dense data type
template <typename algorithmFPType, CpuType cpu, typename NumericTableType>
class BlockHelperDense
{
public:
    BlockHelperDense(NumericTableType* nt, size_t dim, size_t iStartRow, size_t nRowsToProcess) : _dim(dim),
        _ntDataBD(nt, iStartRow, nRowsToProcess){}
    const Status& status() const { return _ntDataBD.status(); }

    void callGemm(const algorithmFPType* pCenters, size_t nRows, size_t nCenters, algorithmFPType* gemmResult)
    {
        char transa = 't';
        char transb = 'n';
        DAAL_INT _m = nCenters;
        DAAL_INT _n = nRows;
        DAAL_INT _k = _dim;
        algorithmFPType alpha = 1.0;
        DAAL_INT lda = _dim;
        DAAL_INT ldy = _dim;
        algorithmFPType beta = 0.0;
        DAAL_INT ldaty = nCenters;

        Blas<algorithmFPType, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, pCenters,
            &lda, _ntDataBD.get(), &ldy, &beta, gemmResult, &ldaty);
    }

    algorithmFPType getRowSumSq(size_t iRow, const algorithmFPType* cen)
    {
        const algorithmFPType* pData = _ntDataBD.get() + iRow*_dim;
        algorithmFPType norm2 = 0;
        for(size_t i = 0; i < _dim; ++i)
            norm2 += (pData[i] - cen[i])* (pData[i] - cen[i]);
        return norm2;
    }

    algorithmFPType getGemmResult(size_t iRow, size_t iCol, size_t nRows, size_t nCols, const algorithmFPType* gemmResult) const
    {
        return gemmResult[iRow*nCols + iCol];
    }

protected:
    ReadRows<algorithmFPType, cpu> _ntDataBD;
    const size_t _dim;
};

//BlockHelperCSR is the helper class for the CSR data type
template <typename algorithmFPType, CpuType cpu, typename NumericTableType>
class BlockHelperCSR
{
public:
    BlockHelperCSR(NumericTableType* nt, size_t dim, size_t iStartRow, size_t nRowsToProcess) : _dim(dim),
        _ntDataBD(nt, iStartRow, nRowsToProcess){}
    const Status& status() const { return _ntDataBD.status(); }

    void callGemm(const algorithmFPType* pCenters, size_t nRows, size_t nCenters, algorithmFPType* gemmResult)
    {
        char transa = 'n';
        DAAL_INT _n = nRows;
        DAAL_INT _p = _dim;
        DAAL_INT _c = nCenters;
        algorithmFPType alpha = 1.0;
        algorithmFPType beta = 0.0;
        DAAL_INT ldaty = nRows;
        char matdescra[6] = { 'G', 0, 0, 'F', 0, 0 };

        const algorithmFPType* pData = _ntDataBD.values();
        const size_t* colIdx = _ntDataBD.cols();
        const size_t* rowIdx = _ntDataBD.rows();

        SpBlas<algorithmFPType, cpu>::xxcsrmm(&transa, &_n, &_c, &_p, &alpha, matdescra,
            pData, (const DAAL_INT *)colIdx, (const DAAL_INT *)rowIdx,
            pCenters, &_p, &beta, gemmResult, &_n);
    }

    algorithmFPType getRowSumSq(size_t iRow, const algorithmFPType* cen)
    {
        const size_t* rowIdx = _ntDataBD.rows();
        const algorithmFPType* pData = _ntDataBD.values() + rowIdx[iRow] - 1;
        const size_t* colIdx = _ntDataBD.cols() + rowIdx[iRow] - 1;
        const size_t nValues = rowIdx[iRow + 1] - rowIdx[iRow];
        algorithmFPType res(0.);
        for(size_t i = 0; i < nValues; ++i)
            res += (pData[i] - cen[colIdx[i] - 1])*(pData[i] - cen[colIdx[i] - 1]);
        return res;
    }

    algorithmFPType getGemmResult(size_t iRow, size_t iCol, size_t nRows, size_t nCols, const algorithmFPType* gemmResult) const
    {
        return gemmResult[iRow + iCol*nRows];
    }

protected:
    ReadRowsCSR<algorithmFPType, cpu> _ntDataBD;
    const size_t _dim;
};

//DataHelperXXX template class is used by kmeans init tasks to hide data-specific manipulations behind general interface

//DataHelperDense is the helper class for the dense data type
template <typename algorithmFPType, CpuType cpu>
class DataHelperDense
{
public:
    typedef BlockHelperDense<algorithmFPType, cpu, NumericTable> BlockHelperType;

    DataHelperDense(NumericTable *ntData) : _nt(ntData), dim(ntData->getNumberOfColumns()), nRows(ntData->getNumberOfRows()){}
    NumericTable* nt() const { return _nt; }
    NumericTable* ntIface() const { return _nt; }

    Status updateMinDistInBlock(algorithmFPType& res, const algorithmFPType* aWeights, size_t iStartRow, size_t nRowsToProcess,
        const algorithmFPType* pLastAddedCenter, algorithmFPType* aMinDist) const
    {
        ReadRows<algorithmFPType, cpu> ntDataBD(nt(), iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS(ntDataBD);
        const algorithmFPType* pData = ntDataBD.get();
        algorithmFPType* pDistSq = aMinDist + iStartRow;
        algorithmFPType sumOfDist2 = 0;
        for(size_t iRow = 0; iRow < nRowsToProcess; ++iRow)
        {
            algorithmFPType dist2(0);
            for(size_t i = 0; i < dim; ++i)
                dist2 += (pData[iRow*dim + i] - pLastAddedCenter[i])*(pData[iRow*dim + i] - pLastAddedCenter[i]);
            if(aWeights)
                dist2 *= aWeights[iStartRow + iRow];
            if(pDistSq[iRow] > dist2)
                pDistSq[iRow] = dist2;
            sumOfDist2 += pDistSq[iRow];
        }
        res = sumOfDist2;
        return Status();
    }
    //copy one row from the given table to the destination buffer and return the sum of squares
    //of the data in this row
    algorithmFPType copyOneRowCalcSumSq(size_t iRow, algorithmFPType* pDst) const
    {
        ReadRows<algorithmFPType, cpu> ntDataBD(nt(), iRow, 1);
        const algorithmFPType* pData = ntDataBD.get();
        algorithmFPType res(0.);
        for(size_t i = 0; i < dim; ++i)
        {
            pDst[i] = pData[i];
            res += pData[i] * pData[i];
        }
        return res;
    }

public:
    const size_t dim;
    const size_t nRows;

protected:
    NumericTable* _nt;
};


//DataHelperCSR is the helper class for the CSR data type
template <typename algorithmFPType, CpuType cpu>
class DataHelperCSR
{
public:
    typedef BlockHelperCSR<algorithmFPType, cpu, CSRNumericTableIface> BlockHelperType;

    DataHelperCSR(NumericTable *ntData) : _nt(ntData), dim(ntData->getNumberOfColumns()), nRows(ntData->getNumberOfRows()),
        _csr(dynamic_cast<CSRNumericTableIface *>(ntData)){}
    NumericTable* nt() const { return _nt; }
    CSRNumericTableIface* ntIface() const { return _csr; }

    Status updateMinDistInBlock(algorithmFPType& res, const algorithmFPType* aWeights, size_t iStartRow, size_t nRowsToProcess,
        const algorithmFPType* pLastAddedCenter, algorithmFPType* aMinDist)
    {
        ReadRowsCSR<algorithmFPType, cpu> ntDataBD(_csr, iStartRow, nRowsToProcess);
        DAAL_CHECK_BLOCK_STATUS(ntDataBD);
        const algorithmFPType* pData = ntDataBD.values();
        const size_t* colIdx = ntDataBD.cols();
        const size_t* rowIdx = ntDataBD.rows();

        algorithmFPType* pDistSq = aMinDist + iStartRow;
        algorithmFPType sumOfDist2 = 0;
        size_t csrCursor = 0;
        for(size_t iRow = 0; iRow < nRowsToProcess; ++iRow)
        {
            algorithmFPType dist2(0.); //dist from iRow to the last added center
            const size_t nValues = rowIdx[iRow + 1] - rowIdx[iRow];
            for(size_t i = 0; i < nValues; ++i, ++csrCursor)
                dist2 += (pData[csrCursor] - pLastAddedCenter[colIdx[csrCursor] - 1])*(pData[csrCursor] - pLastAddedCenter[colIdx[csrCursor] - 1]);
            if(aWeights)
                dist2 *= aWeights[iStartRow + iRow];
            if(pDistSq[iRow] > dist2)
                pDistSq[iRow] = dist2;
            sumOfDist2 += pDistSq[iRow];
        }
        res = sumOfDist2;
        return Status();
    }
    //copy one row from the given table to the destination buffer and return the sum of squares
    //of the data in this row
    algorithmFPType copyOneRowCalcSumSq(size_t iRow, algorithmFPType* pDst) const
    {
        ReadRowsCSR<algorithmFPType, cpu> ntDataBD(_csr, iRow, 1);
        const algorithmFPType* pData = ntDataBD.values();
        const size_t* colIdx = ntDataBD.cols();
        const size_t* rowIdx = ntDataBD.rows();

        daal::services::internal::service_memset<algorithmFPType, cpu>(pDst, algorithmFPType(0.), dim);
        algorithmFPType res(0.);
        const size_t nValues = rowIdx[1] - rowIdx[0];
        for(size_t i = 0; i < nValues; ++i, ++pData, ++colIdx)
        {
            res += (*pData) * (*pData);
            pDst[(*colIdx) - 1] = *pData;
        }
        return res;
    }

public:
    const size_t dim;
    const size_t nRows;

protected:
    NumericTable* _nt;
    CSRNumericTableIface* _csr;
};

//Base task class for kmeans++ and kmeans||
template <typename algorithmFPType, CpuType cpu, typename DataHelper>
class TaskPlusPlusBatchBase
{
public:
    TaskPlusPlusBatchBase(NumericTable *ntData, NumericTable *ntClusters, size_t numClusters, engines::BatchBase &engine) :
        _data(ntData), _ntClusters(ntClusters),
        _nClusters(numClusters), _engine(engine),
        _aMinDist(ntData->getNumberOfRows()),
        _overallError(daal::services::internal::MaxVal<algorithmFPType>::get()),
        _lastAddedCenterSumSq(0.)
    {
        _nBlocks = _data.nRows / _nRowsInBlock;
        _nBlocks += (_nBlocks * _nRowsInBlock != _data.nRows);
        _aMinDistAcc.reset(_nBlocks);
    }

protected:
    //copy _dim*nPt algorithmFPType values from pSrc to pDst
    void copyPoints(algorithmFPType* pDst, const algorithmFPType* pSrc, size_t nPt) const
    {
        daal::services::daal_memcpy_s(pDst, sizeof(algorithmFPType)*_data.dim*nPt, pSrc, sizeof(algorithmFPType)*_data.dim*nPt);
    }
    //get first center at random
    size_t calcFirstCenter();

    //find a row corresponding to the sample
    size_t findSample(algorithmFPType sample);

    //recalculate overall error as a sum of error values per each block
    void calcOverallError()
    {
        const algorithmFPType* aMinDistAcc = _aMinDistAcc.get();
        _overallError = aMinDistAcc[0];
        for(size_t iBlock = 1; iBlock < _nBlocks; ++iBlock)
            _overallError += aMinDistAcc[iBlock];
    }

    //update minimal distance using last added center
    Status updateMinDist(const algorithmFPType* aWeights);

    //current value of overall error (goal function)
    algorithmFPType overallError() const { return _overallError; }

    //generate _aProbability array data
    Status generateProbabilities(size_t iStart, size_t nTotal)
    {
        return UniformKernelDefault<algorithmFPType, cpu>::compute(algorithmFPType(0.), algorithmFPType(1.), _engine, nTotal, _aProbability.get() + iStart);
    }

protected:
    DataHelper _data;
    NumericTable* _ntClusters;
    const size_t _nClusters;
    engines::BatchBase &_engine;
    size_t _nBlocks;
    TArray<algorithmFPType, cpu> _lastAddedCenter; //center last added to the clusters
    TArray<algorithmFPType, cpu> _aMinDist; //distance to the nearest cluster for every point
    TArray<algorithmFPType, cpu> _aMinDistAcc; //accumulated aMinDist per every block
    algorithmFPType _lastAddedCenterSumSq; //sum of squares of last added center
    algorithmFPType _overallError; //current value of overall error (goal function)
    TArray<algorithmFPType, cpu> _aProbability; //array of probabilities for all candidates
};

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
class TaskPlusPlusBatch : public TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper>
{
public:
    typedef TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper> super;
    TaskPlusPlusBatch(NumericTable *ntData, const algorithmFPType* aWeight,
        NumericTable *ntClusters, size_t numClusters, engines::BatchBase &engine) :
        super(ntData, ntClusters, numClusters, engine), _aWeight(aWeight)
    {
        this->_lastAddedCenter.reset(this->_data.dim); //reserve memory for a single point only
        this->_aProbability.reset(numClusters); //reserve memory for all candidates
    }
    Status run();

protected:
    size_t calcCenter(size_t iCluster);
    //sample a point with the probability proportional to its contribution to the overall error,
    //return index of the point
    size_t samplePoint(size_t iCluster);

protected:
    const algorithmFPType* _aWeight;
};

template <typename algorithmFPType, CpuType cpu>
struct TlsPPData
{
    algorithmFPType* gemmResult; //result of gemm call is placed here
    algorithmFPType accMinDist2; //goal function accumulated for all blocks processed by a thread
    int aCandidateRating[1]; //rating of candidates updated in all blocks processed by a thread
};

//TaskParallelPlusUpdateDist class is used by kmeans init batch and distributed kernel
template <typename algorithmFPType, CpuType cpu, typename DataHelper>
class TaskParallelPlusUpdateDist
{
protected:
    typedef TlsPPData<algorithmFPType, cpu> TlsPPData_t;

public:
    TaskParallelPlusUpdateDist(size_t nBlocks,
        int* aCandidateRating,
        int* aNearestCandidateIdx,
        algorithmFPType& overallError, DataHelper& data,
        const algorithmFPType* lastAddedCenters,
        algorithmFPType* lastAddedCenterNorm2,
        algorithmFPType* aMinDist,
        algorithmFPType* aMinDistAcc) :
        _nBlocks(nBlocks), _aCandidateRating(aCandidateRating), _aNearestCandidateIdx(aNearestCandidateIdx),
        _overallError(overallError), _data(data),
        _lastAddedCenter(lastAddedCenters),
        _lastAddedCenterNorm2(lastAddedCenterNorm2),
        _aMinDist(aMinDist), _aMinDistAcc(aMinDistAcc){}
    Status updateMinDist(size_t iFirstOfNewCandidates, size_t nNewCandidates);

protected:
    Status processBlock(size_t iBlock, TlsPPData_t* tlsLocal, size_t iFirstOfNewCandidates, size_t nNewCandidates);
    bool findBestCandidate(typename DataHelper::BlockHelperType& blockHelper, size_t iRow, algorithmFPType* pDistSq, size_t nRowsToProcess,
        size_t nNewCandidates, size_t& iBestCandidate, const algorithmFPType* gemmResult) const;

protected:
    size_t _nBlocks;
    int* _aCandidateRating;
    int* _aNearestCandidateIdx;
    algorithmFPType& _overallError;
    DataHelper& _data;
    const algorithmFPType* _lastAddedCenter;
    algorithmFPType* _lastAddedCenterNorm2;
    algorithmFPType* _aMinDist;
    algorithmFPType* _aMinDistAcc;
};

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
class TaskParallelPlusBatch : public TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper>
{
public:
    typedef TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper> super;
    TaskParallelPlusBatch(NumericTable *ntData, NumericTable *ntClusters, const Parameter *par, engines::BatchBase &engine) :
        super(ntData, ntClusters, par->nClusters, engine),
        _nCandidates(0), _aNearestCandidateIdx(this->_data.nRows),
        _L(par->oversamplingFactor*par->nClusters), _R(par->nRounds)
    {
        this->_lastAddedCenter.reset(this->_data.dim*_L);//reserve memory for L points
        _lastAddedCenterNorm2.reset(_L);//reserve memory for L values
        _aNearestCandidateIdx.reset(this->_data.nRows);
    }
    Status run();

private:
    typedef services::SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > HomogenNumericTableCPUPtr;

private:
    Status updateMinDist(size_t iFirstOfNewCandidates, size_t nNewCandidates);
    size_t calcCenters(size_t nRequired, size_t* aCenters, size_t iRound);
    //sample nPt points with probability proportional to their contribution to the overall error,
    //put result to aPt
    size_t samplePoints(size_t nPt, size_t* aPt, size_t iRound);
    Status getCandidates(HomogenNumericTableCPUPtr& pCandidates);

private:
    const size_t _L;
    const size_t _R;
    //Note: maxNumberOfCandidates = _L*_R + 1;
    size_t _nCandidates; //number of candidates found so far (from 0 to maxNumberOfCandidates)
    TArray<size_t, cpu> _aCandidateIdx; //array[maxNumberOfCandidates], contains row indices of the candidates added so far
    TArray<int, cpu> _aCandidateRating; //array[maxNumberOfCandidates], number of points closest to each candidate found so far
    TArray<int, cpu> _aNearestCandidateIdx; //index of the nearest candidate in _aCandidateIdx per each point
    TArray<algorithmFPType, cpu> _lastAddedCenterNorm2; //array[L] contains 0.5*(center, center) for each last added center
};

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskPlusPlusBatch<algorithmFPType, cpu, DataHelper>::run()
{
    DAAL_CHECK(this->_aMinDist.get() && this->_aMinDistAcc.get() && this->_lastAddedCenter.get() && this->_aProbability.get(),
        ErrorMemoryAllocationFailed);
    WriteOnlyRows<algorithmFPType, cpu> clustersBD(this->_ntClusters, 0, this->_nClusters);
    DAAL_CHECK_BLOCK_STATUS(clustersBD);
    algorithmFPType *clusters = clustersBD.get();
    daal::services::internal::service_memset<algorithmFPType, cpu>(this->_aMinDist.get(),
        daal::services::internal::MaxVal<algorithmFPType>::get(), this->_data.nRows);
    this->generateProbabilities(0, this->_nClusters);

    //get first center at random
    size_t iCenter = this->calcFirstCenter();
    //copy it to the result
    this->copyPoints(clustersBD.get(), this->_lastAddedCenter.get(), 1);

    //get other centers
    for(size_t iCluster = 1; iCluster < this->_nClusters; ++iCluster)
    {
        size_t iCenter = calcCenter(iCluster);
        //copy it to the result
        this->copyPoints(clustersBD.get() + iCluster*this->_data.dim, this->_lastAddedCenter.get(), 1);
    }
    return Status();
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper>::calcFirstCenter()
{
    //use first element of probabilities array to sample a new center
    algorithmFPType prob(this->_aProbability.get()[0]);
    size_t iRow(prob*_data.nRows);
    if(iRow == _data.nRows) //round-off error
        --iRow;
    _lastAddedCenterSumSq = this->_data.copyOneRowCalcSumSq(iRow, _lastAddedCenter.get());
    _aMinDist.get()[iRow] = 0;
    return iRow;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper>::findSample(algorithmFPType sample)
{
    const algorithmFPType* aMinDistAcc = _aMinDistAcc.get();
    algorithmFPType* aMinDist = _aMinDist.get();
    //find the block this sample belongs to
    size_t iBlock = 0;
    for(; (iBlock + 1 < _nBlocks) && (sample >= aMinDistAcc[iBlock]); ++iBlock)
        sample -= aMinDistAcc[iBlock];

    //find the row in the block corresponding to the sample
    size_t nRowsToProcess = _nRowsInBlock;
    if(iBlock == _nBlocks - 1)
        nRowsToProcess = _data.nRows - iBlock * _nRowsInBlock;
    const size_t iStartRow = iBlock*_nRowsInBlock;
    size_t iRow = 0;
    for(; (iRow + 1 < nRowsToProcess) && (sample >= aMinDist[iStartRow + iRow]); ++iRow)
        sample -= aMinDist[iStartRow + iRow];
    return iStartRow + iRow;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskPlusPlusBatch<algorithmFPType, cpu, DataHelper>::samplePoint(size_t iCluster)
{
    if(this->overallError() > 0)
    {
        const algorithmFPType eps = algorithmFPType(0.1)*this->overallError() / algorithmFPType(this->_data.nRows);
        const algorithmFPType* aMinDistAcc = this->_aMinDistAcc.get();
        algorithmFPType* aMinDist = this->_aMinDist.get();
        //take a pre-computed probability value
        algorithmFPType probability = this->_aProbability.get()[iCluster];
        do
        {
            size_t iRow = this->findSample(this->overallError()*probability);
            if(aMinDist[iRow] > eps)
            {
                aMinDist[iRow] = 0;
                return iRow;
            }
            //already taken or duplicate point, sample again
            UniformKernelDefault<algorithmFPType, cpu>::compute(algorithmFPType(0.), algorithmFPType(1.), this->_engine, 1, &probability);
        }
        while(true);
    }
    return 0;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskPlusPlusBatchBase<algorithmFPType, cpu, DataHelper>::updateMinDist(const algorithmFPType* aWeights)
{
    SafeStatus safeStat;
    daal::threader_for(_nBlocks, _nBlocks, [=, &safeStat](size_t iBlock)
    {
        safeStat |= _data.updateMinDistInBlock(_aMinDistAcc.get()[iBlock], aWeights,
            iBlock*_nRowsInBlock,//start row
            (iBlock == _nBlocks - 1) ? _data.nRows - iBlock * _nRowsInBlock : _nRowsInBlock, //rows to process
            _lastAddedCenter.get(),
            _aMinDist.get());
    });
    if(safeStat)
    calcOverallError();
    return safeStat.detach();
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskPlusPlusBatch<algorithmFPType, cpu, DataHelper>::calcCenter(size_t iCluster)
{
    this->updateMinDist(_aWeight);
    const size_t iRow = samplePoint(iCluster);
    this->_lastAddedCenterSumSq = this->_data.copyOneRowCalcSumSq(iRow, this->_lastAddedCenter.get());
    return iRow;
}

template <typename algorithmFPType, CpuType cpu>
services::Status KMeansinitKernel<plusPlusDense, algorithmFPType, cpu>::compute(size_t na, const NumericTable *const *a,
    size_t nr, const NumericTable *const *r, const Parameter *par, engines::BatchBase &engine)
{
    TaskPlusPlusBatch<algorithmFPType, cpu, DataHelperDense<algorithmFPType, cpu> > task(
        const_cast<NumericTable *>(a[0]), //data
        nullptr,
        const_cast<NumericTable *>(r[0]), //clusters
        par->nClusters, engine);
    return task.run();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KMeansinitKernel<plusPlusCSR, algorithmFPType, cpu>::compute(size_t na, const NumericTable *const *a,
    size_t nr, const NumericTable *const *r, const Parameter *par, engines::BatchBase &engine)
{
    TaskPlusPlusBatch<algorithmFPType, cpu, DataHelperCSR<algorithmFPType, cpu> > task(
        const_cast<NumericTable *>(a[0]), //data
        nullptr,
        const_cast<NumericTable *>(r[0]), //clusters
        par->nClusters, engine);
    return task.run();
}

//////////////////////// TaskParallelPlusBatch ///////////////////////////////////
template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskParallelPlusBatch<algorithmFPType, cpu, DataHelper>::samplePoints(size_t nPt, size_t* aPt, size_t iRound)
{
    if(iRound >= _R)
    {
        //generate extra values in _aProbability
        this->generateProbabilities(_nCandidates, nPt);
    }
    //sample each point independently
    daal::threader_for(nPt, nPt, [=](size_t iPt)
    {
        const size_t iCandidate = _nCandidates + iPt;
        algorithmFPType probability = this->_aProbability.get()[iCandidate];
        aPt[iPt] = this->findSample(this->overallError()*probability);
    });

    const algorithmFPType eps = algorithmFPType(0.1)*this->overallError() / algorithmFPType(this->_data.nRows);
    size_t iNewCandidate = 0;
    //update ratings and check for the duplicates
    algorithmFPType* aMinDist = this->_aMinDist.get();
    for(size_t iPt = 0; iPt < nPt; ++iPt)
    {
        const size_t iRow = aPt[iPt];
        if(aMinDist[iRow] > eps)
        {
            aMinDist[iRow] = 0;
            auto pNearestCandIndex = _aNearestCandidateIdx.get();
            const auto iPrevCandidate = pNearestCandIndex[iRow];
            pNearestCandIndex[iRow] = _nCandidates + iNewCandidate; //this point becomes an (_nCandidates + iNewCandidate)-th candidate, increases own rating
            _aCandidateRating.get()[iPrevCandidate] -= 1;
            _aCandidateRating.get()[_nCandidates + iNewCandidate] += 1;
            aPt[iNewCandidate] = iRow;
            ++iNewCandidate;
        }
    }
    return iNewCandidate;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
bool TaskParallelPlusUpdateDist<algorithmFPType, cpu, DataHelper>::findBestCandidate(typename DataHelper::BlockHelperType& blockHelper, size_t iRow,
    algorithmFPType* pDistSq, size_t nRowsToProcess, size_t nNewCandidates, size_t& iBestCandidate, const algorithmFPType* gemmResult) const
{
    size_t iCandidate = 0;
    iBestCandidate = iCandidate;
    algorithmFPType valBest = _lastAddedCenterNorm2[iCandidate] - blockHelper.getGemmResult(iRow, iCandidate, nRowsToProcess, nNewCandidates, gemmResult);
    for(size_t iCandidate = 1; iCandidate < nNewCandidates; ++iCandidate)
    {
        algorithmFPType valCand = _lastAddedCenterNorm2[iCandidate] - blockHelper.getGemmResult(iRow, iCandidate, nRowsToProcess, nNewCandidates, gemmResult);
        if(valBest > valCand)
        {
            valBest = valCand;
            iBestCandidate = iCandidate;
        }
    }
    const algorithmFPType* pLastAddedCenter = _lastAddedCenter;
    const algorithmFPType dist2 = blockHelper.getRowSumSq(iRow, pLastAddedCenter + iBestCandidate*this->_data.dim);
    if(dist2 < pDistSq[iRow])
    {
        pDistSq[iRow] = dist2;
        return true;
    }
    return false;
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskParallelPlusUpdateDist<algorithmFPType, cpu, DataHelper>::processBlock(size_t iBlock, TlsPPData_t* tlsLocal,
    size_t iFirstOfNewCandidates, size_t nNewCandidates)
{
    int* aCandidateRating = tlsLocal->aCandidateRating;
    size_t nRowsToProcess = _nRowsInBlock;
    if(iBlock == _nBlocks - 1)
        nRowsToProcess = _data.nRows - iBlock * _nRowsInBlock;
    const size_t iStartRow = iBlock*_nRowsInBlock;

    typename DataHelper::BlockHelperType blockHelper(_data.ntIface(), _data.dim, iStartRow, nRowsToProcess);
    DAAL_CHECK_BLOCK_STATUS(blockHelper);
    blockHelper.callGemm(_lastAddedCenter, nRowsToProcess, nNewCandidates, tlsLocal->gemmResult);

    algorithmFPType* pDistSq = _aMinDist + iStartRow;
    auto* pNearestCandIndex = _aNearestCandidateIdx + iStartRow;
    algorithmFPType sumOfDist2 = 0;
    for(size_t iRow = 0; iRow < nRowsToProcess; ++iRow)
    {
        size_t iBestCandidate = 0;
        if(findBestCandidate(blockHelper, iRow, pDistSq, nRowsToProcess, nNewCandidates, iBestCandidate, tlsLocal->gemmResult))
        {
            const auto iPrevCandidate = pNearestCandIndex[iRow];
            pNearestCandIndex[iRow] = iFirstOfNewCandidates + iBestCandidate;
            aCandidateRating[iPrevCandidate] -= 1;
            aCandidateRating[iFirstOfNewCandidates + iBestCandidate] += 1;
        }
        sumOfDist2 += pDistSq[iRow];
    }
    _aMinDistAcc[iBlock] = sumOfDist2;
    tlsLocal->accMinDist2 += sumOfDist2;
    return Status();
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskParallelPlusUpdateDist<algorithmFPType, cpu, DataHelper>::updateMinDist(size_t iFirstOfNewCandidates, size_t nNewCandidates)
{
    const size_t nCandidates = iFirstOfNewCandidates + nNewCandidates;
    const size_t gemmDataSize = _nRowsInBlock*nCandidates;
    daal::tls<TlsPPData_t *> tlsData([=]()-> TlsPPData_t*
    {
        const size_t sz = sizeof(TlsPPData_t)+(nCandidates - 1)*sizeof(int);
        byte* ptr = service_scalable_calloc<byte, cpu>(sz);
        TlsPPData_t* pData =  new (ptr)TlsPPData_t;
        //allocate memory for Intel(R) MKL result
        if(pData)
        {
            pData->gemmResult = service_calloc<algorithmFPType, cpu>(gemmDataSize);
            if(!pData->gemmResult)
            {
                service_scalable_free<byte, cpu>(ptr);
                return nullptr;
            }
        }
        return pData;
    });
    bool bMemoryAllocationFailed = false;
    algorithmFPType newOverallError(0.);
    SafeStatus safeStat;
    daal::threader_for(this->_nBlocks, this->_nBlocks, [=, &tlsData, &bMemoryAllocationFailed, &safeStat](size_t iBlock)
    {
        TlsPPData_t* tlsLocal = tlsData.local();
        if(!tlsLocal)
        {
            bMemoryAllocationFailed = true;
            return;
        }
        safeStat |= processBlock(iBlock, tlsLocal, iFirstOfNewCandidates, nNewCandidates);
    });
    tlsData.reduce([=, &newOverallError](TlsPPData_t* ptr)-> void
    {
        if(!ptr)
            return;
        newOverallError += ptr->accMinDist2;
        for(size_t j = 0; j < nCandidates; ++j)
            _aCandidateRating[j] += ptr->aCandidateRating[j];
        service_free<algorithmFPType, cpu>(ptr->gemmResult);
        service_scalable_free<byte, cpu>((byte*)ptr);
    });
    this->_overallError = newOverallError;
    if(!safeStat)
        return safeStat.detach();
    return bMemoryAllocationFailed ? Status(ErrorMemoryAllocationFailed) : Status();
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskParallelPlusBatch<algorithmFPType, cpu, DataHelper>::updateMinDist(size_t iFirstOfNewCandidates, size_t nNewCandidates)
{
    TaskParallelPlusUpdateDist<algorithmFPType, cpu, DataHelper> impl(this->_nBlocks,
        _aCandidateRating.get(),
        _aNearestCandidateIdx.get(),
        this->_overallError, this->_data,
        this->_lastAddedCenter.get(),
        this->_lastAddedCenterNorm2.get(),
        this->_aMinDist.get(), this->_aMinDistAcc.get());
    return impl.updateMinDist(iFirstOfNewCandidates, nNewCandidates);
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskParallelPlusBatch<algorithmFPType, cpu, DataHelper>::getCandidates(HomogenNumericTableCPUPtr& pCandidates)
{
    const size_t maxNumberOfCandidates = _L*_R + 1;
    //reserve memory and init work variables for all candidates
    _aCandidateIdx.reset(maxNumberOfCandidates);
    _aCandidateRating.reset(maxNumberOfCandidates);
    this->_aProbability.reset(maxNumberOfCandidates);
    _aNearestCandidateIdx.reset(this->_data.nRows);
    DAAL_CHECK(_aCandidateIdx.get() && _aCandidateRating.get() && _aNearestCandidateIdx.get() && this->_aProbability.get(),
        ErrorMemoryAllocationFailed);

    this->generateProbabilities(0, maxNumberOfCandidates);

    //get first candidate at random
    auto iCenter = this->calcFirstCenter();
    this->_lastAddedCenterNorm2.get()[0] = algorithmFPType(0.5)*this->_lastAddedCenterSumSq;
    super::updateMinDist(nullptr);

    Status s;
    //create data for candidates output
    pCandidates = HomogenNumericTableCPU<algorithmFPType, cpu>::create(this->_data.dim, maxNumberOfCandidates, &s);
    DAAL_CHECK_STATUS_VAR(s);
    {
        WriteOnlyRows<algorithmFPType, cpu> candidatesBD(pCandidates.get(), 0, pCandidates->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(candidatesBD);

        //no ratings yet
        daal::services::internal::service_memset<int, cpu>(_aCandidateRating.get(), 0, maxNumberOfCandidates);

        //add first candidate
        _aCandidateIdx.get()[0] = iCenter;
        _nCandidates = 1;
        this->copyPoints(candidatesBD.get(), this->_lastAddedCenter.get(), 1);
        //it is nearest for all points
        daal::services::internal::service_memset<int, cpu>(_aNearestCandidateIdx.get(), 0, this->_data.nRows);
        //hence its rating is highest so far
        _aCandidateRating.get()[0] = this->_data.nRows;

        //get other candidates in R rounds
        bool bDone = false;
        for(size_t iRound = 0; !bDone; ++iRound)
        {
            //calculate candidates: _L or min(_L, what remains);
            const size_t nRequired = (iRound < _R || _L < (maxNumberOfCandidates - _nCandidates)) ? _L : (maxNumberOfCandidates - _nCandidates);
            const size_t nNewCandidates = calcCenters(nRequired, _aCandidateIdx.get() + _nCandidates, iRound);
            if(nNewCandidates)
            {
                //copy them to the candidates table
                this->copyPoints(candidatesBD.get() + _nCandidates*this->_data.dim, this->_lastAddedCenter.get(), nNewCandidates);
                const size_t iFirstNewCandidate = _nCandidates;
                _nCandidates += nNewCandidates;
                bDone = ((iRound + 1 >= _R) && (_nCandidates > this->_nClusters));
                DAAL_CHECK_STATUS(s, this->updateMinDist(iFirstNewCandidate, nNewCandidates));
            }
        }
    }
    if(_nCandidates < maxNumberOfCandidates)
        pCandidates->resize(_nCandidates);
    return Status();
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
Status TaskParallelPlusBatch<algorithmFPType, cpu, DataHelper>::run()
{
    DAAL_CHECK(this->_aMinDist.get() && this->_aMinDistAcc.get() && this->_lastAddedCenter.get() && _lastAddedCenterNorm2.get(),
        ErrorMemoryAllocationFailed);
    daal::services::internal::service_memset<algorithmFPType, cpu>(this->_aMinDist.get(),
        daal::services::internal::MaxVal<algorithmFPType>::get(), this->_data.nRows);
    HomogenNumericTableCPUPtr pCandidates;
    Status s = getCandidates(pCandidates);
    if(!s)
        return s;
    const auto nCandidates = pCandidates->getNumberOfRows();

    TArray<algorithmFPType, cpu> aWeight(nCandidates);
    const algorithmFPType div(1. / algorithmFPType(this->_data.nRows));
    for(auto i = 0; i < nCandidates; ++i)
        aWeight.get()[i] = div*algorithmFPType(_aCandidateRating.get()[i]);
    TaskPlusPlusBatch<algorithmFPType, cpu, DataHelperDense<algorithmFPType, cpu> > task(
        pCandidates.get(), aWeight.get(), this->_ntClusters, this->_nClusters, this->_engine);
    return task.run();
}

template <typename algorithmFPType, CpuType cpu, typename DataHelper>
size_t TaskParallelPlusBatch<algorithmFPType, cpu, DataHelper>::calcCenters(size_t nRequired, size_t* aCenters, size_t iRound)
{
    const size_t nPt = samplePoints(nRequired, aCenters, iRound);
    //copy points in parallel
    daal::threader_for(nPt, nPt, [=](size_t iPt)
    {
        const size_t iRow = aCenters[iPt];
        this->_lastAddedCenterNorm2.get()[iPt] = algorithmFPType(0.5)*this->_data.copyOneRowCalcSumSq(iRow,
            this->_lastAddedCenter.get() + iPt*this->_data.dim);
    });
    return nPt;
}

template <typename algorithmFPType, CpuType cpu>
services::Status KMeansinitKernel<parallelPlusDense, algorithmFPType, cpu>::compute(size_t na, const NumericTable *const *a,
    size_t nr, const NumericTable *const *r, const Parameter *par, engines::BatchBase &engine)
{
    TaskParallelPlusBatch<algorithmFPType, cpu, DataHelperDense<algorithmFPType, cpu> > task(
        const_cast<NumericTable *>(a[0]), //data
        const_cast<NumericTable *>(r[0]), //clusters
        par, engine);
    return task.run();
}

template <typename algorithmFPType, CpuType cpu>
services::Status KMeansinitKernel<parallelPlusCSR, algorithmFPType, cpu>::compute(size_t na, const NumericTable *const *a,
    size_t nr, const NumericTable *const *r, const Parameter *par, engines::BatchBase &engine)
{
    TaskParallelPlusBatch<algorithmFPType, cpu, DataHelperCSR<algorithmFPType, cpu> > task(
        const_cast<NumericTable *>(a[0]), //data
        const_cast<NumericTable *>(r[0]), //clusters
        par, engine);
    return task.run();
}

} // namespace daal::algorithms::kmeans::init::internal
} // namespace daal::algorithms::kmeans::init
} // namespace daal::algorithms::kmeans
} // namespace daal::algorithms
} // namespace daal
