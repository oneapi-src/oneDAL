/* file: svm_train_boser_impl.i */
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
//  SVM training algorithm implementation
//--
*/
/*
//  DESCRIPTION
//
//  Definition of the functions for training with SVM 2-class classifier.
//
//  REFERENCES
//
//  1. Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin,
//     Working Set Selection Using Second Order Information
//     for Training Support Vector Machines,
//     Journal of Machine Learning Research 6 (2005), pp. 1889___1918
//  2. Bernard E. boser, Isabelle M. Guyon, Vladimir N. Vapnik,
//     A Training Algorithm for Optimal Margin Classifiers.
//  3. Thorsten Joachims, Making Large-Scale SVM Learning Practical,
//     Advances in Kernel Methods - Support Vector Learning
*/

#ifndef __SVM_TRAIN_BOSER_IMPL_I__
#define __SVM_TRAIN_BOSER_IMPL_I__

#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"
#include "service_utils.h"
#include "service_data_utils.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::data_feature_utils::internal;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{

template <typename algorithmFPType, CpuType cpu>
void SVMTrainImpl<boser, algorithmFPType, cpu>::compute(NumericTablePtr xTable, NumericTable *yTable, daal::algorithms::Model *r,
            const daal::algorithms::Parameter *par)
{
    size_t nFeatures = xTable->getNumberOfColumns();
    size_t nVectors  = xTable->getNumberOfRows();
    Parameter *svmPar = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(par));
    algorithmFPType C    = (algorithmFPType)(svmPar->C);
    algorithmFPType eps  = (algorithmFPType)(svmPar->accuracyThreshold);
    algorithmFPType tau  = (algorithmFPType)(svmPar->tau);
    size_t cacheSize     = svmPar->cacheSize;
    size_t maxIter       = svmPar->maxIterations;
    bool doShrinking     = svmPar->doShrinking;
    size_t shrinkingStep = svmPar->shrinkingStep;
    bool unshrink = false;

    Model *model = static_cast<Model *>(r);
    model->setNFeatures(nFeatures);
    services::SharedPtr<kernel_function::KernelIface> kernel = svmPar->kernel->clone();

    /* Allocate memory for storing intermediate results */
    SVMTrainTask<algorithmFPType, cpu> task(cacheSize, nVectors, kernelFunctionBlockSize, doShrinking,
                                            xTable, yTable, kernel, this->_errors);
    if (this->_errors->size() != 0) { return; }

    task.init(C);
    if (this->_errors->size() != 0) { return; }

    size_t nActiveVectors = nVectors;

    /* Perform Sequential Minimum Optimization (SMO) algorithm
       to find optimal coefficients alpha */
    algorithmFPType curEps = MaxVal<algorithmFPType, cpu>::get();

    algorithmFPType *y     = task.y;
    algorithmFPType *alpha = task.alpha;
    algorithmFPType *grad  = task.grad;
    algorithmFPType *kernelDiag = task.kernelDiag;
    char *I = task.I;

    size_t shrinkingIter = 1;
    size_t iter = 0;

    if (doShrinking)
    {
        for (; iter < maxIter && eps < curEps; iter++, shrinkingIter++)
        {
            int Bi, Bj;
            algorithmFPType delta, ma, Ma;

            if (!findMaximumViolatingPair(nActiveVectors, tau, y, grad, kernelDiag, I, task.cache,
                                          &Bi, &Bj, &delta, &ma, &Ma, &curEps))
            { break; }

            if (curEps < eps)
            {
                /* Check the optimality condition for the task with excluded variables */
                if (unshrink && nActiveVectors < nVectors)
                {
                    nActiveVectors = reconstructGradient(nVectors, nActiveVectors, task.cache, y, alpha, grad);
                }

                if (!findMaximumViolatingPair(nActiveVectors, tau, y, grad, kernelDiag, I, task.cache,
                                              &Bi, &Bj, &delta, &ma, &Ma, &curEps))
                { break; }

                if (curEps < eps)
                {
                    /* Here if the optimality condition holds for the excluded variables */
                    break;
                }

                shrinkingIter = 0;
            }

            updateTask(nActiveVectors, C, Bi, Bj, delta, y, alpha, grad, task);

            if ((shrinkingIter % shrinkingStep) == 0)
            {
                if (!unshrink && curEps < 10.0 * eps)
                {
                    unshrink = true;
                    if (nActiveVectors < nVectors)
                    {
                        nActiveVectors = reconstructGradient(nVectors, nActiveVectors, task.cache, y, alpha, grad);
                    }
                }

                /* Update shrinking flags */
                size_t nShrink = updateShrinkingFlags(nActiveVectors, C, ma, Ma, y, alpha, grad, I);

                /* Do shrinking */
                if (nShrink > 0)
                {
                    task.cache->updateShrinkingRowIndices(nActiveVectors, I);
                    nActiveVectors = shrinkTask(nActiveVectors, y, alpha, grad, kernelDiag, I);
                }
            }
        }

        if (nActiveVectors < nVectors)
        {
            nActiveVectors = reconstructGradient(nVectors, nActiveVectors, task.cache, y, alpha, grad);
        }
    }
    else
    {
        for (; iter < maxIter && eps < curEps; iter++, shrinkingIter++)
        {
            int Bi, Bj;
            algorithmFPType delta, ma, Ma;

            if (!findMaximumViolatingPair(nActiveVectors, tau, y, grad, kernelDiag, I, task.cache,
                                          &Bi, &Bj, &delta, &ma, &Ma, &curEps))
            { break; }
            updateTask(nActiveVectors, C, Bi, Bj, delta, y, alpha, grad, task);
        }
    }

    /* Write support vectors and classification coefficients into model */
    size_t nSV = computeNumberOfSV(nVectors, alpha);
    setSVCoefficients(nVectors, nSV, y, alpha, model);
    if (xTable->getDataLayout() == NumericTableIface::csrArray)
    {
        CSRBlockMicroTable<algorithmFPType, readOnly, cpu> mtX(xTable.get());
        setSV(model, mtX, nFeatures, nVectors, nSV, alpha, task.cache);
    }
    else
    {
        BlockMicroTable<algorithmFPType, readOnly, cpu> mtX(xTable.get());
        setSV(model, mtX, nFeatures, nVectors, nSV, alpha, task.cache);
    }

    /* Calculate bias and write it into model */
    algorithmFPType bias = calculateBias(C, nVectors, y, alpha, grad);
    model->setBias((double)bias);
}

/**
 * \brief Compute number of support vectors found by SVM
 *
 * \param[in]  nVectors     Number of observations in the input data set
 * \param[in]  alpha        Array of classification coefficients
 * \return Number of support vectors
 */
template <typename algorithmFPType, CpuType cpu>
size_t SVMTrainImpl<boser, algorithmFPType, cpu>::computeNumberOfSV(
            size_t nVectors, const algorithmFPType *alpha)
{
    const algorithmFPType zero = (algorithmFPType)0.0;
    size_t nSV = 0;
    for (size_t i = 0; i < nVectors; i++)
    {
        if (alpha[i] > zero)
        {
            nSV++;
        }
    }
    return nSV;
}

/**
 * \brief Write classification coefficients into resulting model
 *
 * \param[in]  nVectors     Number of observations in the input data set
 * \param[in]  nSV          Number of support vectors
 * \param[in]  y            Array of class labels
 * \param[in]  alpha        Array of classification coefficients
 * \param[out] model        Resulting model
 */
template <typename algorithmFPType, CpuType cpu>
void SVMTrainImpl<boser, algorithmFPType, cpu>::setSVCoefficients(
            size_t nVectors, size_t nSV, const algorithmFPType *y, const algorithmFPType *alpha,
            Model *model)
{
    const algorithmFPType zero = (algorithmFPType)0.0;
    NumericTablePtr svCoeffTable = model->getClassificationCoefficients();
    svCoeffTable->setNumberOfRows(nSV);
    svCoeffTable->allocateDataMemory();

    algorithmFPType *svCoeff;
    BlockMicroTable<algorithmFPType, writeOnly, cpu> mtSvCoeff(svCoeffTable.get());
    mtSvCoeff.getBlockOfRows(0, nSV, &svCoeff);

    for (size_t i = 0, iSV = 0; i < nVectors; i++)
    {
        if (alpha[i] != zero)
        {
            svCoeff[iSV] = y[i] * alpha[i];
            iSV++;
        }
    }
    mtSvCoeff.release();
}

/**
 * \brief Write support vectors in dense format into resulting model
 *
 * \param[out] model        Resulting model
 * \param[in]  mtX          Micro table with input data set in dense layout
 * \param[in]  nFeatures    Number of features in the input data set
 * \param[in]  nVectors     Number of observations in the input data set
 * \param[in]  nSV          Number of support vectors
 * \param[in]  alpha        Array of classification coefficients
 * \param[in]  cache        Cache that stores kernel function values
 */
template <typename algorithmFPType, CpuType cpu>
void SVMTrainImpl<boser, algorithmFPType, cpu>::setSV(
            Model *model, BlockMicroTable<algorithmFPType, readOnly, cpu> &mtX,
            size_t nFeatures, size_t nVectors, size_t nSV, const algorithmFPType *alpha,
            SVMCacheIface<algorithmFPType, cpu> *cache)
{
    algorithmFPType zero = (algorithmFPType)0.0;

    /* Allocate memory for storing support vectors and coefficients */
    NumericTablePtr svTable      = model->getSupportVectors();
    svTable->setNumberOfColumns(nFeatures);
    svTable->setNumberOfRows(nSV);
    svTable->allocateDataMemory();

    algorithmFPType *sv;

    BlockMicroTable<algorithmFPType, writeOnly, cpu> mtSv(svTable.get());
    mtSv.getBlockOfRows(0, nSV, &sv);

    for (size_t i = 0, iSV = 0; i < nVectors; i++)
    {
        if (alpha[i] == zero) { continue; }

        algorithmFPType *xi;
        size_t rowIndex = cache->getDataRowIndex(i);
        mtX.getBlockOfRows(rowIndex, 1, &xi);
        for (size_t j = 0; j < nFeatures; j++)
        {
            sv[iSV * nFeatures + j] = (algorithmFPType)xi[j];
        }
        mtX.release();
        iSV++;
    }
    mtSv.release();
}

/**
 * \brief Write support vectors in CSR format into resulting model
 *
 * \param[out] model        Resulting model
 * \param[in]  mtX          Micro table with input data set in CSR layout
 * \param[in]  nFeatures    Number of features in the input data set
 * \param[in]  nVectors     Number of observations in the input data set
 * \param[in]  nSV          Number of support vectors
 * \param[in]  alpha        Array of classification coefficients
 * \param[in]  cache        Cache that stores kernel function values
 */
template <typename algorithmFPType, CpuType cpu>
void SVMTrainImpl<boser, algorithmFPType, cpu>::setSV(
            Model *model, CSRBlockMicroTable<algorithmFPType, readOnly, cpu> &mtX,
            size_t nFeatures, size_t nVectors, size_t nSV, const algorithmFPType *alpha,
            SVMCacheIface<algorithmFPType, cpu> *cache)
{
    algorithmFPType zero = (algorithmFPType)0.0;
    algorithmFPType *xi;
    size_t *xiColIndices, *xiRowOffsets;
    size_t *svRowOffsetsBuffer = (size_t *)daal::services::daal_malloc((nSV + 1) * sizeof(size_t));
    if (!svRowOffsetsBuffer)
    { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    /* Calculate row offsets for the table that stores support vectors */
    svRowOffsetsBuffer[0] = 1;
    for (size_t i = 0, iSV = 0; i < nVectors; i++)
    {
        if (alpha[i] > zero)
        {
            size_t rowIndex = cache->getDataRowIndex(i);
            mtX.getSparseBlock(rowIndex, 1, &xi, &xiColIndices, &xiRowOffsets);
            svRowOffsetsBuffer[iSV + 1] = svRowOffsetsBuffer[iSV] + (xiRowOffsets[1] - xiRowOffsets[0]);
            mtX.release();
            iSV++;
        }
    }

    /* Allocate memory for storing support vectors and coefficients */
    services::SharedPtr<CSRNumericTable> svTable = services::staticPointerCast<CSRNumericTable, NumericTable>(
            model->getSupportVectors());
    svTable->setNumberOfColumns(nFeatures);
    svTable->setNumberOfRows(nSV);

    size_t svDataSize = svRowOffsetsBuffer[nSV] - svRowOffsetsBuffer[0];
    svTable->allocateDataMemory(svDataSize);

    /* Copy row offsets into the table */
    algorithmFPType *sv = NULL;
    size_t *svColIndices = NULL, *svRowOffsets = NULL;
    svTable->getArrays<algorithmFPType>(NULL, NULL, &svRowOffsets);
    for (size_t i = 0; i < nSV + 1; i++)
    {
        svRowOffsets[i] = svRowOffsetsBuffer[i];
    }

    CSRBlockMicroTable<algorithmFPType, writeOnly, cpu> mtSv(svTable.get());
    size_t nRowsRead = mtSv.getSparseBlock(0, nSV, &sv, &svColIndices, &svRowOffsets);

    for (size_t i = 0, iSV = 0, svOffset = 0; i < nVectors; i++)
    {
        if (alpha[i] == zero) { continue; }

        size_t rowIndex = cache->getDataRowIndex(i);
        nRowsRead = mtX.getSparseBlock(rowIndex, 1, &xi, &xiColIndices, &xiRowOffsets);

        size_t nNonZeroValuesInRow = xiRowOffsets[1] - xiRowOffsets[0];
        for (size_t j = 0; j < nNonZeroValuesInRow; j++, svOffset++)
        {
            sv          [svOffset] = xi[j];
            svColIndices[svOffset] = xiColIndices[j];
        }
        mtX.release();
        iSV++;
    }
    mtSv.release();
    daal::services::daal_free(svRowOffsetsBuffer);
}

/**
 * \brief Calculate the bias for the SVM model
 *
 * \param[in]  C        Upper bound in constraints of the quadratic optimization problem
 * \param[in]  nVectors Number of observations in the input data set
 * \param[in]  y        Array of class labels
 * \param[in]  alpha    Array of classification coefficients
 * \param[in]  grad     Gradient of the objective function
 * \return Bias for the SVM model
 */
template <typename algorithmFPType, CpuType cpu>
algorithmFPType SVMTrainImpl<boser, algorithmFPType, cpu>::calculateBias(
            algorithmFPType C, size_t nVectors, const algorithmFPType *y,
            const algorithmFPType *alpha, const algorithmFPType *grad)
{
    algorithmFPType bias;
    algorithmFPType zero = (algorithmFPType)0.0;
    algorithmFPType one  = (algorithmFPType)1.0;
    size_t num_yg = 0;
    algorithmFPType sum_yg = 0.0;

    algorithmFPType fpMax = MaxVal<algorithmFPType, cpu>::get();
    algorithmFPType ub = -(fpMax);
    algorithmFPType lb =   fpMax;

    for (size_t i = 0; i < nVectors; i++)
    {
        if (alpha[i] == zero) { continue; }

        algorithmFPType yg = -y[i] * grad[i];
        if      (y[i] ==  one && alpha[i] == C) { ub = ((ub > yg) ? ub : yg); } /// SVM_MAX(ub, yg);
        else if (y[i] == -one && alpha[i] == C) { lb = ((lb < yg) ? lb : yg); } /// SVM_MIN(lb, yg);
        else
        {
            sum_yg += yg;
            num_yg++;
        }
    }

    if (num_yg == 0)
    {
        bias = 0.5 * (ub + lb);
    }
    else
    {
        bias = sum_yg / (algorithmFPType)num_yg;
    }

    return bias;
}

/**
 * \brief Working set selection (WSS3) function.
 *        Select an index i from a pair of indices B = {i, j} using WSS 3 algorithm from [1].
 *
 * \param[in] nActiveVectors    number of observations in a training data set that are used
 *                              in sequential minimum optimization at the current iteration
 * \param[in] y                 array of class labels (+1 and -1)
 * \param[in] grad              gradient of the objective function
 * \param[in] kernelDiag        diagonal elements of the matrix Q (kernel(x[i], x[i]))
 * \param[in] I                 array of flags I_LOW and I_UP
 * \param[out] BiPtr            resulting index i
 *
 * \return The function returns m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha)
 */
template <typename algorithmFPType, CpuType cpu>
algorithmFPType SVMTrainImpl<boser, algorithmFPType, cpu>::WSSi(
            size_t nActiveVectors, const algorithmFPType *y, const algorithmFPType *grad,
            const algorithmFPType *kernelDiag, char *I, int *BiPtr)
{
    int Bi = -1;
    algorithmFPType GMax = -(MaxVal<algorithmFPType, cpu>::get());  // some big negative number

    /* Find i index of the working set (Bi) */
    for (size_t i = 0; i < nActiveVectors; i++)
    {
        if ((I[i] & up) != up) { continue; }
        algorithmFPType objFunc = -y[i] * grad[i];
        if (objFunc >= GMax)
        {
            GMax = objFunc;
            Bi = i;
        }
    }
    *BiPtr = Bi;
    return GMax;
}

/**
 * \brief Working set selection (WSS3) function for the case when the matrix Q is cached.
 *        Select an index j from a pair of indices B = {i, j} using WSS 3 algorithm from [1].
 *
 * \param[in] nActiveVectors    number of observations in a training data set that are used
 *                              in sequential minimum optimization at the current iteration
 * \param[in] tau               parameter of the working set selection algorithm
 * \param[in] y                 array of class labels (+1 and -1)
 * \param[in] grad              gradient of the objective function
 * \param[in] kernelDiag        diagonal elements of the matrix Q (kernel(x[i], x[i]))
 * \param[in] I                 array of flags I_LOW and I_UP
 * \param[in] Bi                index i from a pair of working set indices B = {i, j}
 * \param[in] Ki                Bi-th row of the matrix K, where K(i, j) = kernel(x[i], x[j])
 * \param[in] GMax              value of m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha) (see p.1891, eqn.6 in [1])
 * \param[out] BjPtr            resulting index j
 * \param[out] deltaPtr         optimal solution of the sub-problem of size 2:
 *                                  delta =  alpha[i]* - alpha[i]
 *                                  delta = -alpha[j]* + alpha[j]
 *
 * \return The function returns M(alpha) where
 *              M(alpha) = min(-y[i]*grad[i]): i belongs to I_low(alpha)
 */
template <typename algorithmFPType, CpuType cpu>
algorithmFPType SVMTrainImpl<boser, algorithmFPType, cpu>::WSSj(
            size_t nActiveVectors, algorithmFPType tau, const algorithmFPType *y,
            const algorithmFPType *grad, const algorithmFPType *kernelDiag, char *I,
            int Bi, SVMCacheIface<algorithmFPType, cpu> *cache, algorithmFPType GMax, int *BjPtr,
            algorithmFPType *deltaPtr)
{
    int Bj = -1;
    algorithmFPType fpMax = MaxVal<algorithmFPType, cpu>::get();
    algorithmFPType GMin  = fpMax; // some big positive number
    algorithmFPType GMin2 = fpMax;
    algorithmFPType delta;

    algorithmFPType zero = (algorithmFPType)0.0;
    algorithmFPType two  = (algorithmFPType)2.0;

    algorithmFPType Kii = kernelDiag[Bi];

    size_t nBlocks = nActiveVectors / kernelFunctionBlockSize;
    if (nBlocks * kernelFunctionBlockSize < nActiveVectors) { nBlocks++; }
    for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
    {
        size_t jStart = iBlock * kernelFunctionBlockSize;
        size_t jEnd   = jStart + kernelFunctionBlockSize;
        if (jEnd > nActiveVectors) { jEnd = nActiveVectors; }

        algorithmFPType *KiBlock = cache->getRowBlock(Bi, jStart, (jEnd - jStart));
        for (size_t j = jStart; j < jEnd; j++)
        {
            algorithmFPType ygrad = -y[j] * grad[j];
            if ((I[j] & low) != low) { continue; }
            if (ygrad <= GMin2)
            {
                GMin2 = ygrad;
            }
            if (ygrad >= GMax) { continue; }

            algorithmFPType b = GMax - ygrad;
            algorithmFPType a = Kii + kernelDiag[j] - two * KiBlock[j - jStart];
            if (a <= zero) { a = tau; }
            algorithmFPType dt = b / a;
            algorithmFPType objFunc = -b * dt;
            if (objFunc <= GMin)
            {
                GMin = objFunc;
                Bj = j;
                delta = dt;
            }
        }
    }

    *BjPtr = Bj;
    *deltaPtr = delta;
    return GMin2;
}

template <typename algorithmFPType, CpuType cpu>
bool SVMTrainImpl<boser, algorithmFPType, cpu>::findMaximumViolatingPair(
            size_t nActiveVectors, algorithmFPType tau, const algorithmFPType *y,
            const algorithmFPType *grad, const algorithmFPType *kernelDiag, char *I,
            SVMCacheIface<algorithmFPType, cpu> *cache, int *BiPtr, int *BjPtr,
            algorithmFPType *deltaPtr, algorithmFPType *maPtr, algorithmFPType *MaPtr,
            algorithmFPType *curEps)
{
    bool status = true;
    *BiPtr = -1;
    *maPtr = WSSi(nActiveVectors, y, grad, kernelDiag, I, BiPtr);
    if (*BiPtr == -1) { status = false; return status; }

    *BjPtr = -1;
    *MaPtr = WSSj(nActiveVectors, tau, y, grad, kernelDiag, I, *BiPtr, cache, *maPtr, BjPtr, deltaPtr);
    *curEps = (*maPtr) - (*MaPtr);
    if (*BjPtr == -1) { status = false; }

    return status;
}

template <typename algorithmFPType, CpuType cpu>
void SVMTrainImpl<boser, algorithmFPType, cpu>::updateTask(
            size_t nActiveVectors, algorithmFPType C, int Bi, int Bj, algorithmFPType delta, const algorithmFPType *y,
            algorithmFPType *alpha, algorithmFPType *grad, SVMTrainTask<algorithmFPType, cpu> &task)
{
    /* Update alpha */
    algorithmFPType newDeltai, newDeltaj;
    updateAlpha(C, Bi, Bj, delta, y, alpha, &newDeltai, &newDeltaj);
    task.updateI(C, Bj);
    task.updateI(C, Bi);

    algorithmFPType dyj = y[Bj] * newDeltaj;
    algorithmFPType dyi = y[Bi] * newDeltai;

    /* Update gradient */
    size_t blockSize = (kernelFunctionBlockSize >> 1);  // 2 rows from kernel function matrix are used
    size_t nBlocks = nActiveVectors / blockSize;
    if (nBlocks * blockSize < nActiveVectors) { nBlocks++; }

    for (size_t iBlock = 0; iBlock < nBlocks; iBlock++)
    {
        size_t tStart = iBlock * blockSize;
        size_t tEnd   = tStart + blockSize;
        if (tEnd > nActiveVectors) { tEnd = nActiveVectors; }

        algorithmFPType *KiBlock;
        algorithmFPType *KjBlock;
        task.cache->getTwoRowsBlock(Bi, Bj, tStart, (tEnd - tStart), &KiBlock, &KjBlock);
        for (size_t t = tStart; t < tEnd; t++)
        {
            grad[t] += dyi * y[t] * KiBlock[t - tStart];
            grad[t] += dyj * y[t] * KjBlock[t - tStart];
        }
    }
}

/**
 * \brief Update the array of classification coefficients. Step 4 of the Algorithm 1 in [1].
 *
 * \param[in] C          Upper bound in constraints of the quadratic optimization problem
 * \param[in] Bi         Index i from a pair of working set indices B = {i, j}
 * \param[in] Bj         Index j from a pair of working set indices B = {i, j}
 * \param[in] delta      Estimated difference between old and new value of the classification coefficients
 * \param[in] y          Array of class labels (+1 and -1)
 * \param[in,out] alpha  Array of classification coefficients
 * \param[out] newDeltai Resulting difference between old and new value of the Bi-th classification coefficient
 * \param[out] newDeltaj Resulting difference between old and new value of the Bj-th classification coefficient
 */
template <typename algorithmFPType, CpuType cpu>
inline void SVMTrainImpl<boser, algorithmFPType, cpu>::updateAlpha(
            algorithmFPType C, int Bi, int Bj, algorithmFPType delta, const algorithmFPType *y,
            algorithmFPType *alpha, algorithmFPType *newDeltai, algorithmFPType *newDeltaj)
{
    algorithmFPType zero = (algorithmFPType)0.0;
    algorithmFPType oldAlphai = alpha[Bi];
    algorithmFPType oldAlphaj = alpha[Bj];
    algorithmFPType yi = y[Bi];
    algorithmFPType yj = y[Bj];

    algorithmFPType newAlphai = oldAlphai + yi * delta;
    algorithmFPType newAlphaj = oldAlphaj - yj * delta;

    /* Project alpha back to the feasible region */
    algorithmFPType sum = yi * oldAlphai + yj * oldAlphaj;

    if (newAlphai > C)      { newAlphai = C;    }
    if (newAlphai < zero)   { newAlphai = zero; }
    newAlphaj = yj * (sum - yi * newAlphai);

    if (newAlphaj > C)      { newAlphaj = C;    }
    if (newAlphaj < zero)   { newAlphaj = zero; }
    newAlphai = yi * (sum - yj * newAlphaj);

    alpha[Bj] = newAlphaj;
    alpha[Bi] = newAlphai;
    *newDeltai = newAlphai - oldAlphai;
    *newDeltaj = newAlphaj - oldAlphaj;
}

/**
 * \brief Performs shrinking by moving the variables that corresponf to the shrunk feature vectors
 *        at the end of the respective arrays.
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in,out] y          Array of class labels (+1 and -1)
 * \param[in,out] alpha      Array of classification coefficients
 * \param[in,out] grad       Gradient of the objective function
 * \param[in,out] kernelDiag Diagonal elements of the matrix Q (kernel(x[i], x[i]))
 * \param[in,out] I          Array of flags that describe the status of feature vectors
 * \return Number of observations that remain active (not shrunk) after shrinking
 */
template <typename algorithmFPType, CpuType cpu>
size_t SVMTrainImpl<boser, algorithmFPType, cpu>::shrinkTask(
            size_t nActiveVectors, algorithmFPType *y, algorithmFPType *alpha, algorithmFPType *grad,
            algorithmFPType *kernelDiag, char *I)
{
    size_t i = 0;
    size_t j = nActiveVectors - 1;
    while(i < j)
    {
        while (!(I[i] & shrink) && i < nActiveVectors - 1)  i++;
        while ( (I[j] & shrink) && j > 0)                   j--;
        if (i >= j) break;
        daal::swap<char, cpu>(I[i], I[j]);
        daal::swap<algorithmFPType, cpu>(y[i], y[j]);
        daal::swap<algorithmFPType, cpu>(alpha[i], alpha[j]);
        daal::swap<algorithmFPType, cpu>(grad[i], grad[j]);
        daal::swap<algorithmFPType, cpu>(kernelDiag[i], kernelDiag[j]);
    }
    return i;
}

/**
 * \brief Update the array of flags that specify if the feature vector will be shrunk or not.
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in] C              Upper bound in constraints of the quadratic optimization problem
 * \param[in] ma             Value of m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha)  (see p.1891, eqn.6 in [1])
 * \param[in] Ma             Value of M(alpha) = min(-y[i]*grad[i]): i belongs to I_LOW (alpha)
 * \param[in] y              Array of class labels (+1 and -1)
 * \param[in] alpha          Array of classification coefficients
 * \param[in] grad           Gradient of the objective function
 * \param[out] I             Array of flags
 * \return Number of observations to be shrunk
 */
template <typename algorithmFPType, CpuType cpu>
size_t SVMTrainImpl<boser, algorithmFPType, cpu>::updateShrinkingFlags(
            size_t nActiveVectors, algorithmFPType C, algorithmFPType ma, algorithmFPType Ma,
            const algorithmFPType *y, const algorithmFPType *alpha, const algorithmFPType *grad, char *I)
{
    size_t nShrink = 0;
    for (size_t i = 0; i < nActiveVectors; i++)
    {
        I[i] &= (~shrink);
        algorithmFPType yg = -y[i]*grad[i];
        if (yg > ma)
        {
            if (alpha[i] <= 0.0 && y[i] == -1.0) { I[i] |= shrink; nShrink++; }
            if (alpha[i] >=   C && y[i] ==  1.0) { I[i] |= shrink; nShrink++; }
        }
        if (yg < Ma)
        {
            if (alpha[i] <= 0.0 && y[i] ==  1.0) { I[i] |= shrink; nShrink++; }
            if (alpha[i] >=   C && y[i] == -1.0) { I[i] |= shrink; nShrink++; }
        }
    }
    return nShrink;
}

/**
 * \brief Compute the values needed to check the optimality condition:
 *        m(alpha) <= M(alpha), where:
 * m(alpha) = max(-y[i]*grad[i]): i belongs to I_UP (alpha)  (see p.1891, eqn.6 in [1])
 * M(alpha) = min(-y[i]*grad[i]): i belongs to I_LOW (alpha)
 *
 * \param[in] nVectors  Number of observations in a training data set
 * \param[in] I         Array of flags that describe the status of feature vectors
 * \param[in] y         Array of class labels (+1 and -1)
 * \param[in] grad      Gradient of the objective function
 * \param[out] maPtr    Pointer to resulting m(alpha)
 * \param[out] MaPtr    Pointer to resulting M(alpha)
 */
template <typename algorithmFPType, CpuType cpu>
void SVMTrainImpl<boser, algorithmFPType, cpu>::computeOptimalityConditionValues(
            size_t nVectors, char *I, const algorithmFPType *y,
            const algorithmFPType *grad, algorithmFPType *maPtr, algorithmFPType *MaPtr)
{
    algorithmFPType ma = -(MaxVal<algorithmFPType, cpu>::get());  // some big negative number
    algorithmFPType Ma =  (MaxVal<algorithmFPType, cpu>::get());  // some big positive number

    for (size_t i = 0; i < nVectors; i++)
    {
        algorithmFPType yg = -y[i]*grad[i];
        if ((I[i] & up)  && (yg > ma)) ma = yg;
        if ((I[i] & low) && (yg < Ma)) Ma = yg;
    }
    *maPtr = ma;
    *MaPtr = Ma;
}

/**
 * \brief Recompute the values of optimization function gradient for the vectors that were shrunk
 *
 * \param[in] nVectors       Number of observations in a training data set
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in] cache          The structure needed to cache the values of matrix Q (kernel(x[i], x[j]))
 * \param[in] y              Array of class labels (+1 and -1)
 * \param[in] alpha          Array of classification coefficients
 * \param[in] grad           Gradient of the objective function
 */
template <typename algorithmFPType, CpuType cpu>
size_t SVMTrainImpl<boser, algorithmFPType, cpu>::reconstructGradient(
            size_t nVectors, size_t nActiveVectors, SVMCacheIface<algorithmFPType, cpu> *cache,
            const algorithmFPType *y, const algorithmFPType *alpha, algorithmFPType *grad)
{
    algorithmFPType negOne = (algorithmFPType)(-1.0);

    size_t nBlocks = nVectors / kernelFunctionBlockSize;
    if (nBlocks * kernelFunctionBlockSize < nVectors) { nBlocks++; }

    for (size_t i = nActiveVectors; i < nVectors; i++)
    {
        algorithmFPType yi = y[i];
        grad[i] = negOne;

        for (size_t jBlock = 0; jBlock < nBlocks; jBlock++)
        {
            size_t jStart = jBlock * kernelFunctionBlockSize;
            size_t jEnd   = jStart + kernelFunctionBlockSize;
            if (jEnd > nVectors) { jEnd = nVectors; }

            algorithmFPType *cacheRow = cache->getRowBlock(i, jStart, (jEnd - jStart));
            for (size_t j = jStart; j < jEnd; j++)
            {
                grad[i] += yi * y[j] * cacheRow[j - jStart] * alpha[j];
            }
        }
    }
    return nVectors;
}

/**
 * \brief Construct the structure that stores the intermediate data used in SVM training
 *
 * \param[in] cacheSize     Size of cache in bytes to store values of the kernel matrix
 * \param[in] nVectors      Number of observations in a training data set
 * \param[in] doShrinking   Flag that enables use of the shrinking optimization technique
 * \param[in] xTable        Pointer to numeric table that contains input data set
 * \param[in] yTable        Pointer to numeric table that contains class labels
 * \param[in] kernel        Kernel function
 * \param[in] _errors       Pointer to error collection associated with SVM training algorithm
 */
template <typename algorithmFPType, CpuType cpu>
SVMTrainTask<algorithmFPType, cpu>::SVMTrainTask(
            size_t cacheSize, size_t nVectors, size_t kernelFunctionBlockSize, bool doShrinking,
            NumericTablePtr xTable, NumericTable *yTable,
            services::SharedPtr<kernel_function::KernelIface> kernel,
            services::SharedPtr<services::KernelErrorCollection> _errors) :
        nVectors(nVectors), _errors(_errors)
{
    alpha      = daal::services::internal::service_calloc<algorithmFPType, cpu>(nVectors);
    I          = daal::services::internal::service_calloc<char,            cpu>(nVectors);
    y          = (algorithmFPType *)daal::services::daal_malloc(nVectors * sizeof(algorithmFPType));
    grad       = (algorithmFPType *)daal::services::daal_malloc(nVectors * sizeof(algorithmFPType));
    kernelDiag = (algorithmFPType *)daal::services::daal_malloc(nVectors * sizeof(algorithmFPType));
    if(alpha == NULL || I == NULL || y == NULL || grad == NULL || kernelDiag == NULL)
    {
        this->_errors->add(services::ErrorMemoryAllocationFailed); return;
    }

    if (cacheSize >= nVectors * nVectors * sizeof(algorithmFPType))
    {
        cache = new SVMCache<simpleCache,  algorithmFPType, cpu>(cacheSize, nVectors,
                    doShrinking, xTable, kernel, _errors);
    }
    else
    {
        cacheSize = kernelFunctionBlockSize;
        cache = new SVMCache<noCache,      algorithmFPType, cpu>(cacheSize, nVectors,
                    doShrinking, xTable, kernel, _errors);
    }

    algorithmFPType *ySrc;
    FeatureMicroTable<algorithmFPType, readOnly, cpu> mtY(yTable);
    mtY.getBlockOfColumnValues(0, 0, nVectors, &ySrc);
    daal::services::daal_memcpy_s(y, nVectors * sizeof(algorithmFPType), ySrc, nVectors * sizeof(algorithmFPType));
    mtY.release();
}

template <typename algorithmFPType, CpuType cpu>
SVMTrainTask<algorithmFPType, cpu>::~SVMTrainTask()
{
    daal::services::daal_free(alpha);
    daal::services::daal_free(y);
    daal::services::daal_free(grad);
    daal::services::daal_free(kernelDiag);
    daal::services::daal_free(I);
    if (cache)  { delete cache; }
}

/**
 * \brief Initialize the intermediate data used in SVM training
 *
 * \param[in] C     Upper bound in constraints of the quadratic optimization problem
 */
template <typename algorithmFPType, CpuType cpu>
void SVMTrainTask<algorithmFPType, cpu>::init(algorithmFPType C)
{
    algorithmFPType negOne  = (algorithmFPType)(-1.0);
    for (size_t i = 0; i < nVectors; i++)
    {
        grad[i] = negOne;
        updateI(C, i);
    }

    algorithmFPType *KiiPtr;
    for (size_t i = 0; i < nVectors; i++)
    {
        KiiPtr = cache->getRowBlock(i, i, 1);
        kernelDiag[i] = *KiiPtr;
    }
}

/**
 * \brief Update the flag that specify the status of the feature vector in the input data set
 *
 * \param[in] C     Upper bound in constraints of the quadratic optimization problem
 * \param[in] index Index of the feature vector
 */
template <typename algorithmFPType, CpuType cpu>
inline void SVMTrainTask<algorithmFPType, cpu>::updateI(algorithmFPType C, size_t index)
{
    algorithmFPType zero = (algorithmFPType)0.0;
    algorithmFPType one  = (algorithmFPType)1.0;
    char Ii = I[index];
    algorithmFPType alphai = alpha[index];
    algorithmFPType yi     = y[index];
    Ii &= (char)shrink;
    if (alphai < C    && yi == +one) { Ii |= up; }
    if (alphai > zero && yi == -one) { Ii |= up; }
    if (alphai < C    && yi == -one) { Ii |= low; }
    if (alphai > zero && yi == +one) { Ii |= low; }
    I[index] = Ii;
}

/**
 * \brief Move the indices of the shrunk feature vector to the end of the array
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in] I              Array of flags that describe the status of feature vectors
 */
template<typename algorithmFPType, CpuType cpu>
void SVMCacheRowGetter<noCache, true, algorithmFPType, cpu>::updateShrinkingRowIndices(
        size_t nActiveVectors, const char *I, size_t _nLines, size_t _lineSize, algorithmFPType *_cache, size_t *shrinkingRowIndices)
{
    size_t i = 0;
    size_t j = nActiveVectors-1;
    while(i < j)
    {
        while (!(I[i] & shrink) && i < nActiveVectors - 1) i++;
        while ( (I[j] & shrink) && j > 0)                  j--;
        if (i >= j) break;
        daal::swap<size_t, cpu>(shrinkingRowIndices[i], shrinkingRowIndices[j]);
        i++;
        j--;
    }
}

/**
 * \brief Move the indices of the shrunk feature vector to the end of the array and
 *        re-order rows and columns in the cache accordingly
 *
 * \param[in] nActiveVectors Number of observations in a training data set that are used
 *                           in sequential minimum optimization at the current iteration
 * \param[in] I              Array of flags that describe the status of feature vectors
 */
template<typename algorithmFPType, CpuType cpu>
void SVMCacheRowGetter<simpleCache, true, algorithmFPType, cpu>::updateShrinkingRowIndices(
        size_t nActiveVectors, const char *I, size_t _nLines, size_t _lineSize, algorithmFPType *_cache, size_t *shrinkingRowIndices)
{
    size_t i = 0;
    size_t j = nActiveVectors-1;
    while(i < j)
    {
        while (!(I[i] & shrink) && i < nActiveVectors - 1) i++;
        while ( (I[j] & shrink) && j > 0)                  j--;
        if (i >= j) break;
        daal::swap<size_t, cpu>(shrinkingRowIndices[i], shrinkingRowIndices[j]);

        algorithmFPType cacheii = _cache[i * _lineSize + i];
        algorithmFPType cacheij = _cache[i * _lineSize + j];
        algorithmFPType cachejj = _cache[j * _lineSize + j];

        /* Swap i-th and j-th row in cache */
        size_t lineSizeInBytes = _lineSize * sizeof(algorithmFPType);
        daal::services::daal_memcpy_s(_tmp,                   lineSizeInBytes, _cache + i * _lineSize, lineSizeInBytes);
        daal::services::daal_memcpy_s(_cache + i * _lineSize, lineSizeInBytes, _cache + j * _lineSize, lineSizeInBytes);
        daal::services::daal_memcpy_s(_cache + j * _lineSize, lineSizeInBytes, _tmp,                   lineSizeInBytes);

        /* Swap i-th and j-th column in cache */
        for (size_t k = 0; k < _nLines; k++)
        {
            daal::swap<algorithmFPType, cpu>(_cache[i + k * _lineSize], _cache[j + k * _lineSize]);
        }

        // maybe not needed
        _cache[i * _lineSize + i] = cachejj;
        _cache[j * _lineSize + j] = cacheii;
        _cache[j * _lineSize + i] = cacheij;
        _cache[i * _lineSize + j] = cacheij;
        i++;
        j--;
    }
}

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif
