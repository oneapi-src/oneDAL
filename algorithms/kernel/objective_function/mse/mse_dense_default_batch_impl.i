/* file: mse_dense_default_batch_impl.i */
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
//  Implementation of mse algorithm
//--
*/
#include "service_math.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace mse
{
namespace internal
{
/**
 *  \brief Kernel for mse objective function calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
inline services::Status MSEKernel<algorithmFPType, method, cpu>::compute(NumericTable * dataNT, NumericTable * dependentVariablesNT,
                                                                         NumericTable * argumentNT, NumericTable * valueNT, NumericTable * hessianNT,
                                                                         NumericTable * gradientNT, NumericTable * nonSmoothTermValue,
                                                                         NumericTable * proximalProjection, NumericTable * lipschitzConstant,
                                                                         NumericTable * componentOfGradient,
                                                                         NumericTable * componentOfHessianDiagonal,
                                                                         NumericTable * componentOfProximalProjection, Parameter * parameter)
{
    const size_t nDataRows = dataNT->getNumberOfRows();
    int result             = 0;
    if (componentOfGradient || componentOfHessianDiagonal || componentOfProximalProjection)
    {
        const size_t id     = parameter->featureId;
        const size_t nTheta = dataNT->getNumberOfColumns();
        DAAL_INT yDim       = dependentVariablesNT->getNumberOfColumns();
        if (!previousFeatureValuesPtr)
        {
            previousFeatureValues.reset(yDim);
            previousFeatureValuesPtr = previousFeatureValues.get();
        }

        if (componentOfGradient || componentOfHessianDiagonal)
        {
            if (xNT != dataNT)
            {
                xNT                                               = dataNT;
                SOANumericTable * soaDataPtr                      = dynamic_cast<SOANumericTable *>(dataNT);
                HomogenNumericTable<algorithmFPType> * hmgDataPtr = dynamic_cast<HomogenNumericTable<algorithmFPType> *>(dataNT);
                if (hmgDataPtr)
                {
                    X              = hmgDataPtr->getArray();
                    transposedData = false;
                }
                else if (soaDataPtr)
                {
                    const bool featuresEqual                            = soaDataPtr->getDictionary()->getFeaturesEqual();
                    bool isContinuous                                   = false;
                    data_management::features::IndexNumType currentType = features::internal::getIndexNumType<algorithmFPType>();
                    if (currentType == (*(soaDataPtr->getDictionary()))[0].getIndexType())
                    {
                        isContinuous = true;
                        if (!featuresEqual)
                        {
                            for (size_t i = 1; i < nTheta; i++)
                            {
                                if (currentType != (*(soaDataPtr->getDictionary()))[i].getIndexType())
                                {
                                    isContinuous = false;
                                    break;
                                }
                            }
                        }
                        for (size_t i = 1; i < nTheta && isContinuous; i++)
                        {
                            algorithmFPType * fisrtArrayPtr = (algorithmFPType *)(soaDataPtr->getArray(i - 1));
                            algorithmFPType * lastArrayPtr  = (algorithmFPType *)(soaDataPtr->getArray(i));
                            if ((lastArrayPtr - fisrtArrayPtr) != (nDataRows))
                            {
                                isContinuous = false;
                                break;
                            }
                        }
                    }
                    if (isContinuous)
                    {
                        X              = (algorithmFPType *)soaDataPtr->getArray(0);
                        transposedData = true;
                    }
                    else
                    {
                        XPtr.set(dataNT, 0, nDataRows);
                        DAAL_CHECK_BLOCK_STATUS(XPtr);
                        X              = const_cast<algorithmFPType *>(XPtr.get());
                        transposedData = false;
                    }
                }
                else
                {
                    XPtr.set(dataNT, 0, nDataRows);
                    DAAL_CHECK_BLOCK_STATUS(XPtr);
                    X              = const_cast<algorithmFPType *>(XPtr.get());
                    transposedData = false;
                }
            }
        }

        if (componentOfGradient)
        {
            char trans               = 'T';
            char notrans             = 'N';
            algorithmFPType one      = 1.0;
            algorithmFPType minusOne = -1.0;
            algorithmFPType zero     = 0.0;
            DAAL_INT n               = (DAAL_INT)nDataRows;
            DAAL_INT dim             = (DAAL_INT)nTheta;
            DAAL_INT ione            = 1;

            WriteRows<algorithmFPType, cpu> grPtr;
            if (gradNT != componentOfGradient)
            {
                DAAL_ASSERT(componentOfGradient->getNumberOfRows() == 1);
                grPtr.set(componentOfGradient, 0, 1);
                DAAL_CHECK_BLOCK_STATUS(grPtr);
                gr     = grPtr.get();
                gradNT = componentOfGradient;
            }

            WriteRows<algorithmFPType, cpu> beta;
            if (betaNT != argumentNT)
            {
                beta.set(argumentNT, 0, nTheta + 1); /* as we have intercept */
                DAAL_CHECK_BLOCK_STATUS(beta);
                b      = beta.get();
                betaNT = argumentNT;
            }
            if (nDataRows < nTheta || parameter->interceptFlag)
            {
                if (dotPtr == nullptr)
                {
                    dot.reset(yDim);
                    dotPtr = dot.get();
                    DAAL_CHECK_MALLOC(dotPtr);
                }
                if (((previousInputData != nullptr) && (previousInputData != X)) || (previousInputData == nullptr))
                {
                    const algorithmFPType * fB = b;

                    WriteRows<algorithmFPType, cpu> YPtr(dependentVariablesNT, 0, nDataRows);
                    DAAL_CHECK_BLOCK_STATUS(YPtr);
                    const algorithmFPType * Y = YPtr.get();

                    previousInputData = X;
                    previousFeatureId = -1;
                    residual.reset(nDataRows * yDim);
                    residualPtr = residual.get();

                    result |= daal::services::internal::daal_memcpy_s(residualPtr, n * yDim * sizeof(algorithmFPType), Y,
                                                                      n * yDim * sizeof(algorithmFPType));
                    size_t compute_matrix = 0;
                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t i = 0; i < (nTheta + 1) * yDim; i++)
                    {
                        compute_matrix += (fB[i] != 0);
                    }

                    const size_t blockSize = 256;
                    size_t nBlocks         = nDataRows / blockSize;
                    nBlocks += (nBlocks * blockSize != nDataRows);
                    if (compute_matrix)
                    {
                        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
                            const size_t startRow   = iBlock * blockSize;
                            const size_t finishRow  = (iBlock + 1 == nBlocks ? nDataRows : (iBlock + 1) * blockSize);
                            DAAL_INT localBlockSize = finishRow - startRow;

                            if (transposedData)
                            {
                                Blas<algorithmFPType, cpu>::xxgemm(&notrans, &notrans, &localBlockSize, &yDim, &dim, &minusOne, X + startRow, &n,
                                                                   fB + yDim, &yDim, &one, residualPtr + startRow * yDim, &yDim);
                            }
                            else
                            {
                                Blas<algorithmFPType, cpu>::xxgemm(&trans, &notrans, &localBlockSize, &yDim, &dim, &minusOne, X + startRow * dim,
                                                                   &dim, fB + yDim, &yDim, &one, residualPtr + startRow * yDim, &yDim);
                            }
                            if (parameter->interceptFlag)
                            {
                                for (size_t ic = 0; ic < yDim; ic++)
                                {
                                    for (size_t i = startRow; i < finishRow; i++) /*threader for*/
                                    {
                                        residualPtr[i * yDim + ic] -= fB[ic];
                                    }
                                }
                            }
                        });
                    }

                    if (componentOfHessianDiagonal)
                    {
                        hessianDiagonal.reset(nTheta);
                        hessianDiagonalPtr           = hessianDiagonal.get();
                        algorithmFPType inverseNData = (algorithmFPType)(1.0) / nDataRows;

                        PRAGMA_IVDEP
                        PRAGMA_VECTOR_ALWAYS
                        for (size_t j = 0; j < nTheta; j++)
                        {
                            hessianDiagonalPtr[j] = 0; /*USE DOTPRODUCT or parallel computation*/
                        }

                        if (transposedData)
                        {
                            TlsMem<algorithmFPType, cpu, services::internal::ScalableCalloc<algorithmFPType, cpu> > tlsData(nTheta);
                            daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
                                algorithmFPType * hessianDiagonalLocal = tlsData.local();
                                const size_t startRow                  = iBlock * blockSize;
                                const size_t finishRow                 = (iBlock + 1 == nBlocks ? nDataRows : (iBlock + 1) * blockSize);

                                for (size_t j = 0; j < nTheta; j++)
                                {
                                    PRAGMA_IVDEP
                                    PRAGMA_VECTOR_ALWAYS
                                    for (size_t i = startRow; i < finishRow; i++)
                                    {
                                        hessianDiagonalLocal[j] += X[j * n + i] * X[j * n + i]; /*USE DOTPRODUCT or parallel computation*/
                                    }
                                }
                            });
                            tlsData.reduce([&](algorithmFPType * localHes) {
                                PRAGMA_IVDEP
                                PRAGMA_VECTOR_ALWAYS
                                for (int j = 0; j < nTheta; j++)
                                {
                                    hessianDiagonalPtr[j] += localHes[j];
                                }
                            });
                        }
                        else
                        {
                            TlsMem<algorithmFPType, cpu, services::internal::ScalableCalloc<algorithmFPType, cpu> > tlsData(nTheta);
                            daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
                                algorithmFPType * hessianDiagonalLocal = tlsData.local();
                                const size_t startRow                  = iBlock * blockSize;
                                const size_t finishRow                 = (iBlock + 1 == nBlocks ? nDataRows : (iBlock + 1) * blockSize);

                                for (size_t i = startRow; i < finishRow; i++)
                                {
                                    PRAGMA_IVDEP
                                    PRAGMA_VECTOR_ALWAYS
                                    for (size_t j = 0; j < nTheta; j++)
                                    {
                                        hessianDiagonalLocal[j] += X[i * dim + j] * X[i * dim + j]; /*USE DOTPRODUCT or parallel computation*/
                                    }
                                }
                            });
                            tlsData.reduce([&](algorithmFPType * localHes) {
                                PRAGMA_IVDEP
                                PRAGMA_VECTOR_ALWAYS
                                for (int j = 0; j < nTheta; j++)
                                {
                                    hessianDiagonalPtr[j] += localHes[j];
                                }
                            });
                        }
                        PRAGMA_IVDEP
                        PRAGMA_VECTOR_ALWAYS
                        for (size_t j = 0; j < nTheta; j++)
                        {
                            hessianDiagonalPtr[j] *= inverseNData;
                        }
                    }
                }

                if (previousFeatureId >= 0)
                {
                    for (size_t ic = 0; ic < yDim; ic++) /*use ger or gemm for better performance*/
                    {
                        const algorithmFPType curentBetaValue = b[previousFeatureId * yDim + ic];
                        const algorithmFPType diff            = previousFeatureValuesPtr[ic] - curentBetaValue;

                        if (diff != 0)
                        {
                            if (previousFeatureId == 0 && parameter->interceptFlag)
                            {
                                PRAGMA_IVDEP
                                PRAGMA_VECTOR_ALWAYS
                                for (size_t i = 0; i < nDataRows; i++) /*threader for*/
                                {
                                    residualPtr[i * yDim + ic] += diff;
                                }
                            }
                            if (previousFeatureId != 0)
                            {
                                if (transposedData)
                                {
                                    daal::internal::Blas<algorithmFPType, cpu>::xxaxpy(&n, &diff, X + (previousFeatureId - 1) * n, &ione,
                                                                                       residualPtr + ic, &yDim);
                                }
                                else
                                {
                                    daal::internal::Blas<algorithmFPType, cpu>::xxaxpy(&n, &diff, X + (previousFeatureId - 1), &dim, residualPtr + ic,
                                                                                       &yDim);
                                }
                            }
                        }
                    }
                }
                for (size_t ic = 0; ic < yDim; ic++)
                {
                    dotPtr[ic] = 0;
                }

                if (id == 0)
                {
                    if (parameter->interceptFlag)
                    {
                        for (size_t i = 0; i < nDataRows; i++) /*threader for*/
                        {
                            PRAGMA_IVDEP
                            PRAGMA_VECTOR_ALWAYS
                            for (size_t ic = 0; ic < yDim; ic++) dotPtr[ic] += residualPtr[i * yDim + ic];
                        }
                    }
                }
                else
                {
                    if (transposedData)
                    {
                        for (size_t ic = 0; ic < yDim; ic++)
                        {
                            dotPtr[ic] += daal::internal::Blas<algorithmFPType, cpu>::xxdot(&n, X + (id - 1) * n, &ione, residualPtr + ic, &yDim);
                        }
                    }
                    else
                    {
                        for (size_t ic = 0; ic < yDim; ic++)
                        {
                            dotPtr[ic] += daal::internal::Blas<algorithmFPType, cpu>::xxdot(&n, X + (id - 1), &dim, residualPtr + ic, &yDim);
                        }
                    }
                }

                /*store previous values for update*/
                previousFeatureId = id;
                for (size_t ic = 0; ic < yDim; ic++)
                {
                    previousFeatureValuesPtr[ic] = b[ic + id * yDim];
                }

                algorithmFPType inverseNData = (algorithmFPType)(1.0) / nDataRows;
                for (size_t ic = 0; ic < yDim; ic++)
                {
                    gr[ic] = (algorithmFPType)(-1.0) * inverseNData * dot[ic];
                }
            }
            else
            {
                if (((previousInputData != nullptr) && (previousInputData != X)) || (previousInputData == nullptr))
                {
                    const algorithmFPType * fB = b;

                    WriteRows<algorithmFPType, cpu> YPtr(dependentVariablesNT, 0, nDataRows);
                    DAAL_CHECK_BLOCK_STATUS(YPtr);
                    const algorithmFPType * Y = YPtr.get();

                    previousInputData = X;
                    previousFeatureId = -1;
                    gramMatrix.reset((nTheta) * (nTheta));
                    gramMatrixPtr = gramMatrix.get();
                    XY.reset(dim * yDim);
                    XYPtr = XY.get();

                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t i = 0; i < dim * yDim; i++) XYPtr[i] = 0;

                    PRAGMA_IVDEP
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t i = 0; i < dim * dim; i++) gramMatrixPtr[i] = 0;
                    char uplo = 'L';

                    const size_t blockSize = 256;
                    DAAL_INT blockSizeDim  = (DAAL_INT)blockSize;
                    size_t nBlocks         = nDataRows / blockSize;
                    nBlocks += (nBlocks * blockSize != nDataRows);
                    TlsMem<algorithmFPType, cpu, services::internal::ScalableCalloc<algorithmFPType, cpu> > tlsData(dim * yDim + (nTheta) * (nTheta));
                    const size_t disp = dim * yDim;
                    daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
                        algorithmFPType * localXY   = tlsData.local();
                        algorithmFPType * localGram = localXY + disp;
                        const size_t startRow       = iBlock * blockSize;
                        DAAL_INT localBlockSizeDim  = (((iBlock + 1) == nBlocks) ? (nDataRows - startRow) : blockSizeDim);

                        if (transposedData)
                        {
                            daal::internal::Blas<algorithmFPType, cpu>::xxgemm(&notrans, &notrans, &yDim, &dim, &localBlockSizeDim, &one,
                                                                               Y + startRow * yDim, &yDim, X + startRow, &n, &one, localXY, &yDim);
                            Blas<algorithmFPType, cpu>::xxsyrk(&uplo, &trans, &dim, &localBlockSizeDim, &one, X + startRow, &n, &one, localGram,
                                                               &dim);
                        }
                        else
                        {
                            daal::internal::Blas<algorithmFPType, cpu>::xxgemm(&notrans, &trans, &yDim, &dim, &localBlockSizeDim, &one,
                                                                               Y + startRow * yDim, &yDim, X + startRow * dim, &dim, &one, localXY,
                                                                               &yDim);
                            Blas<algorithmFPType, cpu>::xxsyrk(&uplo, &notrans, &dim, &localBlockSizeDim, &one, X + startRow * dim, &dim, &one,
                                                               localGram, &dim);
                        }
                    });
                    tlsData.reduce([&](algorithmFPType * local) {
                        PRAGMA_IVDEP
                        PRAGMA_VECTOR_ALWAYS
                        for (int j = 0; j < dim * yDim; j++)
                        {
                            XYPtr[j] += local[j];
                        }
                        PRAGMA_IVDEP
                        PRAGMA_VECTOR_ALWAYS
                        for (int j = 0; j < dim * dim; j++)
                        {
                            gramMatrixPtr[j] += local[j + disp];
                        }
                    });
                    const size_t dimension = dim;
                    for (size_t i = 0; i < dimension; i++)
                    {
                        PRAGMA_IVDEP
                        PRAGMA_VECTOR_ALWAYS
                        for (size_t j = i; j < dimension; j++) gramMatrixPtr[j * dim + i] = gramMatrixPtr[i * dim + j];
                    }

                    gradientForGram.reset(nTheta * yDim);
                    gradientForGramPtr = gradientForGram.get();

                    daal::internal::Blas<algorithmFPType, cpu>::xgemm(&notrans, &trans, &yDim, &dim, &dim, &one, fB, &yDim, gramMatrixPtr, &dim,
                                                                      &zero, gradientForGramPtr, &yDim);
                }
                if (previousFeatureId >= 0)
                {
                    const algorithmFPType * curentBetaValue = b + previousFeatureId * yDim;

                    //use ger or gemm
                    for (size_t i = 0; i < yDim; i++)
                    {
                        algorithmFPType diff = curentBetaValue[i] - previousFeatureValuesPtr[i];
                        if (diff != 0)
                        {
                            if (previousFeatureId != 0)
                            {
                                daal::internal::Blas<algorithmFPType, cpu>::xaxpy(&dim, &diff, gramMatrixPtr + (previousFeatureId - 1) * dim, &ione,
                                                                                  gradientForGramPtr + i, &yDim);
                            }
                        }
                    }
                }

                previousFeatureId = id;
                for (size_t ic = 0; ic < yDim; ic++)
                {
                    previousFeatureValuesPtr[ic] = b[ic + id * yDim];
                }
                if (id > 0)
                {
                    algorithmFPType inverseNData = (algorithmFPType)(1.0) / nDataRows;
                    for (size_t i = 0; i < yDim; i++)
                    {
                        gr[i] = (algorithmFPType)(-1.0) * inverseNData * (XYPtr[(id - 1) * yDim + i] - gradientForGramPtr[(id - 1) * yDim + i]);
                    }
                }
                else
                {
                    for (size_t i = 0; i < yDim; i++)
                    {
                        gr[i] = 0;
                    }
                }
            }
        }
        if (componentOfHessianDiagonal)
        {
            WriteRows<algorithmFPType, cpu> hessianPtr;
            if (hesDiagonalNT != componentOfHessianDiagonal)
            {
                DAAL_ASSERT(componentOfHessianDiagonal->getNumberOfRows() == 1);
                hessianPtr.set(componentOfHessianDiagonal, 0, 1);
                h             = hessianPtr.get();
                hesDiagonalNT = componentOfHessianDiagonal;
            }
            if (id == 0)
            {
                if (parameter->interceptFlag)
                {
                    for (size_t i = 0; i < yDim; i++)
                    {
                        h[i] = 1;
                    }
                }
                else
                {
                    for (size_t i = 0; i < yDim; i++)
                    {
                        h[i] = 0;
                    }
                }
            }
            else
            {
                if (gramMatrixPtr != nullptr)
                {
                    algorithmFPType inverseNData = (algorithmFPType)(1.0) / nDataRows;
                    const algorithmFPType hes    = inverseNData * (gramMatrixPtr[(id - 1) * nTheta + (id - 1)]);
                    for (size_t i = 0; i < yDim; i++)
                    {
                        h[i] = hes;
                    }
                }
                else
                {
                    if (((previousInputData != nullptr) && (previousInputData != X)) || (previousInputData == nullptr))
                    {
                        DAAL_INT dim = (DAAL_INT)nTheta;

                        hessianDiagonal.reset(nTheta);
                        hessianDiagonalPtr           = hessianDiagonal.get();
                        algorithmFPType inverseNData = (algorithmFPType)(1.0) / nDataRows;

                        PRAGMA_IVDEP
                        PRAGMA_VECTOR_ALWAYS
                        for (size_t j = 0; j < nTheta; j++)
                        {
                            hessianDiagonalPtr[j] = 0; /*USE DOTPRODUCT or parallel computation*/
                        }
                        for (size_t i = 0; i < nDataRows; i++)
                        {
                            PRAGMA_IVDEP
                            PRAGMA_VECTOR_ALWAYS
                            for (size_t j = 0; j < nTheta; j++)
                            {
                                hessianDiagonalPtr[j] += X[i * dim + j] * X[i * dim + j]; /*USE DOTPRODUCT or parallel computation*/
                            }
                        }
                        PRAGMA_IVDEP
                        PRAGMA_VECTOR_ALWAYS
                        for (size_t j = 0; j < nTheta; j++)
                        {
                            hessianDiagonalPtr[j] *= inverseNData;
                        }
                    }
                    for (size_t ic = 0; ic < yDim; ic++)
                    {
                        h[ic] = hessianDiagonalPtr[id - 1];
                    }
                }
            }
        }

        if (componentOfProximalProjection)
        {
            ReadRows<float, cpu> penaltyL1BD;
            if (penaltyL1NT != parameter->penaltyL1.get())
            {
                penaltyL1NT = parameter->penaltyL1.get();
                penaltyL1BD.set(*parameter->penaltyL1, 0, 1);
                penaltyL1Ptr = const_cast<float *>(penaltyL1BD.get());
            }
            float l1 = penaltyL1Ptr[0];

            WriteRows<algorithmFPType, cpu> proxBD;
            if (proxNT != componentOfProximalProjection)
            {
                DAAL_ASSERT(componentOfProximalProjection->getNumberOfRows() == 1);
                proxNT = componentOfProximalProjection;
                proxBD.set(componentOfProximalProjection, 0, 1);
                proxPtr = proxBD.get();
            }
            algorithmFPType * p = proxPtr;

            WriteRows<algorithmFPType, cpu> beta;
            if (betaNT != argumentNT)
            {
                beta.set(argumentNT, 0, nTheta + 1); /* as we have intercept */
                DAAL_CHECK_BLOCK_STATUS(beta);
                b      = beta.get();
                betaNT = argumentNT;
            }
            const algorithmFPType * bI = b + id * yDim;

            const size_t proxSize = yDim;
            if (id == 0)
            {
                for (size_t i = 0; i < proxSize; i++) p[i] = bI[i];
            }
            else
            {
                for (size_t i = 0; i < proxSize; i++)
                {
                    if (bI[i] > l1)
                    {
                        p[i] = bI[i] - l1;
                    }
                    if (bI[i] < -l1)
                    {
                        p[i] = bI[i] + l1;
                    }
                    if (daal::internal::Math<algorithmFPType, cpu>::sFabs(bI[i]) <= l1)
                    {
                        p[i] = 0;
                    }
                }
            }
        }
        return services::Status();
    }
    if (proximalProjection)
    {
        const size_t nBeta = argumentNT->getNumberOfRows();
        DAAL_ASSERT(proximalProjection->getNumberOfRows() == nBeta);
        WriteRows<algorithmFPType, cpu> prox(proximalProjection, 0, nBeta);
        ReadRows<algorithmFPType, cpu> beta(argumentNT, 0, nBeta);

        algorithmFPType * p       = prox.get();
        const algorithmFPType * b = beta.get();

        for (int i = 0; i < nBeta; i++)
        {
            p[i] = b[i];
        }
        return services::Status();
    }

    if (lipschitzConstant)
    {
        const size_t n = dataNT->getNumberOfRows();
        const size_t p = dataNT->getNumberOfColumns();
        ReadRows<algorithmFPType, cpu> xPtr(dataNT, 0, n);
        const algorithmFPType * x = xPtr.get();
        DAAL_ASSERT(lipschitzConstant->getNumberOfRows() == 1);
        WriteRows<algorithmFPType, cpu> lipschitzConstantPtr(lipschitzConstant, 0, 1);
        algorithmFPType & c = *lipschitzConstantPtr.get();

        const size_t blockSize = 256;
        size_t nBlocks         = n / blockSize;
        nBlocks += (nBlocks * blockSize != n);
        algorithmFPType globalMaxNorm = 0;

        TlsMem<algorithmFPType, cpu, services::internal::ScalableCalloc<algorithmFPType, cpu> > tlsData(lipschitzConstant->getNumberOfRows());
        daal::threader_for(nBlocks, nBlocks, [&](const size_t iBlock) {
            algorithmFPType & _maxNorm = *tlsData.local();
            const size_t startRow      = iBlock * blockSize;
            const size_t finishRow     = (iBlock + 1 == nBlocks ? n : (iBlock + 1) * blockSize);
            algorithmFPType curentNorm = 0;
            for (size_t i = startRow; i < finishRow; i++)
            {
                curentNorm = 0;
                for (int j = 0; j < p; j++)
                {
                    curentNorm += x[i * p + j] * x[i * p + j];
                }
                if (curentNorm > _maxNorm)
                {
                    _maxNorm = curentNorm;
                }
            }
        });
        tlsData.reduce([&](algorithmFPType * maxNorm) {
            if (globalMaxNorm < *maxNorm)
            {
                globalMaxNorm = *maxNorm;
            }
        });

        algorithmFPType lipschitz = (globalMaxNorm + 1);
        c                         = 2 * lipschitz;
        return services::Status();
    }

    if (nonSmoothTermValue)
    {
        WriteRows<algorithmFPType, cpu> vr(nonSmoothTermValue, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(vr);
        algorithmFPType & v = *vr.get();
        v                   = 0;
        return services::Status();
    }

    if (parameter->batchIndices.get() != NULL && parameter->batchIndices->getNumberOfColumns() != nDataRows)
    {
        MSETaskSample<algorithmFPType, cpu> task(dataNT, dependentVariablesNT, argumentNT, valueNT, hessianNT, gradientNT, parameter,
                                                 blockSizeDefault);
        return run(task);
    }
    MSETaskAll<algorithmFPType, cpu> task(dataNT, dependentVariablesNT, argumentNT, valueNT, hessianNT, gradientNT, parameter, blockSizeDefault);
    return (!result) ? run(task) : services::Status(services::ErrorMemoryCopyFailedInternal);
}

template <typename algorithmFPType, Method method, CpuType cpu>
services::Status MSEKernel<algorithmFPType, method, cpu>::run(MSETask<algorithmFPType, cpu> & task)
{
    algorithmFPType * argumentArray = nullptr;
    Status s                        = task.init(argumentArray);
    if (!s) return s;
    algorithmFPType *dataBlock = nullptr, *dependentVariablesBlock = nullptr;
    algorithmFPType *value = nullptr, *gradient = NULL, *hessian = nullptr;
    DAAL_CHECK_STATUS(s, task.getResultValues(value, gradient, hessian));
    task.setResultValuesToZero(value, gradient, hessian);

    size_t blockSize = blockSizeDefault;
    size_t nBlocks   = task.batchSize / blockSizeDefault;
    nBlocks += (nBlocks * blockSizeDefault != task.batchSize);
    if (nBlocks == 1)
    {
        blockSize = task.batchSize;
    }

    for (size_t block = 0; s.ok() && (block < nBlocks); block++)
    {
        if (block == nBlocks - 1) blockSize = task.batchSize - block * blockSizeDefault;

        s = task.getCurrentBlock(block * blockSizeDefault, blockSize, dataBlock, dependentVariablesBlock);
        if (s) computeMSE(blockSize, task, dataBlock, argumentArray, dependentVariablesBlock, value, gradient, hessian);
        task.releaseCurrentBlock();
    }

    if (s) normalizeResults(task, value, gradient, hessian);
    task.releaseResultValues();
    return s;
}

template <typename algorithmFPType, Method method, CpuType cpu>
inline void MSEKernel<algorithmFPType, method, cpu>::computeMSE(size_t blockSize, MSETask<algorithmFPType, cpu> & task, algorithmFPType * data,
                                                                algorithmFPType * argumentArray, algorithmFPType * dependentVariablesArray,
                                                                algorithmFPType * value, algorithmFPType * gradient, algorithmFPType * hessian)
{
    char trans                   = 'T';
    algorithmFPType one          = 1.0;
    algorithmFPType zero         = 0.0;
    DAAL_INT n                   = (DAAL_INT)blockSize;
    size_t nTheta                = task.nTheta;
    DAAL_INT dim                 = (DAAL_INT)nTheta;
    DAAL_INT ione                = 1;
    algorithmFPType theta0       = argumentArray[0];
    algorithmFPType * theta      = &argumentArray[1];
    algorithmFPType * xMultTheta = task.xMultTheta.get();

    if (task.gradientFlag || task.valueFlag)
    {
        Blas<algorithmFPType, cpu>::xgemv(&trans, &dim, &n, &one, data, &dim, theta, &ione, &zero, xMultTheta, &ione);

        for (size_t i = 0; i < blockSize; i++)
        {
            xMultTheta[i] = xMultTheta[i] + theta0 - dependentVariablesArray[i];
        }
    }

    if (task.gradientFlag)
    {
        for (size_t i = 0; i < blockSize; i++)
        {
            gradient[0] += xMultTheta[i];
            for (size_t j = 0; j < nTheta; j++)
            {
                gradient[j + 1] += xMultTheta[i] * data[i * nTheta + j];
            }
        }
    }

    if (task.valueFlag)
    {
        for (size_t i = 0; i < blockSize; i++)
        {
            value[0] += xMultTheta[i] * xMultTheta[i];
        }
    }

    if (task.hessianFlag)
    {
        char uplo             = 'U';
        char notrans          = 'N';
        DAAL_INT argumentSize = dim + 1;

        Blas<algorithmFPType, cpu>::xsyrk(&uplo, &notrans, &dim, &n, &one, data, &dim, &one, hessian + argumentSize + 1, &argumentSize);

        for (size_t i = 0; i < blockSize; i++)
        {
            for (size_t j = 0; j < nTheta; j++)
            {
                hessian[j + 1] += data[i * nTheta + j];
            }
        }

        for (size_t i = 0; i < argumentSize; i++)
        {
            for (size_t j = 1; j < i; j++)
            {
                hessian[j * argumentSize + i] = hessian[i * argumentSize + j];
            }
            hessian[i * argumentSize] = hessian[i];
        }
    }
}

template <typename algorithmFPType, Method method, CpuType cpu>
void MSEKernel<algorithmFPType, method, cpu>::normalizeResults(MSETask<algorithmFPType, cpu> & task, algorithmFPType * value,
                                                               algorithmFPType * gradient, algorithmFPType * hessian)
{
    size_t argumentSize          = task.argumentSize;
    const algorithmFPType one    = 1.0;
    algorithmFPType batchSizeInv = (algorithmFPType)one / task.batchSize;
    if (task.valueFlag)
    {
        value[0] /= (algorithmFPType)(2 * task.batchSize);
    }

    if (task.gradientFlag)
    {
        for (size_t j = 0; j < argumentSize; j++)
        {
            gradient[j] *= batchSizeInv;
        }
    }

    if (task.hessianFlag)
    {
        hessian[0] = one;
        for (size_t j = 1; j < argumentSize * argumentSize; j++)
        {
            hessian[j] *= batchSizeInv;
        }
    }
}

} // namespace internal

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
