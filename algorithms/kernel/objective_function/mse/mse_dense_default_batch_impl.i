/* file: mse_dense_default_batch_impl.i */
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
//  Implementation of mse algorithm
//--
*/

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
template<typename algorithmFPType, Method method, CpuType cpu>
inline void MSEKernel<algorithmFPType, method, cpu>::compute(NumericTable *dataNT, NumericTable *dependentVariablesNT, NumericTable *argumentNT,
                                                             NumericTable *valueNT, NumericTable *hessianNT, NumericTable *gradientNT, Parameter *parameter)
{
    MSETask<algorithmFPType, cpu> *task = NULL;
    algorithmFPType *argumentArray = NULL, *dataBlock = NULL, *dependentVariablesBlock = NULL;
    algorithmFPType *value = NULL, *gradient = NULL, *hessian = NULL;

    size_t nDataRows = dataNT->getNumberOfRows();
    if(parameter->batchIndices.get() != NULL && parameter->batchIndices->getNumberOfColumns() != nDataRows)
    {
        task = new MSETaskSample<algorithmFPType, cpu>(dataNT, dependentVariablesNT, argumentNT, valueNT, hessianNT, gradientNT, parameter, blockSizeDefault, &argumentArray);
    }
    else
    {
        task = new MSETaskAll<algorithmFPType, cpu>(dataNT, dependentVariablesNT, argumentNT, valueNT, hessianNT, gradientNT, parameter, blockSizeDefault, &argumentArray);
    }
    if(task->error.id() != NoErrorMessageFound) {this->_errors->add(task->error.id()); return;}

    task->getResultValues(&value, &gradient, &hessian);
    if(task->error.id() != NoErrorMessageFound) {this->_errors->add(task->error.id()); return;}

    task->setResultValuesToZero(&value, &gradient, &hessian);

    size_t blockSize = blockSizeDefault;
    size_t nBlocks = task->batchSize / blockSizeDefault;
    nBlocks += (nBlocks * blockSizeDefault != task->batchSize);
    if(nBlocks == 1) { blockSize = task->batchSize; }

    for(size_t block = 0; block < nBlocks; block++)
    {
        if( block == nBlocks - 1 )
        {
            blockSize = task->batchSize - block * blockSizeDefault;
        }

        task->getCurrentBlock(block * blockSizeDefault, blockSize, &dataBlock, &dependentVariablesBlock);
        if(task->error.id() != NoErrorMessageFound) {this->_errors->add(task->error.id()); return;}

        computeMSE(blockSize, task, dataBlock, argumentArray, dependentVariablesBlock, value, gradient, hessian);

        task->releaseCurrentBlock();
    }

    normalizeResults(task, value, gradient, hessian);

    task->releaseResultValues();

    delete task;

    return;
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void MSEKernel<algorithmFPType, method, cpu>::computeMSE(
    size_t blockSize,
    MSETask<algorithmFPType, cpu> *task,
    algorithmFPType *data,
    algorithmFPType *argumentArray,
    algorithmFPType *dependentVariablesArray,
    algorithmFPType *value,
    algorithmFPType *gradient,
    algorithmFPType *hessian)
{
    char trans = 'T';
    algorithmFPType one = 1.0;
    algorithmFPType zero = 0.0;
    MKL_INT n   = (MKL_INT)blockSize;
    size_t nTheta = task->nTheta;
    MKL_INT dim = (MKL_INT)nTheta;
    MKL_INT ione = 1;
    algorithmFPType theta0 = argumentArray[0];
    algorithmFPType *theta = &argumentArray[1];
    algorithmFPType *xMultTheta = task->xMultTheta;

    if (task->gradientFlag || task->valueFlag)
    {
        Blas<algorithmFPType, cpu>::xgemv(&trans, &dim, &n, &one, data, &dim, theta, &ione, &zero, xMultTheta, &ione);

        for(size_t i = 0; i < blockSize; i++)
        {
            xMultTheta[i] = xMultTheta[i] + theta0 - dependentVariablesArray[i];
        }
    }

    if (task->gradientFlag)
    {
        for(size_t i = 0; i < blockSize; i++)
        {
            gradient[0] += xMultTheta[i];
            for(size_t j = 0; j < nTheta; j++)
            {
                gradient[j + 1] += xMultTheta[i] * data[i * nTheta + j];
            }
        }
    }

    if (task->valueFlag)
    {
        for(size_t i = 0; i < blockSize; i++)
        {
            value[0] += xMultTheta[i] * xMultTheta[i];
        }
    }

    if (task->hessianFlag)
    {
        char uplo  = 'U';
        char notrans = 'N';
        MKL_INT nFeatures = dim + 1;

        Blas<algorithmFPType, cpu>::xsyrk(&uplo, &notrans, &dim, &n, &one, data, &dim, &one, hessian + nFeatures + 1, &nFeatures);

        for (size_t i = 0; i < blockSize; i++)
        {
            for (size_t j = 0; j < nTheta; j++)
            {
                hessian[j + 1] += data[i * nTheta + j];
            }
        }

        for (size_t i = 0; i < nFeatures; i++)
        {
            for (size_t j = 1; j < i; j++)
            {
                hessian[j * nFeatures + i] = hessian[i * nFeatures + j];
            }
            hessian[i * nFeatures] = hessian[i];
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void MSEKernel<algorithmFPType, method, cpu>::normalizeResults(
    MSETask<algorithmFPType, cpu> *task,
    algorithmFPType *value,
    algorithmFPType *gradient,
    algorithmFPType *hessian
)
{
    size_t nFeatures = task->nFeatures;
    const algorithmFPType one = 1.0;
    algorithmFPType batchSizeInv = (algorithmFPType)one / task->batchSize;
    if (task->valueFlag)
    {
        value[0] /= (algorithmFPType)(2 * task->batchSize);
    }

    if (task->gradientFlag)
    {
        for(size_t j = 0; j < nFeatures; j++)
        {
            gradient[j] *= batchSizeInv;
        }
    }

    if (task->hessianFlag)
    {
        hessian[0] = one;
        for(size_t j = 1; j < nFeatures * nFeatures; j++)
        {
            hessian[j] *= batchSizeInv;
        }
    }
}

} // namespace daal::internal

} // namespace mse

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
