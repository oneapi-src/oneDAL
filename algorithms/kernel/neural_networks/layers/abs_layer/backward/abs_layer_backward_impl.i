/* file: abs_layer_backward_impl.i */
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
//  Implementation of abs layer.
//--
*/

#ifndef __ABS_LAYER_BACKWARD_IMPL_I__
#define __ABS_LAYER_BACKWARD_IMPL_I__

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace abs
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void AbsKernel<algorithmFPType, method, cpu>::compute(const abs::backward::Input *input, const daal::algorithms::Parameter *par,
                                                      abs::backward::Result *result)
{
    SharedPtr<Tensor> inputTable = input->get(layers::backward::inputGradient);
    SharedPtr<Tensor> dataTable = input->get(abs::auxData);

    SharedPtr<Tensor> resultTable = result->get(layers::backward::gradient);

    const services::Collection<size_t> &dims = inputTable->getDimensions();
    size_t nInputRows = dims[0];

    size_t nBlocks = nInputRows / _nRowsInBlock;
    nBlocks += (nBlocks * _nRowsInBlock != nInputRows);

    daal::threader_for(nBlocks, nBlocks, [ = ](int block)
    {
        size_t nRowsToProcess = _nRowsInBlock;
        if( block == nBlocks - 1 )
        {
            nRowsToProcess = nInputRows - block * _nRowsInBlock;
        }
        processBlock(inputTable, dataTable, block * _nRowsInBlock, nRowsToProcess, resultTable);

    } );
}

template<typename algorithmFPType, Method method, CpuType cpu>
inline void AbsKernel<algorithmFPType, method, cpu>::processBlock(SharedPtr<Tensor> inputTable,
                                                                  SharedPtr<Tensor> dataTable,
                                                                  size_t nProcessedRows, size_t nRowsInCurrentBlock,
                                                                  SharedPtr<Tensor> resultTable)
{
    SubtensorDescriptor<algorithmFPType> inputBlock;
    inputTable->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, inputBlock);
    algorithmFPType *inputArray = inputBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> forwardBlock;
    dataTable->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, readOnly, forwardBlock);
    algorithmFPType *dataArray = forwardBlock.getPtr();

    SubtensorDescriptor<algorithmFPType> resultBlock;
    resultTable->getSubtensor(0, 0, nProcessedRows, nRowsInCurrentBlock, writeOnly, resultBlock);
    algorithmFPType *resultArray = resultBlock.getPtr();

    size_t nDataElements = inputBlock.getSize();
    for(size_t i = 0; i < nDataElements; i++)
    {
        if(dataArray[i] > (algorithmFPType)0)
        {
            resultArray[i] = inputArray[i];
        }
        else if(dataArray[i] < (algorithmFPType)0)
        {
            resultArray[i] = -inputArray[i];
        }
        else
        {
            resultArray[i] = (algorithmFPType)0;
        }
    }

    inputTable->releaseSubtensor(inputBlock);
    dataTable->releaseSubtensor(forwardBlock);
    resultTable->releaseSubtensor(resultBlock);
}

} // internal
} // backward
} // namespace abs
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
