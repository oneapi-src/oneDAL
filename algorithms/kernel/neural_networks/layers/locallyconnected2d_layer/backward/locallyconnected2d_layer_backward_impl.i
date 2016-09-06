/* file: locallyconnected2d_layer_backward_impl.i */
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
//  Implementation of locallyconnected2d algorithm
//--
*/

using namespace daal::internal;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace locallyconnected2d
{
namespace backward
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
void LocallyConnected2dKernel<algorithmFPType, method, cpu>::compute(Tensor *inGradTensor, Tensor *gradientTensor, Tensor *auxDataTensor,
                                                        Tensor *auxWeightsTensor, Tensor *wDerTensor, Tensor *bDerTensor,
                                                        const locallyconnected2d::Parameter *parameter)
{
    if(inGradTensor == 0 || gradientTensor == 0 || auxDataTensor == 0 || auxWeightsTensor == 0 ||
        wDerTensor == 0 || bDerTensor == 0 ) { this->_errors->add(services::ErrorNullTensor); return; }

    const services::Collection<size_t> &inGradDims   = inGradTensor->getDimensions();
    const services::Collection<size_t> &auxDataDims  = auxDataTensor->getDimensions();

    size_t nAuxWeightsRows = auxWeightsTensor->getDimensions()[0];
    size_t nBDerRows       = bDerTensor->getDimensions()[0];

    MKL_INT l3 = inGradDims[2];
    MKL_INT l4 = inGradDims[3];

    size_t firstIdx  = parameter->indices.dims[0];
    size_t secondIdx = parameter->indices.dims[1];
    size_t groupDim  = parameter->groupDimension;
    size_t batchDim  = 6 - groupDim - firstIdx - secondIdx;

    const size_t dimsArray[4] = { batchDim, groupDim, firstIdx, secondIdx };

    size_t n1  = auxDataDims[batchDim];
    MKL_INT n2 = (MKL_INT)auxDataDims[groupDim];
    MKL_INT n3 = (MKL_INT)auxDataDims[firstIdx];
    MKL_INT n4 = (MKL_INT)auxDataDims[secondIdx];

    MKL_INT s3       = (MKL_INT)parameter->strides.size[0];
    MKL_INT s4       = (MKL_INT)parameter->strides.size[1];
    MKL_INT p3       = (MKL_INT)parameter->paddings.size[0];
    MKL_INT p4       = (MKL_INT)parameter->paddings.size[1];
    MKL_INT nKernels = (MKL_INT)parameter->nKernels;
    MKL_INT nGroups  = (MKL_INT)parameter->nGroups;
    MKL_INT m2       = n2 / nGroups;
    MKL_INT nk       = nKernels / nGroups;
    MKL_INT m3       = (MKL_INT)parameter->kernelSizes.size[0];
    MKL_INT m4       = (MKL_INT)parameter->kernelSizes.size[1];

    TensorOffsetLayout inputLayout = auxDataTensor->createDefaultSubtensorLayout();
    inputLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray));

    ReadSubtensor<algorithmFPType, cpu, Tensor> auxWeightsBlock(*auxWeightsTensor, 0, 0, 0, nAuxWeightsRows);
    const algorithmFPType *auxWeightsArray = auxWeightsBlock.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> gradientBlock;
    ReadSubtensor<algorithmFPType, cpu, Tensor> inGradBlock;
    ReadSubtensor<algorithmFPType, cpu, Tensor> auxDataBlock;

    WriteSubtensor<algorithmFPType, cpu, Tensor> wDerBlock(*wDerTensor, 0, 0, 0, nAuxWeightsRows);
    algorithmFPType *wDerArray = wDerBlock.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> bDerBlock(*bDerTensor, 0, 0, 0, nBDerRows);
    algorithmFPType *bDerArray = bDerBlock.get();

    const algorithmFPType zero = 0.0;
    const algorithmFPType one = 1.0;

    for(size_t i = 0; i < bDerTensor->getSize(); i++)
    {
        bDerArray[i] = zero;
    }

    for(size_t i = 0; i < wDerTensor->getSize(); i++)
    {
        wDerArray[i] = zero;
    }

    MKL_INT gradientIndex   = 0;
    MKL_INT inGradIndex     = 0;
    MKL_INT weightsIndex    = 0;
    MKL_INT biasIndex       = 0;
    MKL_INT auxDataIndex    = 0;
    MKL_INT index       = 0;
    MKL_INT wDerIndex       = 0;

    algorithmFPType divider = one / (algorithmFPType)n1;
    MKL_INT offsetBefore = nKernels * l3 * l4;
    MKL_INT d = 0;
    MKL_INT e = 0;

    /*
       gradientArray[t] [q * n2 / nGroups + c] [i] [j]
       inGradArray  [t] [q * nKernels / nGroups + r] [a] [b]
       auxWeightsArray  [q * nKernels / nGroups + r] [a] [b] [c] [i - a * s3 + p3] [j - b * s4 + p4]
    */
    for(size_t t = 0; t < n1; t++)
    {
        gradientBlock.set(*gradientTensor, 1, &t, 0, (size_t)n2);
        algorithmFPType *gradientArray = gradientBlock.get();

        inGradBlock.set(*inGradTensor, 1, &t, 0, (size_t)nKernels);
        const algorithmFPType *inGradArray = inGradBlock.get();

        auxDataBlock.set(*auxDataTensor, 1, &t, 0, (size_t)n2, inputLayout);
        const algorithmFPType *auxDataArray = auxDataBlock.get();

        for(MKL_INT q = 0; q < nGroups; q++)
        {
            for(MKL_INT c = 0; c < m2; c++)
            {
                for(MKL_INT i = 0; i < n3; i++)
                {
                    for(MKL_INT j = 0; j < n4; j++)
                    {
                        gradientIndex = j + (i + (c + q * m2) * n3) * n4;
                        gradientArray[gradientIndex] = zero;

                        for(MKL_INT r = 0; r < nk; r++)
                        {
                            for(MKL_INT a = _floor(i - m3 + p3, s3) + 1; a <= (i + p3) / s3; a++)
                            {
                                if(a >= 0 && a < l3)
                                {
                                    for(MKL_INT b = _floor(j - m4 + p4, s4) + 1; b <= (j + p4) / s4; b++)
                                    {
                                        if(b >= 0 && b < l4)
                                        {
                                            inGradIndex  = b + (a + (r + q * nk) * l3) * l4;
                                            weightsIndex = j - b * s4 + p4 + (i - a * s3 + p3 + (c + (b + (a + (r + q * nk) * l3) * l4) * m2) * m3) * m4;
                                            /* calculate gradient */
                                            gradientArray[gradientIndex] += inGradArray[inGradIndex] *  auxWeightsArray[weightsIndex];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            /*
               auxDataArray [t] [c + q * m2] [i * s3 - p3 + u] [j * s4 - p4 + v]
               inGradArray  [t] [r + q * nk] [i] [j]
               wDerArray        [r + q * nk] [i] [j] [c] [u] [v]
               bDerArray        [r + q * nk] [i] [j]
            */
            for(MKL_INT r = 0; r < nk; r++)
            {
                for(MKL_INT i = 0; i < l3; i++)
                {
                    for(MKL_INT j = 0; j < l4; j++)
                    {
                        index   = j + (i + (r + q * nk) * l3) * l4;
                        /* calculate biases derivatives */
                        bDerArray[index] += divider * inGradArray[index];

                        for(MKL_INT c = 0; c < m2; c++)
                        {
                            for(MKL_INT u = 0; u < m3; u++)
                            {
                                for(MKL_INT v = 0; v < m4; v++)
                                {
                                    wDerIndex = v + (u + (c + (j + (i + (r + q * nk) * l3) * l4) * m2) * m3) * m4;
                                    d = i * s3 - p3 + u;
                                    e = j * s4 - p4 + v;

                                    if(e >= 0 && d >= 0 && e < n4 && d < n3)
                                    {
                                        auxDataIndex = e + (d + (c + q * m2) * n3) * n4 ;
                                        /* calculate weights derivatives */
                                        wDerArray[wDerIndex] += divider * inGradArray[index] * auxDataArray[auxDataIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
MKL_INT LocallyConnected2dKernel<algorithmFPType, method, cpu>::_floor(MKL_INT numerator, MKL_INT denominator)
{
    double result = (double)numerator / (double)denominator;
    MKL_INT _result = (MKL_INT)result;

    return (result < (double)_result) ? (_result - 1) : _result;
}

} // internal
} // backward
} // namespace locallyconnected2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
