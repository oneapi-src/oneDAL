/* file: locallyconnected2d_layer_backward_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
services::Status LocallyConnected2dKernel<algorithmFPType, method, cpu>::compute(const Tensor &inGradTensor, Tensor &gradientTensor, Tensor &auxDataTensor,
                                                        Tensor &auxWeightsTensor, Tensor &wDerTensor, Tensor &bDerTensor,
                                                        const locallyconnected2d::Parameter &parameter)
{
    Status s;

    const services::Collection<size_t> &inGradDims   = inGradTensor.getDimensions();
    const services::Collection<size_t> &auxDataDims  = auxDataTensor.getDimensions();

    size_t nAuxWeightsRows = auxWeightsTensor.getDimensions()[0];
    size_t nBDerRows       = bDerTensor.getDimensions()[0];

    DAAL_INT l3 = inGradDims[2];
    DAAL_INT l4 = inGradDims[3];

    size_t firstIdx  = parameter.indices.dims[0];
    size_t secondIdx = parameter.indices.dims[1];
    size_t groupDim  = parameter.groupDimension;
    size_t batchDim  = 6 - groupDim - firstIdx - secondIdx;

    const size_t dimsArray[4] = { batchDim, groupDim, firstIdx, secondIdx };

    size_t n1  = auxDataDims[batchDim];
    DAAL_INT n2 = (DAAL_INT)auxDataDims[groupDim];
    DAAL_INT n3 = (DAAL_INT)auxDataDims[firstIdx];
    DAAL_INT n4 = (DAAL_INT)auxDataDims[secondIdx];

    DAAL_INT s3       = (DAAL_INT)parameter.strides.size[0];
    DAAL_INT s4       = (DAAL_INT)parameter.strides.size[1];
    DAAL_INT p3       = (DAAL_INT)parameter.paddings.size[0];
    DAAL_INT p4       = (DAAL_INT)parameter.paddings.size[1];
    DAAL_INT nKernels = (DAAL_INT)parameter.nKernels;
    DAAL_INT nGroups  = (DAAL_INT)parameter.nGroups;
    DAAL_INT m2       = n2 / nGroups;
    DAAL_INT nk       = nKernels / nGroups;
    DAAL_INT m3       = (DAAL_INT)parameter.kernelSizes.size[0];
    DAAL_INT m4       = (DAAL_INT)parameter.kernelSizes.size[1];

    TensorOffsetLayout inputLayout = auxDataTensor.createDefaultSubtensorLayout();
    DAAL_CHECK_STATUS(s, inputLayout.shuffleDimensions(services::Collection<size_t>( 4, dimsArray)));

    ReadSubtensor<algorithmFPType, cpu, Tensor> auxWeightsBlock;
    algorithmFPType *auxWeightsArray;

    WriteSubtensor<algorithmFPType, cpu, Tensor> gradientBlock;
    algorithmFPType *gradientArray;
    ReadSubtensor<algorithmFPType, cpu, Tensor> inGradBlock;
    ReadSubtensor<algorithmFPType, cpu, Tensor> auxDataBlock;

    WriteSubtensor<algorithmFPType, cpu, Tensor> wDerBlock(wDerTensor, 0, 0, 0, nAuxWeightsRows);
    DAAL_CHECK_BLOCK_STATUS(wDerBlock);
    algorithmFPType *wDerArray = wDerBlock.get();

    WriteSubtensor<algorithmFPType, cpu, Tensor> bDerBlock(bDerTensor, 0, 0, 0, nBDerRows);
    DAAL_CHECK_BLOCK_STATUS(bDerBlock);
    algorithmFPType *bDerArray = bDerBlock.get();

    const algorithmFPType zero = 0.0;
    const algorithmFPType one = 1.0;

    for(size_t i = 0; i < bDerTensor.getSize(); i++)
    {
        bDerArray[i] = zero;
    }

    for(size_t i = 0; i < wDerTensor.getSize(); i++)
    {
        wDerArray[i] = zero;
    }

    DAAL_INT gradientIndex   = 0;
    DAAL_INT inGradIndex     = 0;
    DAAL_INT weightsIndex    = 0;
    DAAL_INT biasIndex       = 0;
    DAAL_INT auxDataIndex    = 0;
    DAAL_INT index           = 0;
    DAAL_INT wDerIndex       = 0;

    algorithmFPType divider = one / (algorithmFPType)n1;
    DAAL_INT offsetBefore = nKernels * l3 * l4;
    DAAL_INT d = 0;
    DAAL_INT e = 0;

    if (parameter.propagateGradient)
    {
        auxWeightsBlock.set(auxWeightsTensor, 0, 0, 0, nAuxWeightsRows);
        DAAL_CHECK_BLOCK_STATUS(auxWeightsBlock);
        auxWeightsArray = const_cast<algorithmFPType *>(auxWeightsBlock.get());
    }
    /*
       gradientArray[t] [q * n2 / nGroups + c] [i] [j]
       inGradArray  [t] [q * nKernels / nGroups + r] [a] [b]
       auxWeightsArray  [q * nKernels / nGroups + r] [a] [b] [c] [i - a * s3 + p3] [j - b * s4 + p4]
    */
    for(size_t t = 0; t < n1; t++)
    {

        inGradBlock.set(const_cast<Tensor &>(inGradTensor), 1, &t, 0, (size_t)nKernels);
        DAAL_CHECK_BLOCK_STATUS(inGradBlock);
        const algorithmFPType *inGradArray = inGradBlock.get();

        auxDataBlock.set(auxDataTensor, 1, &t, 0, (size_t)n2, inputLayout);
        DAAL_CHECK_BLOCK_STATUS(auxDataBlock);
        const algorithmFPType *auxDataArray = auxDataBlock.get();

        if (parameter.propagateGradient)
        {
            gradientBlock.set(gradientTensor, 1, &t, 0, (size_t)n2);
            DAAL_CHECK_BLOCK_STATUS(gradientBlock);
            gradientArray = gradientBlock.get();

            for(DAAL_INT q = 0; q < nGroups; q++)
            {
                for(DAAL_INT c = 0; c < m2; c++)
                {
                    for(DAAL_INT i = 0; i < n3; i++)
                    {
                        for(DAAL_INT j = 0; j < n4; j++)
                        {
                            gradientIndex = j + (i + (c + q * m2) * n3) * n4;
                            gradientArray[gradientIndex] = zero;

                            for(DAAL_INT r = 0; r < nk; r++)
                            {
                                for(DAAL_INT a = _floor(i - m3 + p3, s3) + 1; a <= (i + p3) / s3; a++)
                                {
                                    if(a >= 0 && a < l3)
                                    {
                                        for(DAAL_INT b = _floor(j - m4 + p4, s4) + 1; b <= (j + p4) / s4; b++)
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
            }
        }

        for(DAAL_INT q = 0; q < nGroups; q++)
        {
            /*
               auxDataArray [t] [c + q * m2] [i * s3 - p3 + u] [j * s4 - p4 + v]
               inGradArray  [t] [r + q * nk] [i] [j]
               wDerArray        [r + q * nk] [i] [j] [c] [u] [v]
               bDerArray        [r + q * nk] [i] [j]
            */
            for(DAAL_INT r = 0; r < nk; r++)
            {
                for(DAAL_INT i = 0; i < l3; i++)
                {
                    for(DAAL_INT j = 0; j < l4; j++)
                    {
                        index   = j + (i + (r + q * nk) * l3) * l4;
                        /* calculate biases derivatives */
                        bDerArray[index] += divider * inGradArray[index];

                        for(DAAL_INT c = 0; c < m2; c++)
                        {
                            for(DAAL_INT u = 0; u < m3; u++)
                            {
                                for(DAAL_INT v = 0; v < m4; v++)
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
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
DAAL_INT LocallyConnected2dKernel<algorithmFPType, method, cpu>::_floor(DAAL_INT numerator, DAAL_INT denominator)
{
    double result = (double)numerator / (double)denominator;
    DAAL_INT _result = (DAAL_INT)result;

    return (result < (double)_result) ? (_result - 1) : _result;
}

} // internal
} // backward
} // namespace locallyconnected2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
