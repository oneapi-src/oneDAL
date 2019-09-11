/* file: lcn_layer_backward_impl.i */
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
//  Implementation of local contrast normalization algorithm
//--
*/

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lcn
{
namespace backward
{
namespace internal
{

using namespace daal::services;

/* TLS structure with local arrays and variables */
template<typename algorithmFPType, Method method, CpuType cpu>
struct Tls_data
{
    Status status;

    TensorPtr wDerTensor; /* only needed by conv compute, not lcn */
    TensorPtr bDerTensor; /* only needed by conv compute, not lcn */

    SharedPtr<HomogenTensor<algorithmFPType> > convResultTensor;
    SharedPtr<HomogenTensor<algorithmFPType> > convInGradTensor;

    /* Create backward convolution algorithm */
    Convolution2dKernel<algorithmFPType, neural_networks::layers::convolution2d::defaultDense, cpu> dconvKernel;

    Tls_data(size_t dataOffsetAfterDim, size_t firstKernelDim,
             size_t secondKrnelDim, size_t firstDim, size_t secondDim)
    {
        Collection<size_t> wDims;
        wDims << 1 << 1 << firstKernelDim << secondKrnelDim;

        Collection<size_t> bDims;
        bDims << 1;

        Collection<size_t> convInpGradDims;
        convInpGradDims << 1 << 1 << firstDim << secondDim;

        wDerTensor       = HomogenTensor<algorithmFPType>::create(wDims, TensorIface::doAllocate, &status);
        bDerTensor       = HomogenTensor<algorithmFPType>::create(bDims, TensorIface::doAllocate, &status);
        convResultTensor = HomogenTensor<algorithmFPType>::create(convInpGradDims, TensorIface::doAllocate, &status);
        convInGradTensor = HomogenTensor<algorithmFPType>::create(convInpGradDims, TensorIface::doAllocate, &status);

        status|= checkTensor(wDerTensor.get(), "wDerTensor" );
        status|= checkTensor(bDerTensor.get(), "bDerTensor" );
        status|= checkTensor(convResultTensor.get(), "convResultTensor" );
        status|= checkTensor(convInGradTensor.get(), "convInGradTensor" );

        status|= dconvKernel.initialize();
    }

    ~Tls_data()
    {
        dconvKernel.reset();
    }
};

template<typename algorithmFPType, Method method, CpuType cpu>
Status LCNKernel<algorithmFPType, method, cpu>::initialize(const Tensor &auxCenteredDataTensor, const Tensor &auxSigmaTensor, const Tensor &auxCTensor,
                                                         const Tensor &kernelTensor, const lcn::Parameter &parameter)
{
    const Collection<size_t> &initialDataDims = auxCenteredDataTensor.getDimensions();
    const Collection<size_t> &cDims           = auxCTensor.getDimensions();
    sigmaDims  = auxSigmaTensor.getDimensions();
    kernelDims = kernelTensor.getDimensions();

    nDataRows   = initialDataDims[0];
    nSigmaRows  = sigmaDims[0];
    nCRows      = cDims[0];
    nKernelRows = kernelDims[0];

    nDataElements   = auxCenteredDataTensor.getSize();
    nKernelElements = kernelTensor.getSize();
    nCElements      = auxCTensor.getSize();
    nDims = initialDataDims.size();

    initialFirstDim  = parameter.indices.dims[0];
    initialSecondDim = parameter.indices.dims[1];

    sigmaThreshold = parameter.sigmaDegenerateCasesThreshold;

    initialSumDimension = 1;
    if(parameter.sumDimension)
    {
        fDimN = 1;
        ReadRows<int, cpu, NumericTable> dimBlock(*parameter.sumDimension, 0, 1);
        DAAL_CHECK_BLOCK_STATUS(dimBlock);
        const int *dimArray = dimBlock.get();
        initialSumDimension = (size_t)dimArray[0];
    }

    /* Get dims collection of repacked data tensor */
    batchDimension = 6 - initialSumDimension - initialFirstDim - initialSecondDim; /* Calculate 4th dimension index. 6 here is a sum of all indexes: 0 + 1 + 2 + 3 */
    dataDims << initialDataDims[batchDimension] << initialDataDims[initialSumDimension] << initialDataDims[initialFirstDim] << initialDataDims[initialSecondDim];

    sumDimension = (size_t)1;
    firstDim     = (size_t)2;
    secondDim    = (size_t)3;

    if(!parameter.sumDimension)
    {
        fDimN = 2;
        dataDims[0] *= dataDims[sumDimension];
        dataDims[sumDimension] = 1;
    }

    dataOffsetBeforeDim = dataDims[0];
    dataOffsetAfterDim  = dataDims[firstDim] * dataDims[secondDim];

    /* Set convolution algorithm parameters */
    convParameter.indices.dims[0] = firstDim;
    convParameter.indices.dims[1] = secondDim;
    convParameter.nGroups = 1;
    convParameter.strides.size[0] = 1;
    convParameter.strides.size[1] = 1;
    convParameter.groupDimension = 1;
    convParameter.nKernels = 1;
    convParameter.kernelSizes.size[0] = kernelDims[0];
    convParameter.kernelSizes.size[1] = kernelDims[1];
    convParameter.paddings.size[0] = kernelDims[0] / 2;
    convParameter.paddings.size[1] = kernelDims[1] / 2;

    return Status();
}

/*  step_1:   g_5   = inputGradient * auxInvMax;
    step_2:   g_13  = sum_sumDimension( inputGradient * auxCenteredData ) * pow(auxInvMax, 2);
    step_3:   g_12  = g_13 * (1 - q) = g_13 - g_10 = step_2 * (1 - q);
    step_4:   g_10  = step_2 * q;
    step_5:   g_8  = (g_10 + g_11) / auxSigma = (g_10 + 1/M * g_12) / auxSigma = (step_3 + 1/M * step_4) / ( auxSigma + e );
    step_6:   g_7   = dconv(g_8) = dconv(step_5);
    step_7:   g_4   = g_5 + g_6 = g_5 + g_7 * auxCenteredData = step_1 + step_6 * auxCenteredData;
    step_8:   g_3   = sum_sumDimension(g_4) = sum_sumDimension(step_7);
    step_9:   g_1   = dconv(g_3) = dconv(step_8);
    step_10:  gradient = g_2 - g_1 = g_4 - g_1 = step_7 - step_9.
*/
template<typename algorithmFPType, Method method, CpuType cpu>
Status LCNKernel<algorithmFPType, method, cpu>::compute(const Tensor &auxCenteredDataTensor, const Tensor &auxSigmaTensor, const Tensor &auxCTensor,
                                                      const Tensor &auxInvMaxTensor, const Tensor &kernelTensor, const Tensor &inGradTensor,
                                                      Tensor &gradientTensor, const lcn::Parameter &parameter)
{
    Status s;

    const algorithmFPType one  = 1.0;
    const algorithmFPType zero = 0.0;

    Collection<size_t> dimsOrder;
    dimsOrder << batchDimension << initialSumDimension << initialFirstDim << initialSecondDim;

    TensorOffsetLayout cdLayout = auxCenteredDataTensor.createDefaultSubtensorLayout();
    DAAL_CHECK_STATUS(s, cdLayout.shuffleDimensions(dimsOrder));

    TensorOffsetLayout inGradLayout = inGradTensor.createDefaultSubtensorLayout();
    DAAL_CHECK_STATUS(s, inGradLayout.shuffleDimensions(dimsOrder));

    TensorOffsetLayout gradientLayout = gradientTensor.createDefaultSubtensorLayout();
    DAAL_CHECK_STATUS(s, gradientLayout.shuffleDimensions(dimsOrder));

    ReadSubtensor<algorithmFPType, cpu, Tensor> cBlock(const_cast<Tensor &>(auxCTensor), 0, 0, 0, nCRows);
    DAAL_CHECK_BLOCK_STATUS(cBlock);
    const algorithmFPType *auxCArray = cBlock.get();

    ReadSubtensor<algorithmFPType, cpu, Tensor> kernelBlock(const_cast<Tensor &>(kernelTensor), 0, 0, 0, nKernelRows);
    DAAL_CHECK_BLOCK_STATUS(kernelBlock);
    const algorithmFPType *kernelArray = kernelBlock.get();

    algorithmFPType divider = one / dataOffsetAfterDim;

    /* Allocate arrays needed for computations */
    TArray<algorithmFPType, cpu> tempArrayOfCSizeBlock(nCElements);
    algorithmFPType *tempArrayOfCSize = tempArrayOfCSizeBlock.get();
    DAAL_CHECK_MALLOC(tempArrayOfCSize);

    TArray<algorithmFPType, cpu> weightsBlock(nKernelElements);
    algorithmFPType *weightsArray = weightsBlock.get();
    DAAL_CHECK_MALLOC(weightsArray);

    TArray<size_t, cpu> fDimsBlock(fDimN);
    size_t *fDims = fDimsBlock.get();
    DAAL_CHECK_MALLOC(fDims);

    /* Compute multiplier to normalize through sumDimension */
    algorithmFPType multiplier = one / dataDims[sumDimension];
    /* Get weightsArray needed for convolution computation */
    for(size_t j = 0; j < nKernelElements; j++)
    {
        weightsArray[j] = kernelArray[j] * multiplier;
    }

    Collection<size_t> wDims;
    wDims << 1 << 1 << kernelDims[0] << kernelDims[1];

    /* Tensors needed for convolution */
    TensorPtr weightsTensor = HomogenTensor<algorithmFPType>::create(wDims, weightsArray, &s);
    DAAL_CHECK_STATUS_VAR(s);

    /* TLS data initialization */
    daal::tls<Tls_data<algorithmFPType, method, cpu> *> tls_data([ & ]()
    {
        return new Tls_data<algorithmFPType, method, cpu>(dataOffsetAfterDim, kernelDims[0], kernelDims[1], dataDims[firstDim], dataDims[secondDim]);
    });

    __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&inGradTensor))
    __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&auxCenteredDataTensor))
    __DAAL_MAKE_TENSOR_THREADSAFE(&gradientTensor)
    __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&auxSigmaTensor))
    __DAAL_MAKE_TENSOR_THREADSAFE(const_cast<Tensor *>(&auxInvMaxTensor))

    SafeStatus safeStat;

    daal::threader_for(dataOffsetBeforeDim, dataOffsetBeforeDim, [ & ](int i)
    {
        Tls_data<algorithmFPType, method, cpu> *tls_data_local = tls_data.local();
        safeStat |= tls_data_local->status;
        if(!tls_data_local->status) return;

        algorithmFPType *gSqTempArray    = tls_data_local->convInGradTensor->getArray();
        algorithmFPType *convResultArray = tls_data_local->convResultTensor->getArray();

        algorithmFPType gConvTempValue;
        size_t dataIndex, sigmaIndex;

        getFixedDimsIndexes(fDims, i);

        ReadSubtensor<algorithmFPType, cpu, Tensor> inGradBlock(const_cast<Tensor &>(inGradTensor), fDimN, fDims, 0, dataDims[fDimN], inGradLayout);
        DAAL_CHECK_BLOCK_STATUS_THR(inGradBlock);
        const algorithmFPType *inGradArray = inGradBlock.get();

        ReadSubtensor<algorithmFPType, cpu, Tensor> cdBlock(const_cast<Tensor &>(auxCenteredDataTensor), fDimN, fDims, 0, dataDims[fDimN], cdLayout);
        DAAL_CHECK_BLOCK_STATUS_THR(cdBlock);
        const algorithmFPType *auxCDArray = cdBlock.get();

        WriteSubtensor<algorithmFPType, cpu, Tensor> gradientBlock(gradientTensor, fDimN, fDims, 0, dataDims[fDimN], gradientLayout);
        DAAL_CHECK_BLOCK_STATUS_THR(gradientBlock);
        algorithmFPType *gradientArray = gradientBlock.get();

        ReadSubtensor<algorithmFPType, cpu, Tensor> sigmaBlock(const_cast<Tensor &>(auxSigmaTensor), fDimN, fDims, 0, sigmaDims[fDimN]);
        DAAL_CHECK_BLOCK_STATUS_THR(sigmaBlock);
        const algorithmFPType *auxSigmaArray = sigmaBlock.get();

        ReadSubtensor<algorithmFPType, cpu, Tensor> invMaxBlock(const_cast<Tensor &>(auxInvMaxTensor), fDimN, fDims, 0, sigmaDims[fDimN]);
        DAAL_CHECK_BLOCK_STATUS_THR(invMaxBlock);
        const algorithmFPType *auxInvMaxArray = invMaxBlock.get();

        tempArrayOfCSize[i] = zero;
        for(size_t k = 0; k < dataOffsetAfterDim; k++)
        {
            gSqTempArray[k] = zero;
        }

        for(size_t j = 0; j < dataDims[sumDimension]; j++)
        {
            size_t j_shift = j * dataOffsetAfterDim;
            for(size_t k = 0; k < dataOffsetAfterDim; k++)
            {
                dataIndex  = k + j_shift;
                /* step_1:   g_5 = inputGradient * auxInvMax */
                gradientArray[dataIndex] = inGradArray[dataIndex] * auxInvMaxArray[k];

                /* step_2:   g_13  = sum_sumDimension( inputGradient * auxCenteredData ) * pow(auxInvMax, 2) */
                gSqTempArray[k] -= inGradArray[dataIndex] * auxCDArray[dataIndex] * auxInvMaxArray[k] * auxInvMaxArray[k];
            }
        }
        invMaxBlock.release();
        inGradBlock.release();

        for(size_t k = 0; k < dataOffsetAfterDim; k++)
        {
            /* step_3:   g_12  = g_13 * (1 - q) = g_13 - g_10 = step_2 * (1 - q) */
            tempArrayOfCSize[i] += gSqTempArray[k] * (one - (auxSigmaArray[k] > auxCArray[i]) );
        }

        for(size_t k = 0; k < dataOffsetAfterDim; k++)
        {
            /* step_4:   g_10  = step_2 * q  */
            gConvTempValue = gSqTempArray[k] * (auxSigmaArray[k] > auxCArray[i]);

            /* step_5:  g_8  = (g_10 + g_11) / auxSigma = (g_10 + 1/M * g_12) / auxSigma = (step_3 + 1/M * step_4) / ( auxSigma + e ) */
            gSqTempArray[k] = (gConvTempValue + divider * tempArrayOfCSize[i]) / (auxSigmaArray[k] + sigmaThreshold);
        }

        sigmaBlock.release();

        Status s;

        /* step_6:  g_7  = dconv(g_8) = dconv(step_5) */
        /* convResultTensor first time is used here as auxData for wDer and bDer not needed calculation, second time as conv result, needed for lcn */
        s = tls_data_local->dconvKernel.compute(tls_data_local->convInGradTensor.get(), tls_data_local->convResultTensor.get(), weightsTensor.get(), convParameter,
                            tls_data_local->wDerTensor.get(), tls_data_local->bDerTensor.get(), tls_data_local->convResultTensor.get());
        safeStat |= s; if(!s) return;

        for(size_t k = 0; k < dataOffsetAfterDim; k++)
        {
            gSqTempArray[k] = zero;
        }

        for(size_t j = 0; j < dataDims[sumDimension]; j++)
        {
            size_t j_shift = j * dataOffsetAfterDim;
            for(size_t k = 0; k < dataOffsetAfterDim; k++)
            {
                dataIndex  = k + j_shift;
                /* step_7:  g_4   = g_5 + g_6 = g_5 + g_7 * auxCenteredData = step_1 + step_6 * auxCenteredData */
                gradientArray[dataIndex] += convResultArray[k] * auxCDArray[dataIndex];

                /* step_8:  g_3   = sum_sumDimension(g_4) = sum_sumDimension(step_7) */
                gSqTempArray[k] += gradientArray[dataIndex];
            }
        }

        cdBlock.release();

        /* step_9:  g_1   = dconv(g_3) = dconv(step_8) */
        s = tls_data_local->dconvKernel.compute(tls_data_local->convInGradTensor.get(), tls_data_local->convResultTensor.get(), weightsTensor.get(), convParameter,
                            tls_data_local->wDerTensor.get(), tls_data_local->bDerTensor.get(), tls_data_local->convResultTensor.get());
        safeStat |= s; if(!s) return;

        for(size_t j = 0; j < dataDims[sumDimension]; j++)
        {
            size_t j_shift = j * dataOffsetAfterDim;
            for(size_t k = 0; k < dataOffsetAfterDim; k++)
            {
                size_t dataIndex  = k + j_shift;
                /* step_10:  gradient = g_2 - g_1 = g_4 - g_1 = step_7 - step_9 */
                gradientArray[dataIndex] -= convResultArray[k];
            }
        }
    });

    tls_data.reduce( [ & ]( Tls_data<algorithmFPType, method, cpu>* tls_data_local )
    {
        delete tls_data_local;
    } );
    return safeStat.detach();
}

template<typename algorithmFPType, Method method, CpuType cpu>
void LCNKernel<algorithmFPType, method, cpu>::getFixedDimsIndexes(size_t *fDims, size_t i)
{
    if(fDimN == 1)
    {
        fDims[0] = i;
    }
    else
    {
        size_t offsetAfter = sigmaDims[fDimN - 1];

        /* Get last fixed dim index as the remainder of the division */
        fDims[fDimN - 1] = i % sigmaDims[fDimN - 1];

        /* Count indexes starting from the penultimate element of the fDims[] array*/
        for(size_t j = fDimN - 1; j > 0; j--)
        {
            size_t totalOffset = offsetAfter * sigmaDims[j - 1];
            size_t nTimes = i / totalOffset;

            fDims[j - 1] = (i - totalOffset * nTimes) / offsetAfter;

            offsetAfter *= sigmaDims[j - 1];
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status LCNKernel<algorithmFPType, method, cpu>::reset()
{
    dataDims.clear();
    kernelDims.clear();
    sigmaDims.clear();
    return Status();
}

} // internal
} // backward
} // namespace lcn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
