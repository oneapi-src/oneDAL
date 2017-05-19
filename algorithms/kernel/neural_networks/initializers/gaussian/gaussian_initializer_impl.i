/* file: gaussian_initializer_impl.i */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
//  Implementation of gaussian algorithm
//--
*/

#ifndef __GAUSSIAN_INITIALIZER_IMPL_I__
#define __GAUSSIAN_INITIALIZER_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace initializers
{
namespace gaussian
{
namespace internal
{

template<typename algorithmFPType, Method method, CpuType cpu>
Status GaussianKernel<algorithmFPType, method, cpu>::compute(const gaussian::Parameter *parameter, Tensor *resultTensor)
{
    BaseRNGs<cpu> baseRng(parameter->seed);
    RNGs<algorithmFPType, cpu> rng;

    WriteOnlySubtensor<algorithmFPType, cpu, Tensor> resultSubtensor(resultTensor, 0, 0, 0, resultTensor->getDimensions()[0]);
    DAAL_CHECK_BLOCK_STATUS(resultSubtensor);
    algorithmFPType *resultArray = resultSubtensor.get();

    size_t size = resultTensor->getSize();

    algorithmFPType a = (algorithmFPType)(parameter->a);
    algorithmFPType sigma = (algorithmFPType)(parameter->sigma);

    DAAL_CHECK(!rng.gaussian(size, resultArray, baseRng, a, sigma), ErrorIncorrectErrorcodeFromGenerator);
    return Status();
}

} // internal
} // namespace gaussian
} // namespace initializers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
