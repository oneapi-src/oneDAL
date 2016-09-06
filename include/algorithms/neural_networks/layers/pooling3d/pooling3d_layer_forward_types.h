/* file: pooling3d_layer_forward_types.h */
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
//  Implementation of the forward 3D pooling layer types.
//--
*/

#ifndef __POOLING3D_LAYER_FORWARD_TYPES_H__
#define __POOLING3D_LAYER_FORWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/homogen_tensor.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"
#include "algorithms/neural_networks/layers/pooling3d/pooling3d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace pooling3d
{
/**
 * @defgroup pooling3d_forward Forward Three-dimensional Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::pooling3d::forward
 * @ingroup pooling3d
 * @{
 */
/**
 * \brief Contains classes for the forward one-dimensional (3D) pooling layer
 */
namespace forward
{
/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__FORWARD__INPUT"></a>
 * \brief %Input objects for the forward 3D pooling layer
 */
class Input : public layers::forward::Input
{
public:
    /**
     * Default constructor
     */
    Input() : layers::forward::Input() {}

    virtual ~Input() {}

    /**
     * Allocates memory to store input objects of forward 3D pooling layer
     * \param[in] parameter %Parameter of forward 3D pooling layer
     * \param[in] method    Computation method for the layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Parameter *parameter, const int method) {}

    /**
     * Returns dimensions of weights tensor
     * \return Dimensions of weights tensor
     */
    virtual const services::Collection<size_t> getWeightsSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        return services::Collection<size_t>();
    }

    /**
     * Returns dimensions of biases tensor
     * \return Dimensions of biases tensor
     */
    virtual const services::Collection<size_t> getBiasesSizes(const layers::Parameter *parameter) const DAAL_C11_OVERRIDE
    {
        return services::Collection<size_t>();
    }

    /**
     * Checks an input object for the 3D pooling layer
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method of the layer
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        layers::forward::Input::check(parameter, method);
        const Parameter *param = static_cast<const Parameter *>(parameter);
        services::SharedPtr<data_management::Tensor> dataTensor = get(layers::forward::data);
        const services::Collection<size_t> &dataDims = dataTensor->getDimensions();
        size_t nDim = dataDims.size();
        const size_t *spatialDimensions = param->indices.size;
        for (size_t i = 0; i < 3; i++)
        {
            if (spatialDimensions[i] > nDim - 1)
            {
                services::SharedPtr<services::Error> error(new services::Error());
                error->setId(services::ErrorIncorrectParameter);
                error->addStringDetail(services::ArgumentName, "indices");
                this->_errors->add(error);
                return;
            }

            size_t kernelSize = param->kernelSizes.size[i];
            if (kernelSize == 0 || kernelSize > dataDims[spatialDimensions[i]] + 2 * param->paddings.size[i])
            {
                services::SharedPtr<services::Error> error(new services::Error());
                error->setId(services::ErrorIncorrectParameter);
                error->addStringDetail(services::ArgumentName, "kernelSize");
                this->_errors->add(error);
                return;
            }
        }

        if (spatialDimensions[0] == spatialDimensions[1] || spatialDimensions[0] == spatialDimensions[2] ||
            spatialDimensions[1] == spatialDimensions[2])
        {
            services::SharedPtr<services::Error> error(new services::Error());
            error->setId(services::ErrorIncorrectParameter);
            error->addStringDetail(services::ArgumentName, "indices");
            this->_errors->add(error);
            return;
        }
    }
};


/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__POOLING3D__FORWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the forward 3D pooling layer
 */
class Result : public layers::forward::Result
{
public:
    /** Default constructor */
    Result() {}
    virtual ~Result() {}

    /**
     * Returns dimensions of value tensor
     * \return Dimensions of value tensor
     */
    virtual const services::Collection<size_t> getValueSize(const services::Collection<size_t> &inputSize,
                                                            const daal::algorithms::Parameter *par, const int method) const DAAL_C11_OVERRIDE
    {
        services::Collection<size_t> valueDims(inputSize);
        computeValueDimensions(valueDims, static_cast<const Parameter *>(par));
        return valueDims;
    }

    /**
     * Allocates memory to store the result of the forward 3D pooling layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] method Computation method for the layer
     * \param[in] parameter %Parameter of the forward 3D pooling layer
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        const Input *in = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);

        services::Collection<size_t> valueDims(in->get(layers::forward::data)->getDimensions());
        computeValueDimensions(valueDims, algParameter);

        if (!get(layers::forward::value))
        {
            DAAL_ALLOCATE_TENSOR_AND_SET(layers::forward::value, valueDims);
        }
        if (!get(layers::forward::resultForBackward))
        {
            set(layers::forward::resultForBackward, services::SharedPtr<LayerData>(new LayerData()));
        }
    }

    /**
     * Checks the result of the forward 3D pooling layer
     * \param[in] input %Input object for the layer
     * \param[in] parameter %Parameter of the layer
     * \param[in] method Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE
    {
        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *param = static_cast<const Parameter *>(parameter);

        services::SharedPtr<data_management::Tensor> dataTensor = algInput->get(layers::forward::data);
        const services::Collection<size_t> &dataDims = dataTensor->getDimensions();

        services::Collection<size_t> valueDims(dataDims);

        computeValueDimensions(valueDims, param);

        if (!data_management::checkTensor(get(layers::forward::value).get(), this->_errors.get(), valueStr(), &valueDims)) { return; }

        services::SharedPtr<LayerData> layerData = get(layers::forward::resultForBackward);
        if (!layerData && param->predictionStage == false) { this->_errors->add(services::ErrorNullLayerData); return; }

        if (!layerData && layerData->size() != 1)
            if (!layerData) { this->_errors->add(services::ErrorIncorrectSizeOfLayerData); return; }

    }

protected:
    size_t computeValueDimension(size_t dataDim, size_t kernelSize, size_t padding, size_t stride) const
    {
        size_t valueDim = (dataDim + 2 * padding - kernelSize + stride) / stride;
        return valueDim;
    }

    void computeValueDimensions(services::Collection<size_t> &dims, const Parameter *param) const
    {
        for (size_t d = 0; d < 3; d++)
        {
            dims[param->indices.size[d]] = computeValueDimension(
                dims[param->indices.size[d]], param->kernelSizes.size[d], param->paddings.size[d], param->strides.size[d]);
        }
    }

    data_management::NumericTablePtr createAuxInputDimensions(const services::Collection<size_t> &dataDims) const
    {
        size_t nInputDims = dataDims.size();
        data_management::HomogenNumericTable<int> *auxInputDimsTable = new data_management::HomogenNumericTable<int>(
            nInputDims, 1, data_management::NumericTableIface::doAllocate);
        int *auxInputDimsData = auxInputDimsTable->getArray();
        for (size_t i = 0; i < nInputDims; i++)
        {
            auxInputDimsData[i] = (int)dataDims[i];
        }
        return data_management::NumericTablePtr(auxInputDimsTable);
    }
};

} // namespace interface1
using interface1::Input;
using interface1::Result;

} // namespace forward
/** @} */
} // namespace pooling3d
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif
