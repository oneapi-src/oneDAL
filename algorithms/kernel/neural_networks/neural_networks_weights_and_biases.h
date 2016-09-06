/* file: neural_networks_weights_and_biases.h */
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
//  Implementation of classes for storing learnable parameters of neural network
//--
*/

#ifndef __NEURAL_NETWORKS_WEIGHTS_AND_BIASES_H__
#define __NEURAL_NETWORKS_WEIGHTS_AND_BIASES_H__

#include "neural_networks_learnable_parameters.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{

template<typename modelFPType>
services::SharedPtr<data_management::HomogenNumericTable<modelFPType> > allocateWeightsAndBiasesTable(size_t wbSize)
{
    using namespace services;
    using namespace data_management;

    NumericTableFeature feature;
    feature.setType<modelFPType>();
    SharedPtr<NumericTableDictionary> dictionary(new NumericTableDictionary(wbSize, true));
    dictionary->setAllFeatures(feature);
    SharedPtr<HomogenNumericTable<modelFPType> > table(new HomogenNumericTable<modelFPType>(dictionary));
    table->setNumberOfRows(1);
    table->allocateDataMemory();
    return table;
}

template<typename modelFPType>
class NumericTableLearnableParametersImpl : public LearnableParametersIface
{
public:

    virtual data_management::NumericTablePtr copyToTable() const DAAL_C11_OVERRIDE
    {
        return _wbTable;
    }

    virtual data_management::NumericTablePtr copyToTable(size_t idx) const DAAL_C11_OVERRIDE
    {
        using namespace services;
        using namespace data_management;

        if (idx > _nLayers || ((_wSize[idx] + _bSize[idx]) == 0))
        {
            return NumericTablePtr();
        }
        const modelFPType *wbArray = _wbTable->getArray();
        size_t nColumns = _wSize[idx] + _bSize[idx];
        SharedPtr<HomogenNumericTable<modelFPType> > resultTable =
            allocateWeightsAndBiasesTable<modelFPType>(nColumns);
        modelFPType *resultArray = resultTable->getArray();
        size_t resultSize = nColumns * sizeof(modelFPType);
        daal_memcpy_s(resultArray, resultSize, wbArray + _wOffsets[idx], resultSize);
        return resultTable;
    }

    void setTable(const services::SharedPtr<data_management::HomogenNumericTable<modelFPType> > &table)
    {
        _wbTable = table;
    }

    virtual void copyFromTable(const data_management::NumericTablePtr &table) DAAL_C11_OVERRIDE
    {
        using namespace services;
        using namespace data_management;

        if (table.get() == _wbTable.get()) { return; }

        BlockDescriptor<modelFPType> block;
        table->getBlockOfRows(0, table->getNumberOfRows(), readOnly, block);
        const modelFPType *inputArray = block.getBlockPtr();
        modelFPType *wbArray = _wbTable->getArray();
        size_t dataSize = _wbSize * sizeof(modelFPType);
        daal_memcpy_s(wbArray, dataSize, inputArray, dataSize);
        table->releaseBlockOfRows(block);
    }

    virtual void copyFromTable(const data_management::NumericTablePtr &table, size_t idx) DAAL_C11_OVERRIDE
    {
        using namespace services;
        using namespace data_management;

        if (table.get() == _wbTable.get() ||
            idx > _nLayers                ||
            ((_wSize[idx] + _bSize[idx]) == 0))
        {
            return;
        }
        BlockDescriptor<modelFPType> block;
        table->getBlockOfRows(0, table->getNumberOfRows(), readOnly, block);
        const modelFPType *inputArray = block.getBlockPtr();
        modelFPType *wbArray = _wbTable->getArray();
        size_t dataSize = (_wSize[idx] + _bSize[idx]) * sizeof(modelFPType);
        daal_memcpy_s(wbArray + _wOffsets[idx], dataSize, inputArray, dataSize);
        table->releaseBlockOfRows(block);
    }

protected:

    size_t getCollectionSize(const services::Collection<size_t> &collection)
    {
        if(collection.size() == 0) { return 0; }

        size_t size = 1;
        for(size_t i = 0; i < collection.size(); i++)
        {
            size *= collection[i];
        }
        return size;
    }

    void computeWeightsAndBiasesSizes(const services::SharedPtr<neural_networks::ForwardLayers> &forwardLayers)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        _nLayers = forwardLayers->size();
        _wbSize = 0;
        size_t weightsSize, biasesSize;
        for (size_t i = 0; i < _nLayers; i++)
        {
            SharedPtr<forward::LayerIface> forwardLayer = forwardLayers->get(i);
            forward::Input *forwardInput = forwardLayer->getLayerInput();
            layers::Parameter *parameter = forwardLayer->getLayerParameter();

            _wDimsCollection.push_back(forwardInput->getWeightsSizes(parameter));
            _bDimsCollection .push_back(forwardInput->getBiasesSizes(parameter));
            weightsSize = getCollectionSize(_wDimsCollection[i]);
            biasesSize  = getCollectionSize(_bDimsCollection[i]);

            _wOffsets.push_back(_wbSize);
            _wbSize += weightsSize;
            _wSize.push_back(weightsSize);
            _bOffsets.push_back(_wbSize);
            _wbSize += biasesSize;
            _bSize.push_back(biasesSize);
        }
    }

    size_t _nLayers;   /*!< Number of layers */
    size_t _wbSize;   /*!< Full number of elements in weights and biases of all the layers in the network */
    services::SharedPtr<data_management::HomogenNumericTable<modelFPType> > _wbTable; /*!< Weights and biases of all the layers in the network */
    services::Collection<services::Collection<size_t> > _wDimsCollection;   /*!< Collection of weights tensors dimensions of all layers */
    services::Collection<services::Collection<size_t> > _bDimsCollection;   /*!< Collection of biases  tensors dimensions of all layers */
    services::Collection<size_t> _wOffsets;        /*!< Collection of the offsets of data blocks that contain weights tensors for each layer */
    services::Collection<size_t> _bOffsets;        /*!< Collection of the offsets of data blocks that contain biases  tensors for each layer */
    services::Collection<size_t> _wSize;           /*!< Collection of number of elements in weights tensors for each layer */
    services::Collection<size_t> _bSize;           /*!< Collection of number of elements in biases  tensors for each layer */
};

template<typename modelFPType>
class TensorLearnableParametersImpl : public LearnableParametersIface
{
public:
    virtual data_management::NumericTablePtr copyToTable() const DAAL_C11_OVERRIDE
    {
        return tensorsToTable(_wbTensors, 0, 2 * _nLayers);
    }

    virtual data_management::NumericTablePtr copyToTable(size_t idx) const DAAL_C11_OVERRIDE
    {
        return tensorsToTable(_wbTensors, 2 * idx, 2);
    }

    virtual void copyFromTable(const data_management::NumericTablePtr &table) DAAL_C11_OVERRIDE
    {
        tableToTensors(table, _wbTensors, 0, 2 * _nLayers);
    }

    virtual void copyFromTable(const data_management::NumericTablePtr &table, size_t idx) DAAL_C11_OVERRIDE
    {
        tableToTensors(table, _wbTensors, 2 * idx, 2);
    }

protected:

    services::SharedPtr<data_management::HomogenNumericTable<modelFPType> > tensorsToTable(
        const services::Collection<services::SharedPtr<data_management::Tensor> > &tensors, size_t startTensor, size_t nTensors) const
    {
        using namespace services;
        using namespace data_management;
        size_t tableSize = 0;
        for (size_t i = startTensor; i < startTensor + nTensors; i++)
        {
            if (tensors[i])
            {
                tableSize += tensors[i]->getSize();
            }
        }
        if (tableSize == 0)
        {
            return SharedPtr<HomogenNumericTable<modelFPType> >();
        }

        SharedPtr<HomogenNumericTable<modelFPType> > table = allocateWeightsAndBiasesTable<modelFPType>(tableSize);
        modelFPType *tableArray = table->getArray();

        size_t tableOffset = 0;
        SubtensorDescriptor<modelFPType> subtensor;
        for (size_t i = startTensor; i < startTensor + nTensors; i++)
        {
            if (!tensors[i]) { continue; }
            size_t tensorSize = tensors[i]->getSize();
            if (tensorSize == 0) { continue; }
            const Collection<size_t> &dims = tensors[i]->getDimensions();
            size_t firstDimension = dims[0];
            tensors[i]->getSubtensor(0, 0, 0, firstDimension, readOnly, subtensor);
            modelFPType *tensorArray = subtensor.getPtr();

            daal_memcpy_s(tableArray + tableOffset, tensorSize * sizeof(modelFPType), tensorArray, tensorSize * sizeof(modelFPType));
            tableOffset += tensorSize;

            tensors[i]->releaseSubtensor(subtensor);
        }
        return table;
    }

    void tableToTensors(const data_management::NumericTablePtr &table,
        services::Collection<services::SharedPtr<data_management::Tensor> > &tensors, size_t startTensor, size_t nTensors)
    {
        using namespace data_management;

        BlockDescriptor<modelFPType> block;
        table->getBlockOfRows(0, table->getNumberOfRows(), readOnly, block);
        modelFPType *tableArray = block.getBlockPtr();

        size_t tableOffset = 0;
        SubtensorDescriptor<modelFPType> subtensor;
        for (size_t i = startTensor; i < startTensor + nTensors; i++)
        {
            if (!tensors[i]) { continue; }
            if (tensors[i]->getSize() == 0) { continue; }
            const services::Collection<size_t> &dims = tensors[i]->getDimensions();
            size_t firstDimension = dims[0];
            tensors[i]->getSubtensor(0, 0, 0, firstDimension, writeOnly, subtensor);
            size_t tensorSize = subtensor.getSize();
            modelFPType *tensorArray = subtensor.getPtr();

            services::daal_memcpy_s(tensorArray, tensorSize * sizeof(modelFPType), tableArray + tableOffset, tensorSize * sizeof(modelFPType));
            tableOffset += tensorSize;

            tensors[i]->releaseSubtensor(subtensor);
        }
        table->releaseBlockOfRows(block);
    }

    size_t _nLayers;
    services::Collection<services::SharedPtr<data_management::Tensor> > _wbTensors;
};

template<typename modelFPType>
class TensorWeightsAndBiases: public TensorLearnableParametersImpl<modelFPType>
{
    using TensorLearnableParametersImpl<modelFPType>::_nLayers;
    using TensorLearnableParametersImpl<modelFPType>::_wbTensors;

public:
    TensorWeightsAndBiases(services::SharedPtr<neural_networks::ForwardLayers> &forwardLayers,
                           modelFPType dummy)
    {
        using namespace services;
        using namespace layers;

        _nLayers = forwardLayers->size();
        for(size_t i = 0; i < _nLayers; i++)
        {
            SharedPtr<forward::LayerIface> forwardLayer = forwardLayers->get(i);

            forwardLayer->allocateInput();

            forward::Input *input = forwardLayer->getLayerInput();
            _wbTensors.push_back(input->get(forward::weights));
            _wbTensors.push_back(input->get(forward::biases));
        }
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return data_management::data_feature_utils::getIndexNumType<modelFPType>() + SERIALIZATION_NEURAL_NETWORKS_TENSOR_WEIGHTS_AND_BIASES_ID;
    }

    /**
     * Serializes an object
     * \param[in]  arch  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     * Deserializes an object
     * \param[in]  arch  Storage for a deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {}
};

template<typename modelFPType>
class TensorWeightsAndBiasesDerivatives: public TensorLearnableParametersImpl<modelFPType>
{
    using TensorLearnableParametersImpl<modelFPType>::_nLayers;
    using TensorLearnableParametersImpl<modelFPType>::_wbTensors;

public:
    TensorWeightsAndBiasesDerivatives(services::SharedPtr<neural_networks::BackwardLayers> &backwardLayers,
                           modelFPType dummy)
    {
        using namespace services;
        using namespace layers;

        _nLayers = backwardLayers->size();
        for(size_t i = 0; i < _nLayers; i++)
        {
            SharedPtr<backward::LayerIface> backwardLayer = backwardLayers->get(i);

            SharedPtr<backward::Result> result = backwardLayer->getLayerResult();
            _wbTensors.push_back(result->get(backward::weightDerivatives));
            _wbTensors.push_back(result->get(backward::biasDerivatives));
        }
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return data_management::data_feature_utils::getIndexNumType<modelFPType>() + SERIALIZATION_NEURAL_NETWORKS_TENSOR_WEIGHTS_AND_BIASES_DERIVATIVES_ID;
    }

    /**
     * Serializes an object
     * \param[in]  arch  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     * Deserializes an object
     * \param[in]  arch  Storage for a deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {}
};

template<typename modelFPType>
class NumericTableWeightsAndBiases: public NumericTableLearnableParametersImpl<modelFPType>
{
    using NumericTableLearnableParametersImpl<modelFPType>::_nLayers;
    using NumericTableLearnableParametersImpl<modelFPType>::_wbSize;
    using NumericTableLearnableParametersImpl<modelFPType>::_wbTable;
    using NumericTableLearnableParametersImpl<modelFPType>::_wDimsCollection;
    using NumericTableLearnableParametersImpl<modelFPType>::_bDimsCollection;
    using NumericTableLearnableParametersImpl<modelFPType>::_wOffsets;
    using NumericTableLearnableParametersImpl<modelFPType>::_bOffsets;
    using NumericTableLearnableParametersImpl<modelFPType>::_wSize;
    using NumericTableLearnableParametersImpl<modelFPType>::_bSize;

    using NumericTableLearnableParametersImpl<modelFPType>::computeWeightsAndBiasesSizes;
public:
    NumericTableWeightsAndBiases(services::SharedPtr<neural_networks::ForwardLayers> &forwardLayers,
                                 modelFPType dummy)
    {
        computeWeightsAndBiasesSizes(forwardLayers);
        _wbTable = allocateWeightsAndBiasesTable<modelFPType>(_wbSize);

        setWeightsAndBiasesFromTable(forwardLayers);
        for (size_t i = 0; i < _nLayers; i++)
        {
            if(!forwardLayers->get(i)->getLayerParameter()->weightsAndBiasesInitialized)
            {
                forwardLayers->get(i)->initializeInput();
            }
        }
    }

    NumericTableWeightsAndBiases(services::SharedPtr<neural_networks::ForwardLayers> &forwardLayers,
                                 services::SharedPtr<data_management::HomogenNumericTable<modelFPType> > &table)
    {
        computeWeightsAndBiasesSizes(forwardLayers);
        _wbTable = table;
        setWeightsAndBiasesFromTable(forwardLayers);
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return data_management::data_feature_utils::getIndexNumType<modelFPType>() + SERIALIZATION_NEURAL_NETWORKS_NUMERIC_TABLE_WEIGHTS_AND_BIASES_ID;
    }

    /**
     * Serializes an object
     * \param[in]  arch  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     * Deserializes an object
     * \param[in]  arch  Storage for a deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {}

    void setWeightsAndBiasesFromTable(services::SharedPtr<ForwardLayers> &forwardLayers)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        modelFPType *wbArray = _wbTable->getArray();

        for (size_t i = 0; i < _nLayers; i++)
        {
            forward::Input *input = forwardLayers->get(i)->getLayerInput();

            SharedPtr<Tensor> weights(new HomogenTensor<modelFPType>(_wDimsCollection[i], wbArray + _wOffsets[i]));
            SharedPtr<Tensor> biases(new HomogenTensor<modelFPType>(_bDimsCollection[i], wbArray + _bOffsets[i]));
            input->set(forward::weights, weights);
            input->set(forward::biases, biases);
        }
    }
};

template<typename modelFPType>
class NumericTableWeightsAndBiasesDerivatives: public NumericTableLearnableParametersImpl<modelFPType>
{
    using NumericTableLearnableParametersImpl<modelFPType>::_nLayers;
    using NumericTableLearnableParametersImpl<modelFPType>::_wbSize;
    using NumericTableLearnableParametersImpl<modelFPType>::_wbTable;
    using NumericTableLearnableParametersImpl<modelFPType>::_wDimsCollection;
    using NumericTableLearnableParametersImpl<modelFPType>::_bDimsCollection;
    using NumericTableLearnableParametersImpl<modelFPType>::_wOffsets;
    using NumericTableLearnableParametersImpl<modelFPType>::_bOffsets;
    using NumericTableLearnableParametersImpl<modelFPType>::_wSize;
    using NumericTableLearnableParametersImpl<modelFPType>::_bSize;

    using NumericTableLearnableParametersImpl<modelFPType>::computeWeightsAndBiasesSizes;
public:
    NumericTableWeightsAndBiasesDerivatives(services::SharedPtr<neural_networks::ForwardLayers>  &forwardLayers,
                                            services::SharedPtr<neural_networks::BackwardLayers> &backwardLayers,
                                            modelFPType dummy)
    {
        computeWeightsAndBiasesSizes(forwardLayers);
        _wbTable = allocateWeightsAndBiasesTable<modelFPType>(_wbSize);
        setWeightsAndBiasesFromTable(backwardLayers);
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return data_management::data_feature_utils::getIndexNumType<modelFPType>() + SERIALIZATION_NEURAL_NETWORKS_NUMERIC_TABLE_WEIGHTS_AND_BIASES_DERIVATIVES_ID;
    }

    /**
     * Serializes an object
     * \param[in]  arch  Storage for a serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     * Deserializes an object
     * \param[in]  arch  Storage for a deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {}

    void setWeightsAndBiasesFromTable(services::SharedPtr<BackwardLayers> &backwardLayers)
    {
        using namespace services;
        using namespace data_management;
        using namespace layers;

        modelFPType *wbArray = _wbTable->getArray();

        for (size_t i = 0; i < _nLayers; i++)
        {
            SharedPtr<backward::Result> result = backwardLayers->get(i)->getLayerResult();

            SharedPtr<Tensor> weightDerivatives(new HomogenTensor<modelFPType>(_wDimsCollection[i], wbArray + _wOffsets[i]));
            SharedPtr<Tensor> biasDerivatives(new HomogenTensor<modelFPType>(_bDimsCollection[i], wbArray + _bOffsets[i]));
            result->set(backward::weightDerivatives, weightDerivatives);
            result->set(backward::biasDerivatives, biasDerivatives);
        }
    }
};

}
}
}

#endif
