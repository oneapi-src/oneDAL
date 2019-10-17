/* file: neural_networks_weights_and_biases.h */
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
//  Implementation of classes for storing learnable parameters of neural network
//--
*/

#ifndef __NEURAL_NETWORKS_WEIGHTS_AND_BIASES_H__
#define __NEURAL_NETWORKS_WEIGHTS_AND_BIASES_H__

#include "neural_networks_learnable_parameters.h"
#include "service_tensor.h"
#include "service_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::internal;

template<typename modelFPType>
SharedPtr<HomogenNumericTable<modelFPType> > allocateWeightsAndBiasesTable(size_t wbSize)
{
    return HomogenNumericTable<modelFPType>::create(1, wbSize, NumericTableIface::doAllocate);
}

template<typename modelFPType>
class NumericTableLearnableParametersImpl : public LearnableParametersIface
{
public:
    virtual NumericTablePtr copyToTable() const DAAL_C11_OVERRIDE
    {
        return _wbTable;
    }

    virtual NumericTablePtr copyToTable(size_t idx) const DAAL_C11_OVERRIDE
    {
        if (idx > _nLayers || ((_wSize[idx] + _bSize[idx]) == 0))
        {
            return NumericTablePtr();
        }
        const modelFPType *wbArray = _wbTable->getArray();
        size_t nRows = _wSize[idx] + _bSize[idx];
        SharedPtr<HomogenNumericTable<modelFPType> > resultTable =
            allocateWeightsAndBiasesTable<modelFPType>(nRows);
        if (!resultTable.get())
        {
            return NumericTablePtr();
        }
        modelFPType *resultArray = resultTable->getArray();
        size_t resultSize = nRows * sizeof(modelFPType);
        daal::services::internal::daal_memcpy_s(resultArray, resultSize, wbArray + _wOffsets[idx], resultSize);
        return resultTable;
    }

    Status setTable(const SharedPtr<HomogenNumericTable<modelFPType> > &table)
    {
        _wbTable = table;
        return Status();
    }

    virtual Status copyFromTable(const NumericTablePtr &table) DAAL_C11_OVERRIDE
    {
        if (table.get() == _wbTable.get())
            return Status();

        return copyFromTableImpl(table, _wbSize, 0);
    }

    virtual Status copyFromTable(const NumericTablePtr &table, size_t idx) DAAL_C11_OVERRIDE
    {
        if (table.get() == _wbTable.get() ||
            idx > _nLayers                ||
            ((_wSize[idx] + _bSize[idx]) == 0))
        {
            return Status();
        }
        return copyFromTableImpl(table, _wSize[idx] + _bSize[idx], _wOffsets[idx]);
    }

protected:

    Status copyFromTableImpl(const NumericTablePtr &table, size_t dataSize, size_t offset)
    {
        ReadRows<modelFPType, sse2> tableRows(table.get(), 0, table->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(tableRows)
        const modelFPType *inputArray = tableRows.get();
        modelFPType *wbArray = _wbTable->getArray();
        size_t dataSizeInBytes = dataSize * sizeof(modelFPType);
        daal::services::internal::daal_memcpy_s(wbArray + offset, dataSizeInBytes, inputArray, dataSizeInBytes);
        return Status();
    }

    size_t getCollectionSize(const Collection<size_t> &collection)
    {
        if(collection.size() == 0) { return 0; }

        size_t size = 1;
        for(size_t i = 0; i < collection.size(); i++)
        {
            size *= collection[i];
        }
        return size;
    }

    Status computeWeightsAndBiasesSizes(const neural_networks::ForwardLayersPtr &forwardLayers)
    {
        using namespace layers;

        _nLayers = forwardLayers->size();
        _wbSize = 0;
        size_t weightsSize, biasesSize;
        for (size_t i = 0; i < _nLayers; i++)
        {
            forward::LayerIfacePtr forwardLayer = forwardLayers->get(i);
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
        return Status();
    }

    size_t _nLayers;   /*!< Number of layers */
    size_t _wbSize;   /*!< Full number of elements in weights and biases of all the layers in the network */
    SharedPtr<HomogenNumericTable<modelFPType> > _wbTable; /*!< Weights and biases of all the layers in the network */
    Collection<Collection<size_t> > _wDimsCollection;   /*!< Collection of weights tensors dimensions of all layers */
    Collection<Collection<size_t> > _bDimsCollection;   /*!< Collection of biases  tensors dimensions of all layers */
    Collection<size_t> _wOffsets;        /*!< Collection of the offsets of data blocks that contain weights tensors for each layer */
    Collection<size_t> _bOffsets;        /*!< Collection of the offsets of data blocks that contain biases  tensors for each layer */
    Collection<size_t> _wSize;           /*!< Collection of number of elements in weights tensors for each layer */
    Collection<size_t> _bSize;           /*!< Collection of number of elements in biases  tensors for each layer */
};

template<typename modelFPType>
class TensorLearnableParametersImpl : public LearnableParametersIface
{
    typedef SharedPtr<HomogenNumericTable<modelFPType> > HomogenNumericTablePtr;
public:
    virtual NumericTablePtr copyToTable() const DAAL_C11_OVERRIDE
    {
        return tensorsToTable(_wbTensors, 0, 2 * _nLayers);
    }

    virtual NumericTablePtr copyToTable(size_t idx) const DAAL_C11_OVERRIDE
    {
        return tensorsToTable(_wbTensors, 2 * idx, 2);
    }

    virtual Status copyFromTable(const NumericTablePtr &table) DAAL_C11_OVERRIDE
    {
        return tableToTensors(table, _wbTensors, 0, 2 * _nLayers);
    }

    virtual Status copyFromTable(const NumericTablePtr &table, size_t idx) DAAL_C11_OVERRIDE
    {
        return tableToTensors(table, _wbTensors, 2 * idx, 2);
    }

protected:

    HomogenNumericTablePtr tensorsToTable(
        const Collection<TensorPtr> &tensors, size_t startTensor, size_t nTensors) const
    {
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
            return HomogenNumericTablePtr();
        }

        HomogenNumericTablePtr table = allocateWeightsAndBiasesTable<modelFPType>(tableSize);
        modelFPType *tableArray = table->getArray();

        size_t tableOffset = 0;
        ReadSubtensor<modelFPType, sse2> subtensor;
        for (size_t i = startTensor; i < startTensor + nTensors; i++)
        {
            if (!tensors[i]) { continue; }
            const size_t tensorSize = tensors[i]->getSize();
            if (tensorSize == 0) { continue; }
            const Collection<size_t> &dims = tensors[i]->getDimensions();
            size_t firstDimension = dims[0];
            subtensor.set(tensors[i].get(), 0, 0, 0, firstDimension);
            if (!subtensor.status())
                return HomogenNumericTablePtr();
            const modelFPType *tensorArray = subtensor.get();

            daal::services::internal::daal_memcpy_s(tableArray + tableOffset, tensorSize * sizeof(modelFPType),
                                                    tensorArray, tensorSize * sizeof(modelFPType));
            tableOffset += tensorSize;
        }
        return table;
    }

    Status tableToTensors(const NumericTablePtr &table,
        Collection<TensorPtr> &tensors, size_t startTensor, size_t nTensors)
    {
        ReadRows<modelFPType, sse2> tableRows(table.get(), 0, table->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(tableRows)
        const modelFPType *tableArray = tableRows.get();

        size_t tableOffset = 0;
        WriteOnlySubtensor<modelFPType, sse2> subtensor;
        for (size_t i = startTensor; i < startTensor + nTensors; i++)
        {
            if (!tensors[i]) { continue; }
            const size_t tensorSize = tensors[i]->getSize();
            if (tensorSize == 0) { continue; }
            const Collection<size_t> &dims = tensors[i]->getDimensions();
            size_t firstDimension = dims[0];
            subtensor.set(tensors[i].get(), 0, 0, 0, firstDimension);
            DAAL_CHECK_BLOCK_STATUS(subtensor)
            modelFPType *tensorArray = subtensor.get();

            daal::services::internal::daal_memcpy_s(tensorArray, tensorSize * sizeof(modelFPType),
                                                    tableArray + tableOffset, tensorSize * sizeof(modelFPType));
            tableOffset += tensorSize;
        }
        return Status();
    }

    size_t _nLayers;
    Collection<TensorPtr > _wbTensors;
};

template<typename modelFPType>
class TensorWeightsAndBiases: public TensorLearnableParametersImpl<modelFPType>
{
    typedef TensorLearnableParametersImpl<modelFPType> super;
    typedef SharedPtr<TensorWeightsAndBiases<modelFPType> > ThisTypePtr;

    using super::_nLayers;
    using super::_wbTensors;

public:
    static ThisTypePtr create(
            neural_networks::ForwardLayersPtr &forwardLayers, Status *pSt = nullptr)
    {
        Status defaultSt;
        Status &s = (pSt ? *pSt : defaultSt);
        auto res = ThisTypePtr(new TensorWeightsAndBiases<modelFPType>(forwardLayers, s));
        return (s ? res : ThisTypePtr());
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() const DAAL_C11_OVERRIDE
    {
        return features::internal::getIndexNumType<modelFPType>() + SERIALIZATION_NEURAL_NETWORKS_TENSOR_WEIGHTS_AND_BIASES_ID;
    }

    /**
     * Serializes an object
     * \param[in]  arch  Storage for a serialized object or data structure
     */
    services::Status serializeImpl(InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    /**
     * Deserializes an object
     * \param[in]  arch  Storage for a deserialized object or data structure
     */
    services::Status deserializeImpl(const OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }

protected:
    TensorWeightsAndBiases(neural_networks::ForwardLayersPtr &forwardLayers, Status &s)
    {
        using namespace layers;

        _nLayers = forwardLayers->size();
        for(size_t i = 0; i < _nLayers; i++)
        {
            forward::LayerIfacePtr forwardLayer = forwardLayers->get(i);

            s = forwardLayer->allocateInput();
            if (!s)
                return;

            s = forwardLayer->initializeInput();
            if (!s)
                return;

            forward::Input *input = forwardLayer->getLayerInput();
            _wbTensors.push_back(input->get(forward::weights));
            _wbTensors.push_back(input->get(forward::biases));
        }
    }

    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return services::Status();
    }
};

template<typename modelFPType>
class TensorWeightsAndBiasesDerivatives: public TensorLearnableParametersImpl<modelFPType>
{
    typedef TensorLearnableParametersImpl<modelFPType> super;
    typedef SharedPtr<TensorWeightsAndBiasesDerivatives<modelFPType> > ThisTypePtr;

    using super::_nLayers;
    using super::_wbTensors;

public:
    static ThisTypePtr create(
            neural_networks::BackwardLayersPtr &backwardLayers, Status *pSt = nullptr)
    {
        Status defaultSt;
        Status &s = (pSt ? *pSt : defaultSt);
        return ThisTypePtr(new TensorWeightsAndBiasesDerivatives<modelFPType>(backwardLayers));
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() const DAAL_C11_OVERRIDE
    {
        return features::internal::getIndexNumType<modelFPType>() + SERIALIZATION_NEURAL_NETWORKS_TENSOR_WEIGHTS_AND_BIASES_DERIVATIVES_ID;
    }

    /**
     * Serializes an object
     * \param[in]  arch  Storage for a serialized object or data structure
     */
    services::Status serializeImpl(InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    /**
     * Deserializes an object
     * \param[in]  arch  Storage for a deserialized object or data structure
     */
    services::Status deserializeImpl(const OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }

protected:
    TensorWeightsAndBiasesDerivatives(neural_networks::BackwardLayersPtr &backwardLayers)
    {
        using namespace layers;

        _nLayers = backwardLayers->size();
        for(size_t i = 0; i < _nLayers; i++)
        {
            backward::LayerIfacePtr backwardLayer = backwardLayers->get(i);

            backward::ResultPtr result = backwardLayer->getLayerResult();
            _wbTensors.push_back(result->get(backward::weightDerivatives));
            _wbTensors.push_back(result->get(backward::biasDerivatives));
        }
    }

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return services::Status();
    }
};

template<typename modelFPType>
class NumericTableWeightsAndBiases: public NumericTableLearnableParametersImpl<modelFPType>
{
    typedef NumericTableLearnableParametersImpl<modelFPType> super;
    typedef SharedPtr<NumericTableWeightsAndBiases<modelFPType> > ThisTypePtr;

    using super::_nLayers;
    using super::_wbSize;
    using super::_wbTable;
    using super::_wDimsCollection;
    using super::_bDimsCollection;
    using super::_wOffsets;
    using super::_bOffsets;
    using super::_wSize;
    using super::_bSize;

    using super::computeWeightsAndBiasesSizes;
public:
    static ThisTypePtr create(neural_networks::ForwardLayersPtr &forwardLayers,
                              Status *pSt = nullptr)
    {
        Status defaultSt;
        Status &s = (pSt ? *pSt : defaultSt);
        auto res = ThisTypePtr(new NumericTableWeightsAndBiases<modelFPType>(forwardLayers, s));
        return (s ? res : ThisTypePtr());
    }

    static ThisTypePtr create(neural_networks::ForwardLayersPtr &forwardLayers,
                              SharedPtr<HomogenNumericTable<modelFPType> > &table,
                              Status *pSt = nullptr)
    {
        Status defaultSt;
        Status &s = (pSt ? *pSt : defaultSt);
        auto res = ThisTypePtr(new NumericTableWeightsAndBiases<modelFPType>(forwardLayers, table, s));
        return (s ? res : ThisTypePtr());
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() const DAAL_C11_OVERRIDE
    {
        return features::internal::getIndexNumType<modelFPType>() + SERIALIZATION_NEURAL_NETWORKS_NUMERIC_TABLE_WEIGHTS_AND_BIASES_ID;
    }

    /**
     * Serializes an object
     * \param[in]  arch  Storage for a serialized object or data structure
     */
    services::Status serializeImpl(InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    /**
     * Deserializes an object
     * \param[in]  arch  Storage for a deserialized object or data structure
     */
    services::Status deserializeImpl(const OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }

protected:
    NumericTableWeightsAndBiases(neural_networks::ForwardLayersPtr &forwardLayers,
                                 Status &s)
    {
        computeWeightsAndBiasesSizes(forwardLayers);
        _wbTable = allocateWeightsAndBiasesTable<modelFPType>(_wbSize);
        if (!_wbTable.get())
        {
            s.add(ErrorMemoryAllocationFailed);
            return;
        }

        s = setWeightsAndBiasesFromTable(forwardLayers);
        if (!s)
            return;

        for (size_t i = 0; i < _nLayers; i++)
        {
            if(!forwardLayers->get(i)->getLayerParameter()->weightsAndBiasesInitialized)
            {
                s = forwardLayers->get(i)->initializeInput();
                if (!s)
                    return;
            }
        }
    }

    NumericTableWeightsAndBiases(neural_networks::ForwardLayersPtr &forwardLayers,
                                 SharedPtr<HomogenNumericTable<modelFPType> > &table,
                                 Status &s)
    {
        computeWeightsAndBiasesSizes(forwardLayers);
        _wbTable = table;
        s = setWeightsAndBiasesFromTable(forwardLayers);
        if (!s)
            return;
    }

    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return services::Status();
    }

    Status setWeightsAndBiasesFromTable(ForwardLayersPtr &forwardLayers)
    {
        using namespace layers;

        modelFPType *wbArray = _wbTable->getArray();
        services::Status s;
        for (size_t i = 0; i < _nLayers; i++)
        {
            forward::Input *input = forwardLayers->get(i)->getLayerInput();
            if (_wDimsCollection[i].size() != 0)
            {
                TensorPtr weights = HomogenTensor<modelFPType>::create(_wDimsCollection[i], wbArray + _wOffsets[i], &s);
                DAAL_CHECK_STATUS_VAR(s);
                input->set(forward::weights, weights);
            }
            if (_bDimsCollection[i].size() != 0)
            {
                TensorPtr biases = HomogenTensor<modelFPType>::create(_bDimsCollection[i], wbArray + _bOffsets[i], &s);
                DAAL_CHECK_STATUS_VAR(s);
                input->set(forward::biases, biases);
            }
        }
        return s;
    }
};

template<typename modelFPType>
class NumericTableWeightsAndBiasesDerivatives: public NumericTableLearnableParametersImpl<modelFPType>
{
    typedef NumericTableLearnableParametersImpl<modelFPType> super;
    typedef SharedPtr<NumericTableWeightsAndBiasesDerivatives> ThisTypePtr;

    using super::_nLayers;
    using super::_wbSize;
    using super::_wbTable;
    using super::_wDimsCollection;
    using super::_bDimsCollection;
    using super::_wOffsets;
    using super::_bOffsets;
    using super::_wSize;
    using super::_bSize;

    using super::computeWeightsAndBiasesSizes;
public:
    static ThisTypePtr create(neural_networks::ForwardLayersPtr  &forwardLayers,
                              neural_networks::BackwardLayersPtr &backwardLayers, Status *pSt = nullptr)
    {
        Status defaultSt;
        Status &s = (pSt ? *pSt : defaultSt);
        auto res = ThisTypePtr(new NumericTableWeightsAndBiasesDerivatives<modelFPType>(forwardLayers, backwardLayers, s));
        return (s ? res : ThisTypePtr());
    }

    /**
     * Returns the serialization tag of the neural network model
     * \return         Serialization tag of the neural network model
     */
    int getSerializationTag() const DAAL_C11_OVERRIDE
    {
        return features::internal::getIndexNumType<modelFPType>() + SERIALIZATION_NEURAL_NETWORKS_NUMERIC_TABLE_WEIGHTS_AND_BIASES_DERIVATIVES_ID;
    }

    /**
     * Serializes an object
     * \param[in]  arch  Storage for a serialized object or data structure
     */
    services::Status serializeImpl(InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<InputDataArchive, false>(arch);

        return services::Status();
    }

    /**
     * Deserializes an object
     * \param[in]  arch  Storage for a deserialized object or data structure
     */
    services::Status deserializeImpl(const OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const OutputDataArchive, true>(arch);

        return services::Status();
    }

protected:

    NumericTableWeightsAndBiasesDerivatives(neural_networks::ForwardLayersPtr  &forwardLayers,
                                            neural_networks::BackwardLayersPtr &backwardLayers, Status &s)
    {
        computeWeightsAndBiasesSizes(forwardLayers);
        _wbTable = allocateWeightsAndBiasesTable<modelFPType>(_wbSize);
        if (!_wbTable.get())
        {
            s.add(ErrorMemoryAllocationFailed);
            return;
        }
        s = setWeightsAndBiasesFromTable(backwardLayers);
        if (!s)
            return;
    }

    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return services::Status();
    }

    Status setWeightsAndBiasesFromTable(BackwardLayersPtr &backwardLayers)
    {
        using namespace layers;
        Status s;
        modelFPType *wbArray = _wbTable->getArray();

        for (size_t i = 0; i < _nLayers; i++)
        {
            backward::ResultPtr result = backwardLayers->get(i)->getLayerResult();
            if (_wDimsCollection[i].size() != 0)
            {
                TensorPtr weightDerivatives = HomogenTensor<modelFPType>::create(_wDimsCollection[i], wbArray + _wOffsets[i], &s);
                DAAL_CHECK_STATUS_VAR(s);
                result->set(backward::weightDerivatives, weightDerivatives);
            }
            if (_bDimsCollection[i].size() != 0)
            {
                TensorPtr biasDerivatives = HomogenTensor<modelFPType>::create(_bDimsCollection[i], wbArray + _bOffsets[i], &s);
                DAAL_CHECK_STATUS_VAR(s);
                result->set(backward::biasDerivatives, biasDerivatives);
            }
        }
        return s;
    }
};

}
}
}

#endif
