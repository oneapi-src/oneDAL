/* file: engine.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#ifndef __DATA_SOURCE_MODIFIERS_INTERNAL_ENGINE_H__
#define __DATA_SOURCE_MODIFIERS_INTERNAL_ENGINE_H__

#include "services/collection.h"
#include "data_management/features/identifiers.h"

namespace daal
{
namespace data_management
{
namespace modifiers
{
namespace internal
{
/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__INTERNAL__INPUTFEATUREINFO"></a>
 * \brief Base class represents input feature for modifier, contains information about single input feature
 */
class InputFeatureInfo : public Base
{};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__INTERNAL__OUTPUTFEATUREINFO"></a>
 * \brief Base class represents output feature for modifier, contains information about single output feature
 */
class OutputFeatureInfo : public Base
{
public:
    OutputFeatureInfo() : _numberOfCategories(0), _featureType(features::DAAL_CONTINUOUS) {}

    void setNumberOfCategories(size_t numberOfCategories) { _numberOfCategories = numberOfCategories; }

    void setFeatureType(features::FeatureType featureType) { _featureType = featureType; }

    void setCategoricalDictionary(const CategoricalFeatureDictionaryPtr & dictionary) { _dictionary = dictionary; }

    void fillDataSourceFeature(DataSourceFeature & feature) const
    {
        setDataSourceFeatureType(feature);
        feature.ntFeature.categoryNumber = _numberOfCategories;
        feature.setCategoricalDictionary(_dictionary);
    }

private:
    void setDataSourceFeatureType(DataSourceFeature & feature) const
    {
        switch (_featureType)
        {
        case features::DAAL_CONTINUOUS: feature.setType<DAAL_DATA_TYPE>(); break;

        case features::DAAL_ORDINAL:
        case features::DAAL_CATEGORICAL: feature.setType<int>(); break;
        }
        feature.ntFeature.featureType = _featureType;
    }

private:
    size_t _numberOfCategories;
    features::FeatureType _featureType;
    CategoricalFeatureDictionaryPtr _dictionary;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__INTERNAL__CONFIG"></a>
 * \brief Base class for modifier configuration
 * \tparam  InputFeatureInfo  The type of input feature info
 * \tparam  OutputFeatureInfo The type of output feature info
 */
template <typename InputFeatureInfo, typename OutputFeatureInfo>
class Config
{
public:
    typedef InputFeatureInfo InputFeatureInfoType;
    typedef OutputFeatureInfo OutputFeatureInfoType;

    Config() {}

    explicit Config(const services::internal::CollectionPtr<InputFeatureInfo *> & pickedInputFeatures, services::Status * status = NULL)
        : _pickedInputFeatures(pickedInputFeatures)
    {
        services::Status localStatus = reallocateOutputFeatures(pickedInputFeatures->size());
        services::internal::tryAssignStatusAndThrow(status, localStatus);
    }

    size_t getNumberOfInputFeatures() const { return _pickedInputFeatures->size(); }

    services::Status setNumberOfOutputFeatures(size_t numberOfOutputFeatures) { return reallocateOutputFeatures(numberOfOutputFeatures); }

    services::Status setOutputFeatureType(size_t outputFeatureIndex, features::FeatureType featureType)
    {
        if (outputFeatureIndex >= _outputFeatures.size())
        {
            return services::throwIfPossible(services::ErrorIncorrectIndex);
        }

        _outputFeatures[outputFeatureIndex].setFeatureType(featureType);
        return services::Status();
    }

    services::Status setNumberOfCategories(size_t outputFeatureIndex, size_t numberOfCategories)
    {
        if (outputFeatureIndex >= _outputFeatures.size())
        {
            return services::throwIfPossible(services::ErrorIncorrectIndex);
        }

        _outputFeatures[outputFeatureIndex].setNumberOfCategories(numberOfCategories);
        return services::Status();
    }

    services::Status setCategoricalDictionary(size_t outputFeatureIndex, const CategoricalFeatureDictionaryPtr & dictionary)
    {
        if (outputFeatureIndex >= _outputFeatures.size())
        {
            return services::throwIfPossible(services::ErrorIncorrectIndex);
        }

        _outputFeatures[outputFeatureIndex].setCategoricalDictionary(dictionary);
        return services::Status();
    }

    size_t getNumberOfOutputFeatures() const { return _outputFeatures.size(); }

    const services::Collection<OutputFeatureInfo> & getOutputFeaturesInfo() const { return _outputFeatures; }

protected:
    const services::Collection<InputFeatureInfo *> & getPickedInputFeatures() const { return *_pickedInputFeatures; }

    const InputFeatureInfo & getPickedInputFeature(size_t index) const { return *(_pickedInputFeatures->get(index)); }

    services::Collection<OutputFeatureInfo> & getOutputFeatures() { return _outputFeatures; }

    const OutputFeatureInfo & getOutputFeature(size_t index) const { return _outputFeatures[index]; }

    OutputFeatureInfo & getOutputFeature(size_t index) { return _outputFeatures[index]; }

private:
    services::Status reallocateOutputFeatures(size_t numberOfOutputFeatures)
    {
        _outputFeatures = services::Collection<OutputFeatureInfo>(numberOfOutputFeatures);
        if (!_outputFeatures.data())
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }

        return services::Status();
    }

private:
    services::internal::CollectionPtr<InputFeatureInfo *> _pickedInputFeatures;
    services::Collection<OutputFeatureInfo> _outputFeatures;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__INTERNAL__CONTEXT"></a>
 * \brief Base class for modifier context
 * \tparam  InputFeatureInfo  The type of input feature info
 * \tparam  OutputFeatureInfo The type of output feature info
 */
template <typename InputFeatureInfo, typename OutputFeatureInfo>
class Context
{
public:
    typedef InputFeatureInfo InputFeatureInfoType;
    typedef OutputFeatureInfo OutputFeatureInfoType;

    Context() {}

    explicit Context(const services::internal::CollectionPtr<InputFeatureInfoType *> & pickedInputFeatures, services::Status * /*status*/ = NULL)
        : _pickedInputFeatures(pickedInputFeatures)
    {}

    size_t getNumberOfInputFeatures() const { return _pickedInputFeatures->size(); }

    services::BufferView<DAAL_DATA_TYPE> getOutputBuffer() const { return _outputBuffer; }

    void setOutputBuffer(const services::BufferView<DAAL_DATA_TYPE> & buffer) { _outputBuffer = buffer; }

protected:
    const services::Collection<InputFeatureInfoType *> & getPickedInputFeatures() const { return *_pickedInputFeatures; }

    const InputFeatureInfoType & getPickedInputFeature(size_t index) const { return *(_pickedInputFeatures->get(index)); }

private:
    services::BufferView<DAAL_DATA_TYPE> _outputBuffer;
    services::internal::CollectionPtr<InputFeatureInfoType *> _pickedInputFeatures;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__INTERNAL__MODIFIERBINDING"></a>
 * \brief Class that binds feature identifiers to concrete feature indices, performs
 *        initialization of a modifier and manages Config and Context objects
 * \tparam  Modifier The type of a modifier
 * \tparam  Config   The type of configuration object used by a modifier
 * \tparam  Context  The type of context object used by a modifier
 */
template <typename Modifier, typename Config, typename Context>
class ModifierBinding : public Base
{
public:
    typedef Modifier ModifierType;
    typedef typename Config::InputFeatureInfoType InputFeatureInfoType;
    typedef typename Config::OutputFeatureInfoType OutputFeatureInfoType;

    ModifierBinding() : _outputFeaturesOffset(0), _numberOfOutputFeatures(0) {}

    explicit ModifierBinding(const features::FeatureIdCollectionIfacePtr & identifiers, const services::SharedPtr<Modifier> & modifier,
                             services::Status * /*status*/ = NULL)
        : _outputFeaturesOffset(0), _numberOfOutputFeatures(0), _modifier(modifier), _identifiers(identifiers)
    {}

    services::Status bind(size_t outputFeaturesOffset, const features::FeatureIdMappingIfacePtr & mapping,
                          const services::internal::CollectionPtr<InputFeatureInfoType> & inputFeaturesInfo)
    {
        services::Status status;

        features::FeatureIndicesIfacePtr indices = _identifiers->mapToFeatureIndices(*mapping, &status);
        DAAL_CHECK_STATUS_VAR(status);

        services::internal::CollectionPtr<InputFeatureInfoType *> pickedInputFeatureInfo =
            features::internal::pickElements(indices, inputFeaturesInfo, &status);
        DAAL_CHECK_STATUS_VAR(status);

        _config  = Config(pickedInputFeatureInfo, &status);
        _context = Context(pickedInputFeatureInfo, &status);
        DAAL_CHECK_STATUS_VAR(status);

        _modifier->initialize(_config);

        _outputFeaturesOffset   = outputFeaturesOffset;
        _numberOfOutputFeatures = _config.getNumberOfOutputFeatures();

        return status;
    }

    void apply(const services::BufferView<DAAL_DATA_TYPE> & outputBuffer)
    {
        _context.setOutputBuffer(outputBuffer.getBlock(_outputFeaturesOffset, _numberOfOutputFeatures));
        _modifier->apply(_context);
    }

    void finalize() { _modifier->finalize(_config); }

    const OutputFeatureInfoType & getOutputFeatureInfo(size_t featureIndex) const { return _config.getOutputFeaturesInfo()[featureIndex]; }

    size_t getNumberOfOutputFeatures() const { return _numberOfOutputFeatures; }

private:
    Config _config;
    Context _context;

    size_t _outputFeaturesOffset;
    size_t _numberOfOutputFeatures;

    services::SharedPtr<Modifier> _modifier;
    features::FeatureIdCollectionIfacePtr _identifiers;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__INTERNAL__MODIFIERSBINDER"></a>
 * \brief Class that creates and manages bindings for a modifier
 * \tparam  ModifierBinding The type of binding for a modifier
 */
template <typename ModifierBinding>
class ModifiersBinder : public Base
{
public:
    typedef typename ModifierBinding::ModifierType ModifierType;
    typedef typename ModifierBinding::InputFeatureInfoType InputFeatureInfoType;
    typedef typename ModifierBinding::OutputFeatureInfoType OutputFeatureInfoType;

    ModifiersBinder() : _numberOfOutputFeatures(0) {}

    services::Status add(const features::FeatureIdCollectionIfacePtr & identifiers, const services::SharedPtr<ModifierType> & modifier)
    {
        if (!identifiers || !modifier)
        {
            return services::throwIfPossible(services::ErrorNullPtr);
        }

        if (!_bindings.safe_push_back(ModifierBinding(identifiers, modifier)))
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }

        return services::Status();
    }

    void apply(const services::BufferView<DAAL_DATA_TYPE> & outputBuffer)
    {
        for (size_t i = 0; i < _bindings.size(); i++)
        {
            _bindings[i].apply(outputBuffer);
        }
    }

    void finalize()
    {
        for (size_t i = 0; i < _bindings.size(); i++)
        {
            _bindings[i].finalize();
        }
    }

    services::Status bind(const features::FeatureIdMappingIfacePtr & mapping,
                          const services::internal::CollectionPtr<InputFeatureInfoType> & inputFeaturesInfo)
    {
        DAAL_ASSERT(mapping);
        DAAL_ASSERT(inputFeaturesInfo);

        services::Status status;

        size_t outputFeaturesOffset = 0;
        for (size_t i = 0; i < _bindings.size(); i++)
        {
            status |= _bindings[i].bind(outputFeaturesOffset, mapping, inputFeaturesInfo);
            DAAL_CHECK_STATUS_VAR(status);

            outputFeaturesOffset += _bindings[i].getNumberOfOutputFeatures();
        }

        _inputFeaturesInfo      = inputFeaturesInfo;
        _numberOfOutputFeatures = outputFeaturesOffset;

        return status;
    }

    size_t getNumberOfOutputFeatures() const { return _numberOfOutputFeatures; }

    size_t getNumberOfModifiers() const { return _bindings.size(); }

    const ModifierBinding & getBinding(size_t index) const { return _bindings[index]; }

    services::Collection<InputFeatureInfoType> & getInputFeaturesInfo() { return *_inputFeaturesInfo; }

    InputFeatureInfoType & getInputFeatureInfo(size_t featureIndex) { return _inputFeaturesInfo->get(featureIndex); }

private:
    size_t _numberOfOutputFeatures;
    services::Collection<ModifierBinding> _bindings;
    services::internal::CollectionPtr<InputFeatureInfoType> _inputFeaturesInfo;
};

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__MODIFIERS__INTERNAL__MODIFIERSMANAGER"></a>
 * \brief Class that holds modifiers and implements logic of modifiers applying flow
 * \tparam  Modifier The type of a modifier
 * \tparam  Config   The type of configuration object used by a modifier
 * \tparam  Context  The type of context object used by a modifier
 */
template <typename Modifier, typename Config, typename Context>
class ModifiersManager : public Base
{
public:
    typedef ModifierBinding<Modifier, Config, Context> ModifierBindingType;
    typedef typename ModifierBindingType::InputFeatureInfoType InputFeatureInfoType;
    typedef typename ModifierBindingType::OutputFeatureInfoType OutputFeatureInfoType;

    services::Status addModifier(const features::FeatureIdCollectionIfacePtr & identifiers, const services::SharedPtr<Modifier> & modifier)
    {
        return _binder.add(identifiers, modifier);
    }

    void applyModifiers(const services::BufferView<DAAL_DATA_TYPE> & outputBuffer) { _binder.apply(outputBuffer); }

    void finalize() { _binder.finalize(); }

    services::Status fillDictionary(DataSourceDictionary & dictionary)
    {
        const size_t numberOfOutputFeatures = _binder.getNumberOfOutputFeatures();
        dictionary.setNumberOfFeatures(numberOfOutputFeatures);

        size_t featureCounter = 0;
        for (size_t i = 0; i < _binder.getNumberOfModifiers(); i++)
        {
            const ModifierBindingType & binding = _binder.getBinding(i);
            for (size_t j = 0; j < binding.getNumberOfOutputFeatures(); j++)
            {
                const OutputFeatureInfoType & fi = binding.getOutputFeatureInfo(j);
                fi.fillDataSourceFeature(dictionary[featureCounter++]);
            }
        }
        DAAL_ASSERT(numberOfOutputFeatures == featureCounter);

        return services::Status();
    }

    size_t getNumberOfOutputFeatures() const { return _binder.getNumberOfOutputFeatures(); }

protected:
    ModifiersManager() {}

    modifiers::internal::ModifiersBinder<ModifierBindingType> & getBinder() { return _binder; }

private:
    modifiers::internal::ModifiersBinder<ModifierBindingType> _binder;
};

} // namespace internal
} // namespace modifiers
} // namespace data_management
} // namespace daal

#endif
