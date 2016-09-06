/* file: data_dictionary.h */
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
//  Implementation of a data dictionary.
//--
*/

#ifndef __DATA_DICTIONARY_H__
#define __DATA_DICTIONARY_H__

#include "services/daal_defines.h"
#include "services/daal_memory.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_archive.h"
#include "data_management/data/data_utils.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @defgroup data_dictionary Data Dictionaries
 * \brief Contains classes that represent a dictionary of a data set and provide methods to work with the data dictionary
 * @ingroup data_management
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__NUMERICTABLEFEATURE"></a>
 *  \brief Data structure describes the Numeric Table feature
 */
class NumericTableFeature : public SerializationIface
{
public:
    data_feature_utils::IndexNumType      indexType;
    data_feature_utils::PMMLNumType       pmmlType;
    data_feature_utils::FeatureType       featureType;
    size_t                              typeSize;
    size_t                              categoryNumber;

public:
    /**
     *  Constructor of a data feature
     */
    NumericTableFeature()
    {
        indexType          = data_feature_utils::DAAL_OTHER_T;
        pmmlType           = data_feature_utils::DAAL_GEN_UNKNOWN;
        featureType        = data_feature_utils::DAAL_CONTINUOUS;
        typeSize           = 0;
        categoryNumber     = 0;
    }

    /**
     *  Copy operator for a data feature
     */
    NumericTableFeature &operator= (const NumericTableFeature &f)
    {
        indexType          = f.indexType     ;
        pmmlType           = f.pmmlType      ;
        featureType        = f.featureType   ;
        typeSize           = f.typeSize      ;
        categoryNumber     = f.categoryNumber;

        return *this;
    }

    virtual ~NumericTableFeature() {}

    /**
     *  Fills the class based on a specified type
     *  \tparam  T  Name of the data feature
     */
    template<typename T>
    void setType()
    {
        typeSize  = sizeof(T);
        indexType = data_feature_utils::getIndexNumType<T>();
        pmmlType  = data_feature_utils::getPMMLNumType<T>();
    }

    /** \private */
    void serializeImpl  (InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>( arch );}

    /** \private */
    void deserializeImpl(OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>( arch );}

    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        arch->set( pmmlType         );
        arch->set( featureType      );
        arch->set( typeSize         );
        arch->set( categoryNumber   );
        arch->set( indexType        );
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SERIALIZATION_DATAFEATURE_NT_ID;
    }
};

/** \private */
class DictionaryIface {};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__DICTIONARY"></a>
 *  \brief Class that represents a dictionary of a data set
 *  and provides methods to work with the data dictionary
 */
template<typename Feature, int SerializationTag>
class Dictionary : public SerializationIface, public DictionaryIface
{
public:
    /**
     *  Constructor of a data dictionary
     *  \param[in]  nfeat  Number of features in the table
     *  \param[in]  featuresEqual Flag specifying that all features have equal types and properties
     */
    Dictionary( size_t nfeat, bool featuresEqual = false ):
        _nfeat(0), _featuresEqual(featuresEqual), _dict(0), _errors(new services::KernelErrorCollection())
    {
        if(nfeat) { setNumberOfFeatures(nfeat); }
    }

    /**
     *  Default constructor of a data dictionary
     */
    Dictionary(): _nfeat(0), _dict(0), _featuresEqual(false), _errors(new services::KernelErrorCollection()) {}

    /** \private */
    ~Dictionary()
    {
        resetDictionary();
    }

    /**
     *  Resets a dictionary and sets the number of features to 0
     */
    void resetDictionary()
    {
        if(_dict)
        {
            delete[] _dict;
            _dict = NULL;
        }
        _nfeat = 0;
    }

    /**
     *  Sets all features of a dictionary to the same type
     *  \param[in]  defaultFeature  Default feature class to which to set all features
     */
    virtual void setAllFeatures(const Feature &defaultFeature)
    {
        if (_featuresEqual)
        {
            if (_nfeat > 0)
            {
                _dict[0] = defaultFeature;
            }
        }
        else
        {
            for( size_t i = 0 ; i < _nfeat ; i++ )
            {
                _dict[i] = defaultFeature;
            }
        }
    }

    /**
     *  Sets the number of features
     *  \param[in]  numberOfFeatures  Number of features
     */
    virtual void setNumberOfFeatures(size_t numberOfFeatures)
    {
        resetDictionary();
        _nfeat = numberOfFeatures;
        if (_featuresEqual)
        {
            _dict  = new Feature[1];
        }
        else
        {
            _dict  = new Feature[_nfeat];
        }
    }

    /**
     *  Returns the number of features
     *  \return Number of features
     */
    size_t getNumberOfFeatures() const
    {
        return _nfeat;
    }

    /**
     *  Returns the value of the featuresEqual flag
     *  \return Value of the featuresEqual flag
     */
    size_t getFeaturesEqual() const
    {
        return _featuresEqual;
    }

    /**
     *  Returns a feature with a given index
     *  \param[in]  idx  Index of the feature
     *  \return Requested feature
     */
    Feature &operator[](size_t idx)
    {
        if (_featuresEqual)
        {
            return _dict[0];
        }
        else
        {
            return _dict[idx];
        }
    }

    /**
     *  \brief Adds a feature to a data dictionary
     *
     *  \param[in] feature  Data feature
     *  \param[in] idx      Index of the data feature
     *
     */
    void setFeature(const Feature &feature, size_t idx)
    {
        if(idx >= _nfeat) { this->_errors->add(services::ErrorIncorrectNumberOfFeatures); return; }
        if (_featuresEqual)
        {
            _dict[0] = feature;
        }
        else
        {
            _dict[idx] = feature;
        }
    }

    /**
     *  Adds a feature to a data dictionary
     *  \param[in] idx              Index of the data feature
     */
    template<typename T>
    void setFeature(size_t idx)
    {
        Feature df;
        df.template setType<T>();
        setFeature(df, idx);
    }

    /**
     * Returns errors during the computation
     * \return Errors during the computation
     */
    services::SharedPtr<services::KernelErrorCollection> getErrors()
    {
        return _errors;
    }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return SerializationTag;
    }

    /** \private */
    void serializeImpl  (InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<InputDataArchive, false>( arch );}

    /** \private */
    void deserializeImpl(OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<OutputDataArchive, true>( arch );}

private:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl( Archive *arch )
    {
        arch->segmentHeader();

        arch->set( _nfeat );
        arch->set( _featuresEqual );

        if( onDeserialize )
        {
            size_t nfeat = _nfeat;
            _nfeat = 0;
            setNumberOfFeatures(nfeat);
        }

        if (_featuresEqual)
        {
            arch->setObj( _dict, 1 );
        }
        else
        {
            arch->setObj( _dict, _nfeat );
        }

        arch->segmentFooter();
    }

protected:
    size_t   _nfeat;
    bool     _featuresEqual;
    Feature *_dict;
    services::SharedPtr<services::KernelErrorCollection> _errors;
};
typedef Dictionary<NumericTableFeature, SERIALIZATION_DATADICTIONARY_NT_ID> NumericTableDictionary;
/** @} */

} // namespace interface1
using interface1::NumericTableFeature;
using interface1::DictionaryIface;
using interface1::Dictionary;
using interface1::NumericTableDictionary;

}
} // namespace daal
#endif
