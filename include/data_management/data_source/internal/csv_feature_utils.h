/* file: csv_feature_utils.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

#ifndef __CSV_FEATURE_UTILS_H__
#define __CSV_FEATURE_UTILS_H__

#include <sstream>

#include "services/collection.h"
#include "services/daal_string.h"
#include "data_management/features/defines.h"

namespace daal
{
namespace data_management
{
namespace internal
{

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__CSVROWTOKENIZER"></a>
 *  \brief Class that parses single row in CSV file and implements iterator-like
 *         interface to iterate over the parsed tokens separated by comma
 */
class CSVRowTokenizer : public Base
{
private:
    char *_rawData;
    const size_t _rawDataSize;
    const char _delimiter;

    size_t _pos;
    size_t _prevPos;
    size_t _tokenSize;
    bool _goodFlag;

public:
    explicit CSVRowTokenizer(char *rawData, size_t rawDataSize, char delimiter) :
        _rawData(rawData),
        _rawDataSize(rawDataSize),
        _delimiter(delimiter),
        _pos(0),
        _prevPos(0),
        _tokenSize(0),
        _goodFlag(true) { }

    void reset()
    {
        _pos       = 0;
        _prevPos   = 0;
        _tokenSize = 0;
        _goodFlag  = true;

        next();
    }

    DAAL_FORCEINLINE void next()
    {
        /* We assume _rawData is single line of CSV file and
         * has a termination character in the end */

        if (!good()) { return; }

        _prevPos = _pos;

        while (isValidSymbol(_pos) && !isStopSymbol(_pos))
        { _pos++; }

        _tokenSize = _pos - _prevPos;
        _goodFlag = isValidSymbol(_prevPos);

        if (isValidSymbol(_pos) && isStopSymbol(_pos))
        {
            _rawData[_pos] = '\0';
            _pos++;
        }
    }

    DAAL_FORCEINLINE bool good() const
    {
        return _goodFlag;
    }

    DAAL_FORCEINLINE services::StringView getCurrentToken() const
    {
        return services::StringView(_rawData + _prevPos, _tokenSize);
    }

private:
    DAAL_FORCEINLINE bool isValidSymbol(size_t index) const
    {
        return index < _rawDataSize &&
               _rawData[index] != '\0';
    }

    DAAL_FORCEINLINE bool isStopSymbol(size_t index) const
    {
        return _rawData[index] == _delimiter;
    }

    CSVRowTokenizer(const CSVRowTokenizer &);
    CSVRowTokenizer &operator=(const CSVRowTokenizer &);
};

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__INTERNAL__CSVFEATURESINFO"></a>
 *  \brief Class that holds auxiliary information about features being parsed
 */
class CSVFeaturesInfo : public Base
{
public:
    services::Status addFeatureName(const services::StringView &featureName)
    {
        const services::String featureNameStr(featureName.begin(), featureName.end());
        if ( !_featureNames.safe_push_back(featureNameStr) )
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }
        return services::Status();
    }

    services::Status addFeatureType(const services::StringView &token)
    {
        const features::FeatureType featureType = detectFeatureType(token);
        if ( !_featureTypes.safe_push_back(featureType) )
        {
            return services::throwIfPossible(services::ErrorMemoryAllocationFailed);
        }
        return services::Status();
    }

    size_t getNumberOfFeatures() const
    {
        /* We allow _featureNames to be empty to support a no-header case */
        if (_featureNames.size() != 0)
        {
            DAAL_ASSERT( _featureNames.size() == _featureTypes.size() );
            return _featureNames.size();
        }
        return _featureTypes.size();
    }

    const services::String &getFeatureName(size_t featureIndex) const
    {
        DAAL_ASSERT( _featureNames.size() == 0 ||
                     _featureNames.size() == _featureTypes.size() );
        DAAL_ASSERT( featureIndex < _featureNames.size() );
        return _featureNames[featureIndex];
    }

    features::FeatureType getDetectedFeatureType(size_t featureIndex) const
    {
        DAAL_ASSERT( featureIndex < _featureTypes.size() );
        return _featureTypes[featureIndex];
    }

    bool areFeatureNamesAvailable() const
    {
        return _featureNames.size() > 0;
    }

private:
    static features::FeatureType detectFeatureType(const services::StringView &token)
    {
        return isNumericalFeature(token)
            ? features::DAAL_CONTINUOUS
            : features::DAAL_CATEGORICAL;
    }

    static bool isNumericalFeature(const services::StringView &token)
    {
        std::istringstream iss(token.c_str());
        DAAL_DATA_TYPE f = 0.0; iss >> f;
        return !(iss.fail());
    }

private:
    services::Collection<services::String> _featureNames;
    services::Collection<features::FeatureType> _featureTypes;
};

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
