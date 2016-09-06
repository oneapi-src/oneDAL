/* file: csv_feature_manager.h */
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
//  Implementation of the CSV feature manager class.
//--
*/

#ifndef __CSV_DATA_SOURCE_H__
#define __CSV_DATA_SOURCE_H__

#include <sstream>
#include <fstream>
#include <list>

#include "services/daal_memory.h"
#include "data_management/data_source/data_source.h"
#include "data_management/data_source/data_source_dictionary.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @defgroup data_sources Data Sources
 * \brief Specifies methods to access data
 * @ingroup data_management
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__CSVFEATUREMANAGER"></a>
 *  \brief Methods of the class to preprocess data represented in the CSV format.
 */
class CSVFeatureManager : public StringRowFeatureManagerIface
{
protected:
    char _delimiter;

public:
    /**
     *  Default constructor
     */
    CSVFeatureManager() : _delimiter(',') {}

    /**
     *  Sets a new character as a delimiter for parsing CSV data (default ',')
     */
    void setDelimiter( char delimiter )
    {
        _delimiter = delimiter;
    }

public:
    virtual void parseRowAsDictionary( char *rawRowData, size_t rawDataSize,
                                       DataSourceDictionary *dict ) DAAL_C11_OVERRIDE
    {
        std::stringstream ssLine(rawRowData);
        char *word = new char[rawDataSize];

        std::list<DataSourceFeature> featureList;

        bool isEmpty = false;
        size_t nCols = 0;
        while( ssLine.getline( word, rawDataSize, _delimiter ) )
        {
            double f;
            isEmpty = (word[0] == 0 || word[0] == '\r' || word[0] == '\n');
            bool isNumeric = readNumeric<>( word, f );

            DataSourceFeature feat;

            if( isNumeric )
            {
                feat.setType<double>();
            }
            else
            {
                feat.setType<int>();
                feat.ntFeature.featureType = data_feature_utils::DAAL_CATEGORICAL;
            }

            featureList.push_back(feat);

            nCols++;
        }

        if(isEmpty) { nCols--; }

        delete[] word;

        dict->setNumberOfFeatures(nCols);

        size_t idx = 0;
        for( std::list<DataSourceFeature>::iterator it = featureList.begin() ; it != featureList.end() ; it++ )
        {
            dict->setFeature( *it, idx );
            idx++;
            if( idx == nCols ) { break; }
        }
    }

    /**
     *  Parses a string that represents a feature vector and converts it into a numeric representation
     *  \param[in]  rawRowData   Array of characters with the string that represents the feature vector
     *  \param[in]  rawDataSize  Size of the rawRowData array
     *  \param[in]  dict         Pointer to the dictionary
     *  \param[out] nt           Pointer to a Numeric Table to store the result of parsing
     *  \param[in]  ntRowIndex   Position in the Numeric Table at which to store the result of parsing
     */
    virtual void parseRowIn ( char *rawRowData, size_t rawDataSize, DataSourceDictionary *dict,
                              NumericTable *nt, size_t  ntRowIndex  ) DAAL_C11_OVERRIDE
    {
        std::stringstream ssLine( rawRowData );
        char *word = new char[rawDataSize];

        size_t nCols = nt->getNumberOfColumns();

        BlockDescriptor<double> block;
        nt->getBlockOfRows( ntRowIndex, 1, writeOnly, block );
        double *row = block.getBlockPtr();

        size_t i;
        for( i = 0; i < nCols; i++ )
        {
            ssLine.getline( word, rawDataSize, _delimiter );
            if( ssLine.fail() ) { break; }

            DataSourceFeature   &dsFeat = (*dict)[i];
            NumericTableFeature &ntFeat = dsFeat.ntFeature;
            if( ntFeat.featureType == data_feature_utils::DAAL_CONTINUOUS )
            {
                double f;
                if( readNumeric<>( word, f ) )
                {
                    row[ i ] = f;
                }
                else
                {
                    /* NonNumeric data in NumericTable is invalid */
                    row[ i ] = 0;
                }
            }
            else
            {
                std::string sWord(word);

                CategoricalFeatureDictionary *catDict = dsFeat.getCategoricalDictionary();
                CategoricalFeatureDictionary::iterator it = catDict->find( sWord );

                if( it != catDict->end() )
                {
                    row[ i ] = it->second.first;
                    it->second.second++;
                }
                else
                {
                    int index = (int)(catDict->size());
                    catDict->insert( std::pair<std::string, std::pair<int, int> >( sWord, std::pair<int, int>(index, 1) ) );
                    row[ i ] = index;
                }
            }
        }

        delete[] word;

        nt->releaseBlockOfRows( block );
    }

protected:
    template<class T>
    bool readNumeric(char *text, T &f)
    {
        std::istringstream iss(text);
        iss >> f;
        return !(iss.fail());
    }

};
/** @} */
} // namespace interface1
using interface1::CSVFeatureManager;

}
}
#endif
