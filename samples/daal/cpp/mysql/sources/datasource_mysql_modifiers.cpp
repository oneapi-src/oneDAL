/* file: datasource_mysql_modifiers.cpp */
/*******************************************************************************
* Copyright 2017 Intel Corporation
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
! Content:
!
! C++ sample of a MySQL data source.
!
! 1) Connect to the database and create a table there.
!
! 2) Create a dictionary from the data table and read the data from it using
!    the ODBCDataSource functionality. Print the data from the table.
!
! 3) Delete the table from the database and disconnect.
!
!******************************************************************************/

#include <cassert>
#include <algorithm>

#include "utils.h"
#include "odbc_wrapper.h"

#include "daal.h"
#include "data_management/data_source/odbc_data_source.h"

using namespace daal::data_management;

/** User-defined feature modifier that computes a square for every feature */
class MySquaringModifier : public modifiers::sql::FeatureModifier {
public:
    /* This method is called for every row in CSV file */
    virtual void apply(modifiers::sql::Context& context) {
        const size_t numberOfColumns = context.getNumberOfColumns();
        daal::services::BufferView<DAAL_DATA_TYPE> outputBuffer = context.getOutputBuffer();

        /* By default buffer size is equal to the number of columns.
         * This behavior can be redefined by calling 'setNumberOfOutputFeatures' on
         * initialization stage of the modifier (see 'MyMaxFeatureModifier') */
        assert(numberOfColumns == outputBuffer.size());

        for (size_t i = 0; i < numberOfColumns; i++) {
            const float x = context.getValue<float>(i);
            outputBuffer[i] = x * x;
        }
    }
};

/** User-defined feature modifier that selects max element among all features  */
class MyMaxFeatureModifier : public modifiers::sql::FeatureModifier {
public:
    /* This method is called once before CSV parsing */
    virtual void initialize(modifiers::sql::Config& config) {
        /* Set number of output features for the modifier. We assume modifier
         * computes function y = max { x_1, ..., x_n }, where x_i is input
         * features and y is output feature, so there is single output feature  */
        config.setNumberOfOutputFeatures(1);
    }

    /* This method is called for every row in CSV file */
    virtual void apply(modifiers::sql::Context& context) {
        const size_t numberOfColumns = context.getNumberOfColumns();

        /* Iterate throughout tokens, parse every token as float and compute max value  */
        float maxFeature = context.getValue<float>(0);
        for (size_t i = 1; i < numberOfColumns; i++) {
            maxFeature = (std::max)(maxFeature, context.getValue<float>(i));
        }

        /* Write max value to the output buffer, buffer size is equal to the
         * number of output features that specified in 'initialize' method */
        context.getOutputBuffer()[0] = maxFeature;
    }
};

int main(int argc, char const* argv[]) {
    /*
     * This sample demonstrates how to connect to MySQL server using connection string and
     * perform basic data selection and loading with ODBCDataSource component.
     */

    std::string connectionString;

    if (argc > 1) {
        connectionString = argv[1];
    }

    if (utils::trim(connectionString).empty()) {
        utils::printHelp();
        return 0;
    }

    /* Example of user's connection string: */
    // connectionString = "DRIVER=MySQL;"
    //                    "SERVER=<host_name>;"
    //                    "USER=<user_name>;"
    //                    "PASSWORD=<password>;"
    //                    "DATABASE=<data_base_name>";

    /* Establish connection to MySQL server */
    odbc_wrapper::Connection connection(connectionString);

    /* Create a table and insert a few rows into it */
    const std::string tableName = utils::generateTableName(connection.id()) + "_mod";
    connection.execute("CREATE TABLE ? (Col1 float, Col2 float)", tableName);
    connection.execute("INSERT INTO ? VALUES (2.71, 3.90), (1.11, 0.538), (3.44, 1.41)", tableName);

    /* Crate ODBC Data Source via connection string */
    const ODBCDataSourceOptions options = ODBCDataSourceOptions::allocateNumericTable |
                                          ODBCDataSourceOptions::createDictionaryFromContext;
    ODBCDataSource<SQLFeatureManager> ds(connectionString, options);

    /* Execute SQL query, you can execute arbitrary query supported by your DB */
    ds.executeQuery("SELECT Col1, Col2 FROM " + tableName);

    /* Configure format of output numeric table by applying modifiers.
     * Output numeric table will have the following format:
     * | Col1 | Col2 ^ 2 | max(Col1, Col2) | */
    ds.getFeatureManager()
        .addModifier(features::list("Col1"), modifiers::sql::continuous())
        .addModifier(features::list("Col2"), modifiers::sql::custom<MySquaringModifier>())
        .addModifier(features::all(), modifiers::sql::custom<MyMaxFeatureModifier>());

    /* Cause loading data from the table */
    ds.loadDataBlock();

    /* Print loaded numeric table */
    utils::printNumericTable(ds.getNumericTable(), "The loaded table:");

    /* Remove created table */
    connection.execute("DROP TABLE ?", tableName);

    return 0;
}
