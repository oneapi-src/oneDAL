/* file: datasource_mysql.cpp */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#include <ctime>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <stdexcept>

#include "utils.h"
#include "odbc_wrapper.h"

#include "daal.h"
#include "data_management/data_source/odbc_data_source.h"
#include "data_management/data_source/mysql_feature_manager.h"

using namespace daal::data_management;


std::vector<std::string> injectString2(const std::vector<std::string>& inputs, size_t numberString, size_t position) {
    std::vector<std::string> res(inputs);
    res[numberString].insert(position, 1, '\0');
    return res;
}


int main(int argc, char const *argv[])
{
    /*
     * This sample demonstrates how to connect to MySQL server using connection string and
     * perform basic data selection and loading with ODBCDataSource component.
     */


    std::vector<std::string> inputs{
        std::string("dbname"),
        std::string("tablename"),
        std::string("username"),
        std::string("password")
    };
    daal::data_management::ODBCDataSourceOptions options;





    for (size_t numberString = 0; numberString < inputs.size(); ++numberString)
    {
        for (size_t position = 0; position <= inputs[numberString].size(); ++position)
        {
            bool success = false;
            std::vector<std::string> modifiedInputs(injectString2(inputs, numberString, position));
            try
            {
                daal::data_management::ODBCDataSource<daal::data_management::SQLFeatureManager>(modifiedInputs[0], modifiedInputs[1], modifiedInputs[2], modifiedInputs[3]);
            }
            catch (daal::services::Exception& daalExc)
            {
                std::string receivedMessage(daalExc.what());
                receivedMessage = receivedMessage.substr(0, receivedMessage.find('\n'));
                if (receivedMessage == "Null byte injection has been detected")
                {
                    success = true;
                }
                else
                {
                    std::cout << "ERR: " << receivedMessage << std::endl;
                }
            }
            if (!success)
            {
                std::cout << "EEEERROR" << std::endl;
                return 1;
            }




            try
            {
                daal::data_management::ODBCDataSource<daal::data_management::SQLFeatureManager>(modifiedInputs[0], modifiedInputs[1], modifiedInputs[2], modifiedInputs[3], options);
            }
            catch (daal::services::Exception& daalExc)
            {
                std::string receivedMessage(daalExc.what());
                receivedMessage = receivedMessage.substr(0, receivedMessage.find('\n'));
                if (receivedMessage == "Null byte injection has been detected")
                {
                    success = true;
                }
                else
                {
                    std::cout << "ERR: " << receivedMessage << std::endl;
                }
            }
            if (!success)
            {
                std::cout << "EEEERROR" << std::endl;
                return 1;
            }
        }
    }


    std::string connectionString("connection");
    for (size_t position = 0; position <= connectionString.size(); ++position)
    {
        bool success = false;
        std::string copy(connectionString);
        copy.insert(position, 1, '\0');
        try
        {
            daal::data_management::ODBCDataSource<daal::data_management::SQLFeatureManager>(copy, options);
        }
        catch (daal::services::Exception& daalExc)
        {
            std::string receivedMessage(daalExc.what());
            receivedMessage = receivedMessage.substr(0, receivedMessage.find('\n'));
            if (receivedMessage == "Null byte injection has been detected")
            {
                success = true;
            }
            else
            {
                std::cout << "ERR: " << receivedMessage << std::endl;
            }
        }
        if (!success)
        {
            std::cout << "EEEERROR" << std::endl;
            return 1;
        }
    }

    std::string str("connection");
    for (size_t position = 0; position <= str.size(); ++position)
    {
        bool success = false;
        try
        {
            std::string copy(str);
            copy.insert(position, 1, '\0');
            daal::data_management::SQLFeatureManager sqlfm;
            sqlfm.setLimitQuery(copy, 0, 0);
        }
        catch (daal::services::Exception& daalExc)
        {
            std::string receivedMessage(daalExc.what());
            receivedMessage = receivedMessage.substr(0, receivedMessage.find('\n'));
            if (receivedMessage == "Null byte injection has been detected")
            {
                success = true;
            }
            else
            {
                std::cout << "ERR: " << receivedMessage << std::endl;
            }
        }
        if (!success)
        {
            std::cout << "EEEERROR" << std::endl;
            return 1;
        }


        try
        {
            std::string connectionStringInner;

            if (argc > 1)
            { connectionStringInner = argv[1]; }

            if (utils::trim(connectionStringInner).empty())
            { utils::printHelp(); return 0; }
            odbc_wrapper::Connection connection(connectionStringInner);


            /* Create a table and insert a few rows into it */
            const std::string tableName = utils::generateTableName(connection.id());
            /* Crate ODBC Data Source via connection string */
            const ODBCDataSourceOptions options = ODBCDataSourceOptions::allocateNumericTable |
                                                ODBCDataSourceOptions::createDictionaryFromContext;
            ODBCDataSource<SQLFeatureManager> ds(connectionStringInner, options);

            std::string copy(str);
            copy.insert(position, 1, '\0');
            /* Execute SQL query, you can execute arbitrary query supported by your DB */
            ds.executeQuery(copy);
        }
        catch (daal::services::Exception& daalExc)
        {
            std::string receivedMessage(daalExc.what());
            receivedMessage = receivedMessage.substr(0, receivedMessage.find('\n'));
            if (receivedMessage == "Null byte injection has been detected")
            {
                success = true;
            }
            else
            {
                std::cout << "ERR: " << receivedMessage << std::endl;
            }
        }
        if (!success)
        {
            std::cout << "EEEERROR" << std::endl;
            return 1;
        }
    }


    return 0;
}
