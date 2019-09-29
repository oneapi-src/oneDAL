/* file: datasource_kdb.cpp */
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
! C++ sample of a Kdb data source.
!
! 1) Connect to the database and create a table there.
!
! 2) Read the data from it using the KDBDataSource functionality. Print the data from the table.
!
! 3) Delete the table from the database and disconnect.
!
!******************************************************************************/

#if defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#endif

#define KXVER 3

#include <iostream>
#include <string>

#include "daal.h"
#include "service.h"
#include "data_management/data_source/kdb_data_source.h"

using namespace daal::algorithms;
using namespace daal;
using namespace std;

string dataSourceName = "daal-examples-kdb.inn.intel.com";
size_t dataSourcePort = 9999;
string dataSourceUsername = "";
string dataSourcePassword = "";

string tableName = "test_table";

std::vector<std::string> injectString(const std::vector<std::string>& inputs, size_t numberString, size_t position) {
    std::vector<std::string> res(inputs);
    res[numberString].insert(position, 1, '\0');
    return res;
}

int main(int argc, char *argv[])
{
    std::vector<std::string> inputs{
        std::string("dbname"),
        std::string("tablename"),
        std::string("username"),
        std::string("password")
    };
    for (size_t numberString = 0; numberString < inputs.size(); ++numberString)
    {
        for (size_t position = 0; position <= inputs[numberString].size(); ++position)
        {
            bool success = false;
            std::vector<std::string> modifiedInputs(injectString(inputs, numberString, position));
            try
            {
                daal::data_management::KDBDataSource<daal::data_management::KDBFeatureManager>(
                    modifiedInputs[0], dataSourcePort, modifiedInputs[1], modifiedInputs[2], modifiedInputs[3]);
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
    return 0;
}
