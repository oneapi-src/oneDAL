/* file: service_service.cpp */
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

/*
//++
//  Implementation of service functions
//--
*/

#include "src/externals/service_service.h"

float daal::services::daal_string_to_float(const char * nptr, char ** endptr)
{
    return daal::internal::ServiceInst::serv_string_to_float(nptr, endptr);
}

double daal::services::daal_string_to_double(const char * nptr, char ** endptr)
{
    return daal::internal::ServiceInst::serv_string_to_double(nptr, endptr);
}

int daal::services::daal_string_to_int(const char * nptr, char ** endptr)
{
    return daal::internal::ServiceInst::serv_string_to_int(nptr, endptr);
}

int daal::services::daal_int_to_string(char * buffer, size_t n, int value)
{
    return daal::internal::ServiceInst::serv_int_to_string(buffer, n, value);
}

int daal::services::daal_double_to_string(char * buffer, size_t n, double value)
{
    return daal::internal::ServiceInst::serv_double_to_string(buffer, n, value);
}
