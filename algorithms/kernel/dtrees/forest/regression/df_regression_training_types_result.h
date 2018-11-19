/* file: df_regression_training_types_result.h */
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

/*
//++
//  Implementation of decision forest algorithm classes.
//--
*/

#include "algorithms/decision_forest/decision_forest_regression_training_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{

namespace regression
{
namespace training
{
namespace interface1
{

class Result::ResultImpl
{
public:
    ResultImpl() {}
    ResultImpl(const ResultImpl& other)
    {
        if(other._engine) _engine = other._engine->clone();
    }

    void setEngine(engines::EnginePtr engine) { _engine = engine; }
    engines::EnginePtr getEngine()
    {
        if(!_engine) _engine = engines::mt2203::Batch<>::create();
        return _engine;
    }
private:
    engines::EnginePtr _engine;
};

} // namespace interface1
} // namespace training
} // namespace regression
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
