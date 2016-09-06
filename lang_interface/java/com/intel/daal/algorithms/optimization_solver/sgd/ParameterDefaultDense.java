/* file: ParameterDefaultDense.java */
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

package com.intel.daal.algorithms.optimization_solver.sgd;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.algorithms.optimization_solver.sgd.BaseParameter;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SGD__PARAMETERDEFAULTDENSE"></a>
 * @brief ParameterDefaultDense of the SGD algorithm
 */
public class ParameterDefaultDense extends BaseParameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
    * Constructs the parameter for SGD algorithm
    * @param context       Context to manage the SGD algorithm
    */
    public ParameterDefaultDense(DaalContext context) {
        super(context);
    }

    /**
     * Constructs the parameter for SGD algorithm
     * @param context       Context to manage the SGD algorithm
     * @param cParameter    Pointer to C++ implementation of the parameter
     */
    public ParameterDefaultDense(DaalContext context, long cParameter) {
        super(context, cParameter);
    }
}
