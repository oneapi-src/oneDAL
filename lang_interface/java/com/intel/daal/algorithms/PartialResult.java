/* file: PartialResult.java */
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

package com.intel.daal.algorithms;

import com.intel.daal.data_management.data.SerializableBase;
import com.intel.daal.services.DaalContext;

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__PARTIALRESULT"></a>
 *  \brief %Base class to represent partial results of the computation.
 *         Algorithm-specific partial results are represented as derivative classes of the PartialResult class.
 */
abstract public class PartialResult extends SerializableBase {

    /**
     * @brief PartialResult default constructor. Constructs empty partial results
     */
    protected PartialResult(DaalContext context) {
        super(context);
    }

    /**
     * Constructs parameter from C++ parameter
     * @param context Context to manage the partial result
     * @param cPartialResult Address of C++ parameter
     */
    public PartialResult(DaalContext context, long cPartialResult) {
        super(context);
        this.cObject = cPartialResult;
    }

    /**
     * Checks the correctness of the partial results
     * @param input         Input object
     * @param parameter     Parameters of the algorithm
     * @param method        Computation method
     */
    public void check(Input input, Parameter parameter, int method) {
        cCheckInput(this.cObject, input.getCObject(), parameter.getCObject(), method);
    }

    /**
     * Checks the correctness of the partial result
     * @param parameter     Parameters of the algorithm
     * @param method        Computation method
     */
    public void check(Parameter parameter, int method) {
        cCheck(this.cObject, parameter.getCObject(), method);
    }

    private native void cCheckInput(long partResAddr, long inputAddr, long parAddr, int method);

    private native void cCheck(long partResAddr, long parAddr, int method);
}
