/* file: Result.java */
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
 *  <a name="DAAL-CLASS-ALGORITHMS__RESULT"></a>
 *  \brief %Base class to represent final results of the computation.
 *         Algorithm-specific final results are represented as derivative classes of the Result class.
 */
abstract public class Result extends SerializableBase {

    /**
     * @brief Result default constructor. Constructs empty final results
     */
    protected Result(DaalContext context) {
        super(context);
    }

    /**
     * Constructs parameter from C++ parameter
     * @param context Context to manage the result
     * @param cResult Address of C++ parameter
     */
    public Result(DaalContext context, long cResult) {
        super(context);
        this.cObject = cResult;
    }

    /**
     * Checks the correctness of the result
     * @param input         Input object
     * @param parameter     Parameters of the algorithm
     * @param method        Computation method
     */
    public void check(Input input, Parameter parameter, int method) {
        cCheckInput(this.cObject, input.getCObject(), parameter.getCObject(), method);
    }

    /**
     * Checks the correctness of the result
     * @param partialResult Partial result of the algorithm
     * @param parameter     Parameters of the algorithm
     * @param method        Computation method
     */
    public void check(PartialResult partialResult, Parameter parameter, int method) {
        cCheckPartRes(this.cObject, partialResult.getCObject(), parameter.getCObject(), method);
    }

    private native void cDispose(long parAddr);

    private native void cCheckInput(long resAddr, long inputAddr, long parAddr, int method);

    private native void cCheckPartRes(long resAddr, long partResAddr, long parAddr, int method);

}
