/* file: DistributedStep4Local.java */
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

/**
 * @ingroup dbscan_distributed
 * @{
 */
package com.intel.daal.algorithms.dbscan;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDSTEP4LOCAL"></a>
 * @brief Runs the DBSCAN algorithm in the fourth step of the distributed processing mode
 */
public class DistributedStep4Local extends AnalysisDistributed {
    public  DistributedStep4LocalInput input;      /*!< %Input data */
    public  Parameter                  parameter;  /*!< Parameters of the algorithm */
    public  Method                     method;     /*!< Computation method for the algorithm */
    private Precision                  precision;  /*!< Precision of intermediate computations */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the DBSCAN algorithm by copying input objects and parameters
     * of another DBSCAN algorithm
     * @param context   Context to manage the algorithm
     * @param other     An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    public DistributedStep4Local(DaalContext context, DistributedStep4Local other) {
        super(context);
        this.method = other.method;
        precision = other.precision;

        this.cObject = cClone(other.cObject, precision.getValue(), this.method.getValue());

        input     = new DistributedStep4LocalInput(getContext(), cGetInput     (cObject, precision.getValue(), method.getValue()));
        parameter = new Parameter                 (getContext(), cInitParameter(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Constructs the DBSCAN algorithm
     * @param context     Context to manage the algorithm
     * @param cls         Data type to use in intermediate computations for the algorithm,
     *                    Double.class or Float.class
     * @param method      Computation method of the algorithm, @ref Method
     * @param leftBlocks  Number of blocks that will process observations with value of selected
     *                    split feature lesser than selected split value
     * @param rightBlocks Number of blocks that will process observations with value of selected
     *                    split feature greater than selected split value
     */
    public DistributedStep4Local(DaalContext context, Class<? extends Number> cls, Method method, long leftBlocks, long rightBlocks) {
        super(context);

        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            precision = Precision.doublePrecision;
        } else {
            precision = Precision.singlePrecision;
        }

        this.cObject = cInit(precision.getValue(), this.method.getValue(), leftBlocks, rightBlocks);

        input     = new DistributedStep4LocalInput(getContext(), cGetInput     (cObject, precision.getValue(), method.getValue()));
        parameter = new Parameter                 (getContext(), cInitParameter(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Runs the DBSCAN algorithm
     * @return  Partial results of the DBSCAN algorithm
     */
    @Override
    public DistributedPartialResultStep4 compute() {
        super.compute();
        return new DistributedPartialResultStep4(getContext(), cGetPartialResult(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store partial results of the DBSCAN algorithm
     * @param partialResult         Structure to store partial results of the DBSCAN algorithm
     */
    public void setPartialResult(DistributedPartialResultStep4 partialResult) {
        cSetPartialResult(cObject, precision.getValue(), method.getValue(), partialResult.getCObject());
    }

    /**
     * Returns the newly allocated DBSCAN algorithm with a copy of input objects
     * and parameters of this DBSCAN algorithm
     * @param context   Context to manage the algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public DistributedStep4Local clone(DaalContext context) {
        return new DistributedStep4Local(context, this);
    }

    private native long cInit(int precision, int method, long leftBlocks, long rightBlocks);
    private native long cInitParameter(long addr, int precision, int method);
    private native long cGetInput(long addr, int precision, int method);
    private native long cGetPartialResult(long addr, int precision, int method);
    private native void cSetPartialResult(long addr, int precision, int method, long cPartialResult);
    private native long cClone(long addr, int precision, int method);
}
/** @} */
