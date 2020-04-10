/* file: DistributedStep1Local.java */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
 * @defgroup dbscan_distributed Distributed
 * @ingroup dbscan_compute
 * @{
 */
/**
 * @brief Contains classes of the DBSCAN algorithm
 */
package com.intel.daal.algorithms.dbscan;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.AnalysisDistributed;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDSTEP1LOCAL"></a>
 * @brief Runs the DBSCAN algorithm in the first step of the distributed processing mode
 */
public class DistributedStep1Local extends AnalysisDistributed {
    public  DistributedStep1LocalInput input;      /*!< %Input data */
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
    public DistributedStep1Local(DaalContext context, DistributedStep1Local other) {
        super(context);
        this.method = other.method;
        precision = other.precision;

        this.cObject = cClone(other.cObject, precision.getValue(), this.method.getValue());

        input     = new DistributedStep1LocalInput(getContext(), cGetInput     (cObject, precision.getValue(), method.getValue()));
        parameter = new Parameter                 (getContext(), cInitParameter(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Constructs the DBSCAN algorithm
     * @param context    Context to manage the algorithm
     * @param cls        Data type to use in intermediate computations for the algorithm,
     *                   Double.class or Float.class
     * @param method     Computation method of the algorithm, @ref Method
     * @param blockIndex Unique identifier of block initially passed for computation on the local node
     * @param nBlocks    Number of blocks initially passed for computation on all nodes
     */
    public DistributedStep1Local(DaalContext context, Class<? extends Number> cls, Method method, long blockIndex, long nBlocks) {
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

        this.cObject = cInit(precision.getValue(), this.method.getValue(), blockIndex, nBlocks);

        input     = new DistributedStep1LocalInput(getContext(), cGetInput     (cObject, precision.getValue(), method.getValue()));
        parameter = new Parameter                 (getContext(), cInitParameter(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Runs the DBSCAN algorithm
     * @return  Partial results of the DBSCAN algorithm
     */
    @Override
    public DistributedPartialResultStep1 compute() {
        super.compute();
        return new DistributedPartialResultStep1(getContext(), cGetPartialResult(cObject, precision.getValue(), method.getValue()));
    }

    /**
     * Registers user-allocated memory to store partial results of the DBSCAN algorithm
     * @param partialResult         Structure to store partial results of the DBSCAN algorithm
     */
    public void setPartialResult(DistributedPartialResultStep1 partialResult) {
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
    public DistributedStep1Local clone(DaalContext context) {
        return new DistributedStep1Local(context, this);
    }

    private native long cInit(int precision, int method, long blockIndex, long nBlocks);
    private native long cInitParameter(long addr, int precision, int method);
    private native long cGetInput(long addr, int precision, int method);
    private native long cGetPartialResult(long addr, int precision, int method);
    private native void cSetPartialResult(long addr, int precision, int method, long cPartialResult);
    private native long cClone(long addr, int precision, int method);
}
/** @} */
