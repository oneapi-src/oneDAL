/* file: Batch.java */
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

/**
 * @brief Contains classes for computing K-Means
 */
package com.intel.daal.algorithms.kmeans;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KMEANS__BATCH"></a>
 * @brief Computes the results of the K-Means algorithm in the batch processing mode
 * \n<a href="DAAL-REF-KMEANS-ALGORITHM">K-Means algorithm description and usage models</a>
 *
 * @par References
 *      - @ref Method class
 *      - @ref InputId class
 *      - @ref ResultId class
 *      - @ref Input class
 *      - @ref Result class
 */
public class Batch extends AnalysisBatch {
    public Input      input;      /*!< %Input data */
    public Parameter  parameter;  /*!< Parameters of the algorithm */
    public Method     method;     /*!< Computation method for the algorithm */
    private Precision precision; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

     /**
     * Constructs the K-Means algorithm in the batch processing mode by copying input objects and parameters
     * of another K-Means algorithm
     * @param context Context to manage the constructed algorithm
     * @param other   An algorithm to be used as the source to initialize the input objects
     *                and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);

        this.method = other.method;
        precision = other.precision;

        this.cObject = cClone(other.cObject, precision.getValue(), this.method.getValue());

        input     = new Input    (getContext(), cGetInput     (cObject, precision.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(cObject, precision.getValue(), method.getValue()));
    }

     /**
     * Constructs the K-Means algorithm in the batch processing mode
     * @param context       Context to manage the constructed algorithm
     * @param cls           Data type to use in intermediate computations for the algorithm,
     *                      Double.class or Float.class
     * @param method        Computation method of the algorithm, @ref Method
     * @param nClusters     Number of clusters for the algorithm
     * @param maxIterations Maximum number of iterations
     */

    public Batch(DaalContext context, Class<? extends Number> cls, Method method, long nClusters, long maxIterations) {
        super(context);

        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.lloydDense && this.method != Method.lloydCSR) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            precision = Precision.doublePrecision;
        } else {
            precision = Precision.singlePrecision;
        }

        this.cObject = cInit(precision.getValue(), this.method.getValue(), nClusters, maxIterations);

        input     = new Input    (getContext(), cGetInput     (cObject, precision.getValue(), method.getValue()));
        parameter = new Parameter(getContext(), cInitParameter(cObject, precision.getValue(), method.getValue()));
        parameter.setNClusters(nClusters);
        parameter.setMaxIterations(maxIterations);
    }

    /**
     * Runs the K-Means algorithm
     * @return  Result of the K-Means algorithm    */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, precision.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the results of the K-Means algorithm
     * @param result    Structure to store the results of the K-Means algorithm
     */
    public void setResult(Result result) {
        cSetResult(cObject, precision.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated K-Means algorithm in the batch processing mode
     * with a copy of input objects and parameters of this K-Means algorithm
     * @param context Context to manage the constructed algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int precision, int method, long nClusters, long maxIterations);

    private native long cGetInput(long addr, int prec, int method);

    private native long cGetResult(long addr, int prec, int method);

    private native long cInitParameter(long addr, int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cClone(long addr, int prec, int method);
}
