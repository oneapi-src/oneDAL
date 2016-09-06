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

package com.intel.daal.algorithms.em_gmm;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__EM_GMM__BATCH"></a>
 * \brief Runs the EM for GMM algorithm in the batch processing mode.
 * \n<a href="DAAL-REF-EM_GMM-ALGORITHM">EM for GMM algorithm description and usage models</a>
 *
 * \par References
 *      - @ref Method class
 *      - @ref Parameter class
 *      - @ref InputId class
 *      - @ref ResultId class
 *      - @ref Input class
 *      - @ref Result class
 *
 */
public class Batch extends AnalysisBatch {
    public Input          input;     /*!< %Input data */
    public Parameter  parameter;     /*!< Parameters of the algorithm */
    public Method     method; /*!< Computation method for the algorithm */
    private Result    result;      /*!< %Result of the algorithm */
    private Precision precision; /*!< Precision of intermediate computations */
    private long      nComponents; /*!< Number of components in the Gaussian mixture model */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the EM for GMM algorithm by copying input objects and parameters
     * of another EM for GMM algorithm
     * @param context      Context to manage the EM for GMM algorithm
     * @param other        An algorithm to be used as the source to initialize the input objects
     *                     and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        method = other.method;
        precision = other.precision;
        nComponents = other.nComponents;

        this.cObject = cClone(other.cObject, precision.getValue(), method.getValue());

        input = new Input(getContext(), cObject, precision, method, ComputeMode.batch);
        parameter = new Parameter(getContext(), cInitParameter(this.cObject, precision.getValue(), method.getValue()));
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__EM_GMM__BATCH__BATCH"></a>
     * Constructs the EM for GMM algorithm
     *
     * @param context      Context to manage the EM for GMM algorithm
     * @param cls          Data type to use in intermediate computations for the EM for GMM algorithm, Double.class or Float.class
     * @param method       EM for GMM computation method, @ref Method
     * @param nComponents  Number of components in the Gaussian mixture model
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method, long nComponents) {
        super(context);
        this.method = method;
        this.nComponents = nComponents;
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

        this.cObject = cInit(precision.getValue(), this.method.getValue(), nComponents);

        input = new Input(getContext(), cObject, precision, method, ComputeMode.batch);
        parameter = new Parameter(getContext(), this.cObject, precision.getValue(), method.getValue(),
                ComputeMode.batch.getValue(), nComponents, 100, 1.0e-03);
    }

    /**
    * Runs the EM for GMM algorithm
    * @return Results of the EM for GMM algorithm
    */
    @Override
    public Result compute() {
        super.compute();
        result = new Result(getContext(), cObject, precision, method, ComputeMode.batch);
        return result;
    }

    /**
    * Registers user-allocated memory for storing results of the EM for GMM algorithm
    * @param result    Structure for storing the results EM for GMM algorithm
    */
    public void setResult(Result result) {
        cSetResult(cObject, precision.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated EM for GMM algorithm with a copy of input objects
     * of this EM for GMM algorithm
     * @param context      Context to manage the EM for GMM algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int precision, int method, long nComponents);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native long cInitParameter(long algAddr, int precision, int method);

    private native long cClone(long algAddr, int prec, int method);
}
