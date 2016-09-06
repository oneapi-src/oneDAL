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
 * @brief Contains classes for computing the correlation distance
 */
package com.intel.daal.algorithms.cordistance;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CORDISTANCE__BATCH"></a>
 * \brief Computes correlation distance in the batch processing mode.
 * \n<a href="DAAL-REF-CORDISTANCE-ALGORITHM">Correlation distance algorithm description and usage models</a>
 *
 * \par References
 *      - @ref Method class
 *      - @ref InputId class
 *      - @ref ResultId class
 *      - @ref Input class
 *      - @ref Result class
 *
 */
public class Batch extends AnalysisBatch {
    public Input          input;     /*!< %Input data */
    public Method     method; /*!< Computation method for the algorithm */
    private Precision                 prec; /*!< Precision of intermediate computations */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * Constructs the correlation distance algorithm by copying input objects
     * of another correlation distance algorithm
     * @param context    Context to manage the correlation distance algorithm
     * @param other      An algorithm to be used as the source to initialize the input objects
     *                   and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;
        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new Input(getContext(), cObject, prec, method);
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__CORDISTANCE__BATCH__BATCH"></a>
     * Constructs the correlation distance algorithm
     *
     * @param context    Context to manage the correlation distance algorithm
     * @param cls        Data type to use in intermediate computations for the correlation distance algorithm, Double.class or Float.class
     * @param method     Correlation distance computation method, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);

        this.method = method;

        if (method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        } else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue());
        input = new Input(getContext(), cObject, prec, method);
    }

    /**
     * Computes the correlation distance
     * @return  Results of the algorithm
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cObject, prec, method);
        return result;
    }

    /**
     * Registers user-allocated memory to store results of the correlation distance algorithm
     * @param result Object to store the results
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated correlation distance algorithm with a copy of input objects
     * of this correlation distance algorithm
     * @param context    Context to manage the correlation distance algorithm
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);

    private native long cClone(long cAlgorithm, int prec, int method);
}
