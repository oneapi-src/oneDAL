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
 * @brief Contains classes to run the sorting
 */
package com.intel.daal.algorithms.sorting;

import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SORTING__BATCH"></a>
 * @brief Sorts data in the batch processing mode.
 * \n<a href="DAAL-REF-SORTING-ALGORITHM">Sorting algorithm description and usage models</a>
 *
 * @tparam algorithmFPType  Data type to use in intermediate computations for the sorting, double or float
 * @tparam method           Sorting computation method, @ref daal::algorithms::sorting::Method
 *
 * @par Enumerations
 *      - @ref Method   Sorting computation methods
 *      - @ref InputId  Identifiers of sorting input objects
 *      - @ref ResultId Identifiers of sorting results
 *
 * @par References
 *      - Input class
 *      - Result class
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
     * Constructs sorting algorithm by copying input objects and parameters
     * of another sorting algorithm
     * @param context      Context to manage the sorting
     * @param other        An algorithm to be used as the source to initialize the input objects
     *                     and parameters of the algorithm
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), this.method.getValue());
        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * <a name="DAAL-METHOD-ALGORITHM__SORTING__BATCH__BATCH"></a>
     *  @brief Sorts data in the batch processing mode
     * @param context      Context to manage the sorting
     * @param cls          Data type to use in intermediate computations for the sorting algorithms, Double.class or Float.class
     * @param method       Sorting computation methods, @ref Method
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method) {
        super(context);
        this.method = method;
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (this.method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), this.method.getValue());

        input = new Input(getContext(), cGetInput(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Runs the sorting algorithm
     * @return  Sorting computation results
     */
    @Override
    public Result compute() {
        super.compute();
        Result result = new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store results of the sorting
     * @param result    Structure to store results of the sorting
     */
    public void setResult(Result result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the newly allocated sorting algorithm
     * with a copy of input objects and parameters of this sorting algorithm
     * @param context      Context to manage the sorting
     *
     * @return The newly allocated algorithm
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method);

    private native long cGetInput(long algAddr, int prec, int method);

    private native long cGetResult(long algAddr, int prec, int method);

    private native void cSetResult(long cObject, int prec, int method, long cResult);

    private native long cClone(long algAddr, int prec, int method);
}
