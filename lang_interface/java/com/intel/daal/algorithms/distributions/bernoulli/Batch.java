/* file: Batch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
 * @defgroup distributions_bernoulli_batch Batch
 * @ingroup distributions_bernoulli
 * @{
 */
/**
 * @brief Contains classes for the bernoulli distribution
 */
package com.intel.daal.algorithms.distributions.bernoulli;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.distributions.Input;
import com.intel.daal.algorithms.distributions.Result;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DISTRIBUTIONS__BERNOULLI__BATCH"></a>
 * \brief Provides methods for bernoulli distribution computations in the batch processing mode
 *
 * \par References
 *      - @ref com.intel.daal.algorithms.distributions.Input class
 */
public class Batch extends com.intel.daal.algorithms.distributions.BatchBase {
    public  Parameter    parameter; /*!< Parameters of the bernoulli distribution */
    public  Method       method;    /*!< Computation method for the distribution */
    private Precision    prec;      /*!< Data type to use in intermediate computations for the distribution */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs bernoulli distribution by copying input objects and parameters of another bernoulli distribution
     * @param context Context to manage the bernoulli distribution
     * @param other   A distributions to be used as the source to initialize the input objects
     *                and parameters of this distribution
     */
    public Batch(DaalContext context, Batch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new Input(context, cGetInput(cObject));
        parameter = new Parameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the bernoulli distribution
     * @param context    Context to manage the distribution
     * @param cls        Data type to use in intermediate computations for the distribution, Double.class or Float.class
     * @param method     The distribution computation method, @ref Method
     * @param p          Success probability of a trial, value from [0.0; 1.0]
     */
    public Batch(DaalContext context, Class<? extends Number> cls, Method method, double p) {
        super(context);
        constructBatch(context, cls, method, p);
    }

    private void constructBatch(DaalContext context, Class<? extends Number> cls, Method method, double p) {
        this.method = method;

        if (method != Method.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue(), p);
        input = new Input(context, cGetInput(cObject));
        parameter = new Parameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        parameter.setP(p);
    }

    /**
     * Computes the result of the bernoulli distribution
     * @return  Bernoulli distribution result
     */
    @Override
    public Result compute() {
        super.compute();
        return new Result(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the newly allocated bernoulli distribution
     * with a copy of input objects and parameters of this bernoulli distribution
     * @param context    Context to manage the distribution
     * @return The newly allocated bernoulli distribution
     */
    @Override
    public Batch clone(DaalContext context) {
        return new Batch(context, this);
    }

    private native long cInit(int prec, int method, double p);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
