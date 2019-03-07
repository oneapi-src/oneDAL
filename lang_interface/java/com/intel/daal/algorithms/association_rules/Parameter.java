/* file: Parameter.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/**
 * @ingroup association_rules
 * @{
 */
package com.intel.daal.algorithms.association_rules;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ASSOCIATION_RULES__PARAMETER"></a>
 * @brief Parameters for the association rules compute method
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Parameter(DaalContext context, long cParameter) {
        super(context);
        this.cObject = cParameter;
    }

    /**
     * Sets the minimum support
     * @param val   Minimum support, 0.0 <= val < 1.0
     */
    public void setMinSupport(double val) {
        cSetMinSupport(this.cObject, val);
    }

    /**
     * Gets the minimum support
     * @return  Minimum support, 0.0 <= val < 1.0
     */
    public double getMinSupport() {
        return cGetMinSupport(this.cObject);
    }

    /**
     * Sets the minimum confidence
     * @param val   Minimum confidence, 0.0 <= val < 1.0
     */
    public void setMinConfidence(double val) {
        cSetMinConfidence(this.cObject, val);
    }

    /**
     * Gets the minimum confidence
     * @return  Minimum confidence, 0.0 <= val < 1.0
     */
    public double getMinConfidence() {
        return cGetMinConfidence(this.cObject);
    }

    /**
     * Sets the number of unique items
     * @param val    Number of unique items
     */
    public void setNUniqueItems(long val) {
        cSetNUniqueItems(this.cObject, val);
    }

    /**
     * Gets the number of unique items
     * @return   Number of unique items
     */
    public long getNUniqueItems() {
        return cGetNUniqueItems(this.cObject);
    }

    /**
     * Sets the number of transactions
     * @param val    Number of transactions
     */
    public void setNTransactions(long val) {
        cSetNTransactions(this.cObject, val);
    }

    /**
     * Gets the number of transactions
     * @return   Number of transactions
     */
    public long getNTransactions() {
        return cGetNTransactions(this.cObject);
    }

    /**
     * Sets the discoverRules flag
     * @param flag    The flag. If true, association rules are built from large itemsets
     */
    public void setDiscoverRules(boolean flag) {
        cSetDiscoverRules(this.cObject, flag);
    }

    /**
     * Gets the discoverRules flag
     * @return The flag
     */
    public boolean getDiscoverRules() {
        return cGetDiscoverRules(this.cObject);
    }

    /**
     * Sets the minimum number of items in large itemsets
     * @param val    Minimum number of items in large itemsets
     */
    public void setMinItemsetSize(long val) {
        cSetMinItemsetSize(this.cObject, val);
    }

    /**
     * Gets the minimum number of items in large itemsets
     * @return   Minimum number of items in large itemsets
     */
    public long getMinItemsetSize() {
        return cGetMinItemsetSize(this.cObject);
    }

    /**
     * Sets the maximum number of items in large itemsets
     * @param val    Maximum number of items in large itemsets
     */
    public void setMaxItemsetSize(long val) {
        cSetMaxItemsetSize(this.cObject, val);
    }

    /**
     * Gets the maximum number of items in large itemsets
     * @return   Maximum number of items in large itemsets
     */
    public long getMaxItemsetSize() {
        return cGetMaxItemsetSize(this.cObject);
    }

    /**
     * Sets the order option for the resulting itemsets
     * @param id    Order identifier, @ref ItemsetsOrderId
     */
    public void setItemsetsOrder(ItemsetsOrderId id) {
        cSetItemsetsOrder(this.cObject, id.getValue());
    }

    /**
     * Gets the order option for the resulting itemsets
     * @return    Order identifier, @ref ItemsetsOrderId
     */
    public ItemsetsOrderId getItemsetsOrder() {
        ItemsetsOrderId id = new ItemsetsOrderId(cGetItemsetsOrder(this.cObject));
        return id;
    }

    /**
     * Sets the order option for the resulting association rules
     * @param id    Order identifier, @ref RulesOrderId
     */
    public void setRulesOrder(RulesOrderId id) {
        cSetRulesOrder(this.cObject, id.getValue());
    }

    /**
     * Gets the order option for the resulting association rules
     * @return    Order identifier, @ref RulesOrderId
     */
    public RulesOrderId getRulesOrder() {
        RulesOrderId id = new RulesOrderId(cGetRulesOrder(this.cObject));
        return id;
    }

    private native void cSetMinSupport(long parAddr, double val);

    private native double cGetMinSupport(long parAddr);

    private native void cSetMinConfidence(long parAddr, double val);

    private native double cGetMinConfidence(long parAddr);

    private native void cSetNUniqueItems(long parAddr, long val);

    private native long cGetNUniqueItems(long parAddr);

    private native void cSetNTransactions(long parAddr, long val);

    private native long cGetNTransactions(long parAddr);

    private native void cSetDiscoverRules(long parAddr, boolean flag);

    private native boolean cGetDiscoverRules(long parAddr);

    private native void cSetMinItemsetSize(long parAddr, long val);

    private native long cGetMinItemsetSize(long parAddr);

    private native void cSetMaxItemsetSize(long parAddr, long val);

    private native long cGetMaxItemsetSize(long parAddr);

    private native void cSetItemsetsOrder(long parAddr, int id);

    private native int cGetItemsetsOrder(long parAddr);

    private native void cSetRulesOrder(long parAddr, int id);

    private native int cGetRulesOrder(long parAddr);

}
/** @} */
