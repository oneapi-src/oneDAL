/* file: npy_aos_numeric_table.h */
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

/*
//++
//  Implementation of a heterogeneous table stored as a structure of arrays.
//--
*/

#ifndef __NUMPY_AOS_NUMERIC_TABLE_H__
#define __NUMPY_AOS_NUMERIC_TABLE_H__

#include <string.h>
#include <iostream>
#include "npy_utils.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__AOSNUMERICTABLE"></a>
 *  \brief Class that provides methods to access data stored as a numpy structured array.
 *         E.g. as an array of heterogeneous feature vectors, while each feature vector
 *         is represented by a python tuple.
 *         Therefore, the data is represented as a 1-dimensional numpy array of structures.
 */
class NpyAOSNumericTable : public NumericTable
{
protected:
    PyArrayObject * _ary;
public:

    /**
     *  Constructor for a Numeric Table with user-allocated memory
     *  \param[in]  ary  The  structured array
     */
    NpyAOSNumericTable(PyArrayObject * ary)
        : NumericTable(NumericTableDictionaryPtr()),
          _ary(ary)
    {
        Py_XINCREF(ary);
        _layout = aos;
        // we assume numpy.i has done typechecks and this is a 1-dimensional structured array
        // e.g. each element is a tuple.
        PyArray_Descr * descr = PyArray_DESCR(ary);              // type descriptor

        if(!descr->names) {
            std::cerr << "No dtype argument provided. Unable to create AOSNumericTable" << std::endl;
            this->_status.add(services::ErrorIncorrectTypeOfInputNumericTable);
            return;
        }

        PyObject * fnames = PySequence_Fast(descr->names, NULL); // list of names of tuple-elements
        Py_ssize_t N = PySequence_Fast_GET_SIZE(fnames);         // number of elements in tuple

        if( _ddict.get() == NULL ) {
            _ddict = NumericTableDictionaryPtr(new NumericTableDictionary(N));
        }
        setNumberOfRows(PyArray_DIMS(ary)[0]);
        // setNumberOfColumns not needed, done by providing size to ddict

        // iterate through all elements in tuple
        // get their type and init ddict feature accordingly
        for (Py_ssize_t i=0; i<N; i++) {
            PyObject * name = PySequence_Fast_GET_ITEM(fnames, i);  // tuple elements are identified by name
            PyObject * ftr = PyObject_GetItem(descr->fields, name); // desr->fields is a dict
            if(!PyTuple_Check(ftr)) {
                std::cerr << "Not a tuple: " << ftr << " is a " << PyString_AsString(PyObject_Str(PyObject_Type(ftr))) << "\n.";
                this->_status.add(services::ErrorIncorrectTypeOfInputNumericTable);
                return;
            }
            PyArray_Descr *id = NULL;
            // here we convert the dtype string into type descriptor
            if (PyArray_DescrConverter(PyTuple_GetItem(ftr, 0), &id) != NPY_SUCCEED) {
                std::cerr << "Couldn't get typedescr\n.";
                this->_status.add(services::ErrorIncorrectTypeOfInputNumericTable);
                return;
            }
#define SETFEATURE_(_T) _ddict->setFeature<_T>(i)
            SET_NPY_FEATURE( id->type, SETFEATURE_ );
        }
    }

    /** \private */
    ~NpyAOSNumericTable()
    {
        Py_XDECREF(_ary);
    }

    virtual int getSerializationTag() const
    {
        return SERIALIZATION_AOS_NT_ID;
    }

    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, vector_num, rwflag, block);
    }
    services::Status getBlockOfRows(size_t vector_idx, size_t vector_num, ReadWriteMode rwflag, BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, vector_num, rwflag, block);
    }

    services::Status releaseBlockOfRows(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<double>(block);
    }
    services::Status releaseBlockOfRows(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<float>(block);
    }
    services::Status releaseBlockOfRows(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<int>(block);
    }

    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<double>(vector_idx, value_num, rwflag, block, feature_idx, 1 );
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<float>(vector_idx, value_num, rwflag, block, feature_idx, 1);
    }
    services::Status getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, ReadWriteMode rwflag,
                                BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return getTBlock<int>(vector_idx, value_num, rwflag, block, feature_idx, 1);
    }

    services::Status releaseBlockOfColumnValues(BlockDescriptor<double>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<double>(block);
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<float>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<float>(block);
    }
    services::Status releaseBlockOfColumnValues(BlockDescriptor<int>& block) DAAL_C11_OVERRIDE
    {
        return releaseTBlock<int>(block);
    }

    services::Status allocateDataMemory(daal::MemType type = daal::dram) DAAL_C11_OVERRIDE
    {
        return services::Status(services::ErrorMethodNotSupported);
    }

    void freeDataMemory() DAAL_C11_OVERRIDE
    {
        services::Status ec(services::ErrorMethodNotSupported);
    }

    /** \private */
    services::Status serializeImpl  (InputDataArchive  *archive)
    {
        // First serialize the type descriptor in string representation
        Py_ssize_t len = 0;
#if PY_MAJOR_VERSION < 3
        char * ds = NULL;
        PyString_AsStringAndSize(PyObject_Repr((PyObject*)PyArray_DESCR(_ary)), &ds, &len);
#else
        char * ds = PyUnicode_AsUTF8AndSize(PyObject_Repr((PyObject*)PyArray_DESCR(_ary)), &len);
#endif
        if( ds == NULL ) {
            this->_status.add(services::UnknownError);
            return services::Status();
        }
        archive->set( len );
        if( len == 0 ) return services::Status();
        archive->set( ds, len );
        // now the array data
        archive->set( PyArray_DIMS(_ary)[0] );
        archive->set( (char*)PyArray_DATA(_ary), PyArray_DIMS(_ary)[0] );

        return services::Status();
    }

    /** \private */
    services::Status deserializeImpl(const OutputDataArchive *archive)
    {
        // First deserialize the type descriptor in string representation...
        size_t len;
        archive->set( len );
        char * nds = new char[len];
        archive->set( nds, len );
        // ..then create the type descriptor
        PyObject * npy = PyImport_ImportModule("numpy");
        PyObject * globalDictionary = PyModule_GetDict(npy);
        PyArray_Descr* nd = (PyArray_Descr*)PyRun_String(PyString_AsString(PyObject_Str(PyString_FromString(nds))), Py_eval_input, globalDictionary,
                                                         NULL);
        delete [] nds;
        if( nd == NULL ) {
            this->_status.add(services::UnknownError);
            return services::Status();
        }
        // now get the array data
        npy_intp dim;
        archive->set( dim );
        npy_intp dims[] = {dim};
        // create the array...
        _ary = (PyArrayObject*)PyArray_SimpleNewFromDescr(1, dims, nd);
        if( _ary == NULL ) {
            this->_status.add(services::UnknownError);
            return services::Status();
        }
        // ...then copy data
        archive->set( (char*)PyArray_DATA(_ary), dims[0] );

        return services::Status();
    }

private:

    // this is a generic copy function
    // set template parameter Down to true for down-casts, to false for upcasts
    template<typename T>
    void do_cpy(BlockDescriptor<T>& block, size_t startcol, size_t endcol, size_t startrow, size_t nrows, bool down)
    {
        // tuple elements are identified by name, need the list of names
        PyObject * fnames = PySequence_Fast(PyArray_DESCR(_ary)->names, NULL);
        for( long j = startcol ; j < endcol ; j++ ) {
            PyObject * name = PySequence_Fast_GET_ITEM(fnames, j);
            // get column by name
            PyArrayObject * col = (PyArrayObject *)PyObject_GetItem((PyObject *)_ary, name); assert(col);
            // need the descriptor to create an iterator
            PyArray_Descr * dtype = PyArray_DTYPE(col); assert(dtype);
            // get an iterator for the column
            NpyIter * iter = NpyIter_New(col, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_SAME_KIND_CASTING, dtype); assert(iter);
            NpyIter_IterNextFunc * iternext = NpyIter_GetIterNext(iter, NULL);
            // fast forward to first element we want
            NpyIter_GotoIterIndex(iter, startrow);
            size_t n = 0;
            // ptr to column in block
            T * blockPtr = block.getBlockPtr() + j - startcol;
            // feature for column
            NumericTableFeature &f = (*_ddict)[j];
            // iterate through column, use casting functions to upcast, dataptr will point to current element
            void ** dataptr = (void **) NpyIter_GetDataPtrArray(iter);
            if( down ) { // could be templeate arg to eliminate conditional, prefer smaller binaires for now
                do {
                    data_feature_utils::getVectorDownCast(f.indexType,
                                                       data_feature_utils::getInternalNumType<T>())(1,
                                                                                                    blockPtr + n*block.getNumberOfColumns(),
                                                                                                    *dataptr);
                    ++n;
                } while (iternext(iter) && n < nrows);
            } else {
                do {
                    data_feature_utils::getVectorUpCast(f.indexType,
                                                     data_feature_utils::getInternalNumType<T>())(1,
                                                                                                  *dataptr,
                                                                                                  blockPtr + n*block.getNumberOfColumns());
                    ++n;
                } while (iternext(iter) && n < nrows);
            }
            // deallocate iterator
            NpyIter_Deallocate(iter);
        }
    }

    template <typename T>
    services::Status getTBlock( size_t idx, size_t numrows, int rwFlag, BlockDescriptor<T>& block, size_t firstcol=0, size_t numcols=0xffffffff )
    {
        // sanitize bounds
        const size_t ncols = firstcol + numcols <= getNumberOfColumns() ? numcols : getNumberOfColumns() - firstcol;
        const size_t nrows = idx + numrows <= getNumberOfRows()         ? numrows : getNumberOfRows() - idx;

        // set shape of blockdescr
        block.setDetails( firstcol, idx, rwFlag );

        if (idx >= getNumberOfRows() || firstcol >= getNumberOfColumns() ) {
            block.resizeBuffer( ncols, 0 );
            return services::Status();
        }

        if( !block.resizeBuffer( ncols, nrows ) ) {
            return services::Status(services::ErrorMemoryAllocationFailed);
        }

        if( !(rwFlag & (int)readOnly) ) return services::Status();

        // use our copy method in upcast mode
        do_cpy(block, firstcol, firstcol+ncols, idx, nrows, false);
        return services::Status();
    }


    template <typename T>
    services::Status releaseTBlock( BlockDescriptor<T>& block )
    {
        if(block.getRWFlag() & (int)writeOnly) {
            const size_t ncols = block.getNumberOfColumns();
            const size_t nrows = block.getNumberOfRows();

            // use our copy method in downcast mode
            do_cpy(block, block.getColumnsOffset(), block.getColumnsOffset() + ncols, 0, nrows, true);

            block.reset();
        }
        return services::Status();
    }
};
} // namespace interface1
using interface1::NpyAOSNumericTable;

}
} // namespace daal
#endif // __NUMPY_AOS_NUMERIC_TABLE_H__
