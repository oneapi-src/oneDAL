/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

%import <std_string.i>

%pythoncode %{
import numpy
%}

%{
#define SWIG_FILE_WITH_INIT
%}

%include "numpy.i"

%init %{
import_array();
%}

%{
#include <daal.h>
#include <iostream>

using namespace daal;

// /////////////////////////////////////
// defines for generic typemaps

#if ((PY_VERSION_HEX <  0x02070000)                 \
     || ((PY_VERSION_HEX >= 0x03000000)             \
         && (PY_VERSION_HEX <  0x03010000)) )

template< typename T >
void del_daal_ptr(void * ptr)
{
    delete reinterpret_cast< T* >(ptr);
}
# define PyCapsule_New(pointer, name, destructor)    \
    (PyCObject_FromVoidPtr(pointer, destructor))
# define PyCapsule_GetPointer(capsule, name) \
    (PyCObject_AsVoidPtr(capsule))
#define PyCapsule_CheckExact(capsule)\
    (PyCObject_Check(capsule))

#else // PY_VERSION_HEX

template< typename T >
void del_daal_ptr(PyObject * ptr)
{
    void * tmp = PyCapsule_GetPointer(ptr, NULL);
    delete reinterpret_cast< T* >(tmp);
}

#endif // PY_VERSION_HEX

#if PY_VERSION_HEX < 0x03000000
#define PyUnicode_Check(_x) PyString_Check(_x)
#define PyUnicode_AsUTF8(_x) PyString_AsString(_x)
#define PyUnicode_FromString(_x) PyString_FromString(_x)
#endif
%}

// /////////////////////////////////////
// defines for generic typemaps

%inline %{
// our native type
#define NTYPE PyObject*
%}

%{
// native NULL/Nil value
#define NNULL Py_None

// GC - nothing to do
#define TMGC(_n)

// create a list object with named elements
#define MK_LIST(_res, _names, _cls, _gc) PyObject* _res = PyDict_New()

// create an object wrapping native pointer
#define MK_DAALPTR(_name, _ptr, _type, _gc) PyObject* _name = PyCapsule_New(_ptr, NULL, del_daal_ptr< _type >)

// set element in list
#define SET_ELT(_res, _i, _val, _names) PyDict_SetItem(_res, PyUnicode_FromString(_names[_i]), _val)

// /////////////////////////////////////

template< typename T > void set_sp_base(PyArrayObject * ary, daal::services::SharedPtr<T> & sp);

template<typename T>
NTYPE native_type(T & ptr, int & gc);

NTYPE native_type(data_management::DataCollectionPtr&, int&)
{
    return Py_None;
}

NTYPE native_type(daal::data_management::NumericTablePtr & ptr, int&)
{
    if(!ptr) return Py_None;
    npy_intp dims[2] = {static_cast<npy_intp>(ptr->getNumberOfRows()), static_cast<npy_intp>(ptr->getNumberOfColumns())};
    {
        auto dptr = dynamic_cast< data_management::HomogenNumericTable< double >* >(ptr.get());
        if(dptr) {
            daal::services::SharedPtr< double > data_tmp(dptr->getArraySharedPtr());
            PyObject* obj = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, (void*)data_tmp.get());
            if (!obj) throw std::invalid_argument("conversion to numpy array failed");
            set_sp_base((PyArrayObject*)obj, data_tmp);
            return obj;
        }
    }
    {
        auto dptr = dynamic_cast< data_management::HomogenNumericTable< int >* >(ptr.get());
        if(dptr) {
            daal::services::SharedPtr< int > data_tmp(dptr->getArraySharedPtr());
            PyObject* obj = PyArray_SimpleNewFromData(2, dims, NPY_INT, (void*)data_tmp.get());
            if (!obj) throw std::invalid_argument("conversion to numpy array failed");
            set_sp_base((PyArrayObject*)obj, data_tmp);
            return obj;
        }
    }
    {
        auto dptr = dynamic_cast< data_management::HomogenNumericTable< float >* >(ptr.get());
        if(dptr) {
            daal::services::SharedPtr< float > data_tmp(dptr->getArraySharedPtr());
            PyObject* obj = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, (void*)data_tmp.get());
            if (!obj) throw std::invalid_argument("conversion to numpy array failed");
            set_sp_base((PyArrayObject*)obj, data_tmp);
            return obj;
        }
    }
    throw std::invalid_argument("Encountered unsupported table type.");
}

NTYPE native_type(int s, int & gc)
{
    return PyLong_FromLong((long)s);
}
NTYPE native_type(unsigned int s, int & gc)
{
    return PyLong_FromUnsignedLong((unsigned long)s);
}
NTYPE native_type(size_t s, int & gc)
{
    return PyLong_FromSize_t(s);
}
NTYPE native_type(double s, int & gc)
{
    return PyFloat_FromDouble(s);
}
NTYPE native_type(float s, int & gc)
{
    return PyFloat_FromDouble((double)s);
}
NTYPE native_type(bool s, int & gc)
{
    return s ? Py_True : Py_False;
}
%}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="NumPy_Macros")
(const daal::data_management::NumericTablePtr)
{
    $1 = is_array($input) && PyArray_EquivTypenums(array_type($input), NPY_FLOAT64);
}
%typemap(in, fragment="NumPy_Fragments")
(const daal::data_management::NumericTablePtr)
{
    if($input == Py_None) {
        $1.reset();
    } else {
        int is_new_object = 0;
        PyArrayObject* array = obj_to_array_contiguous_allow_conversion($input, NPY_FLOAT64, &is_new_object);
        if (!array || !require_dimensions(array,2) || !require_contiguous(array) || !require_native(array)) {
            SWIG_fail;
        }
        // we provide the SharedPtr with a deleter which decrements the pyref
        $1.reset(new data_management::HomogenNumericTable<double>(daal::services::SharedPtr<double>((double*)array_data(array),
                                                                                                    NumpyDeleter(array)),
                                                                  (size_t)array_size(array,1),
                                                                  (size_t)array_size(array,0)));
        // we need it increment the ref-count if we use the $input array in-place
        // if we copied/converted it we already own our own reference
        if((PyObject*)array == $input) Py_INCREF(array);
    }
}

%wrapper %{
bool to_TableOrFList(PyObject * input, TableOrFList * tof)
{
    tof->table.reset();
    tof->tlist.resize(0);
    tof->file.resize(0);
    tof->flist.resize(0);
    if(input == Py_None) {
        ;
    } else if(is_array(input)) {
        int is_new_object = 0;
        PyArrayObject* array = obj_to_array_contiguous_allow_conversion(input, NPY_FLOAT64, &is_new_object);
        if (!array || !require_dimensions(array,2) || !require_contiguous(array) || !require_native(array)) {
            return true;
        }
        // we provide the SharedPtr with a deleter which decrements the pyref
        tof->table.reset(new data_management::HomogenNumericTable<double>(daal::services::SharedPtr<double>((double*)array_data(array),
                                                                                                           NumpyDeleter(array)),
                                                                         (size_t)array_size(array,1),
                                                                         (size_t)array_size(array,0)));
        // we need it increment the ref-count if we use the input array in-place
        // if we copied/converted it we already own our own reference
        if((PyObject*)array == input) Py_INCREF(array);
    } else if(PyList_Check(input) && PyList_Size(input) > 0) {
        PyObject * first = PyList_GetItem(input, 0);
        if(is_array(first)) {
            tof->tlist.resize(PyList_Size(input));
            for(auto i = 0; i < tof->tlist.size(); i++) {
                int is_new_object = 0;
                PyObject * el = PyList_GetItem(input, i);
                PyArrayObject* array = obj_to_array_contiguous_allow_conversion(el, NPY_FLOAT64, &is_new_object);
                if (!array || !require_dimensions(array,2) || !require_contiguous(array) || !require_native(array)) {
                    return true;
                }
                // we provide the SharedPtr with a deleter which decrements the pyref
                tof->tlist[i].reset(new data_management::HomogenNumericTable<double>(daal::services::SharedPtr<double>((double*)array_data(array),
                                                                                                                       NumpyDeleter(array)),
                                                                                     (size_t)array_size(array,1),
                                                                                     (size_t)array_size(array,0)));
                // we need it increment the ref-count if we use the input array in-place
                // if we copied/converted it we already own our own reference
                if((PyObject*)array == el) Py_INCREF(array);
            }
        } else if(PyUnicode_Check(first)) {
            tof->flist.resize(PyList_Size(input));
            for(auto i = 0; i < tof->flist.size(); i++) {
                tof->flist[i] = PyUnicode_AsUTF8(PyList_GetItem(input, i));
            }
        }
    } else if(PyUnicode_Check(input)) {
        //        tof->file = PyUnicode_AsUTF8AndSize(input, &size);
        tof->file =  PyUnicode_AsUTF8(input);
    }
    return false;
}
%}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY)
(const TableOrFList)
{
    $1 = (is_array($input) && PyArray_EquivTypenums(array_type($input), NPY_FLOAT64)) || PyList_Check($input) || PyUnicode_Check($input);
}
%typemap(in)
(const TableOrFList &)
(TableOrFList tof)
{
    $1 = &tof;
    if(to_TableOrFList($input, $1)) {
        SWIG_fail;
    }
}

%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY)
(const daal::algorithms::classifier::ModelPtr)
{
    $1 = PyDict_Check($input);
}
%typemap(in)
(const daal::algorithms::classifier::ModelPtr)
{
    if($input == Py_None) {
        $1.reset();
    } else {
        PyObject* tmp = PyDict_GetItem($input, PyUnicode_FromString("__daalptr__"));
        $1 = *reinterpret_cast< $1_ltype* >(PyCapsule_GetPointer(tmp, NULL));
        assert($1);
    }
}
%apply(const daal::algorithms::classifier::ModelPtr) {
    (const daal::algorithms::svm::ModelPtr),
    (const daal::algorithms::multinomial_naive_bayes::ModelPtr),
    (const daal::algorithms::linear_regression::ModelPtr),
    (const daal::algorithms::multi_class_classifier::ModelPtr)
};
