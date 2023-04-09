#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <string>
#include <iostream>
#include "lib/tree/tree.hh"
#include "ted.hpp"
#include "tee_helper.hpp"
#include <numpy/arrayobject.h>

static PyObject *compute(PyObject *self, PyObject *args)
{
    import_array();
    PyObject *list1;
    PyObject *list2;

    if (!PyArg_ParseTuple(args, "OO", &list1, &list2))
    {
        return NULL;
    }

    // Convert input lists to C arrays for better performance
    int len1 = PyList_Size(list1);
    std::vector<tree<int>> arr1(len1);
    for (int i = 0; i < len1; ++i)
        arr1[i] = build_tree<int>(std::string(PyUnicode_AsUTF8(PyList_GetItem(list1, i))));

    int len2 = PyList_Size(list2);
    std::vector<tree<int>> arr2(len2);
    for (int i = 0; i < len2; ++i)
        arr2[i] = build_tree<int>(std::string(PyUnicode_AsUTF8(PyList_GetItem(list2, i))));

    // Allocate memory for the result matrix
    npy_intp dims[] = {len1, len2};
    PyObject *distancesObj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    PyArrayObject *distancesArr = reinterpret_cast<PyArrayObject *>(distancesObj);
    double *distances = reinterpret_cast<double *>(PyArray_DATA(distancesArr));

    for (int i = 0; i < len1; ++i)
    {
        for (int j = 0; j < len2; ++j)
        {
            double d = TreeEditDistance(arr1[i], arr2[j]);
            distances[i * len2 + j] = d;
        }
    }

    return distancesObj;
}

static PyMethodDef treeEditDistance_methods[] = {
    {"treeEditDistance", compute, METH_VARARGS, "Compute the tree edit distance."},
    {NULL, NULL}};

static struct PyModuleDef treeEditDistanceModule = {
    PyModuleDef_HEAD_INIT,
    "treeEditDistance",
    "A module to efficiently compute tree Edit Distance between two trees.",
    -1,
    treeEditDistance_methods};

PyMODINIT_FUNC PyInit_treeEditDistance(void)
{
    return PyModule_Create(&treeEditDistanceModule);
}
