#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>
#include "lib/tree/tree.hh"
#include "ted.hpp"
#include "tee_helper.hpp"

static PyObject *compute(PyObject *self, PyObject *args)
{
    const char *str1;
    const char *str2;

    if (!PyArg_ParseTuple(args, "ss", &str1, &str2))
    {
        return NULL;
    }

    const std::string s1(str1);
    const std::string s2(str2);

    tree<int> t1 = build_tree<int>(s1);
    tree<int> t2 = build_tree<int>(s2);

    double distance = TreeEditDistance(t1, t2);

    return Py_BuildValue("s", distance);
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
