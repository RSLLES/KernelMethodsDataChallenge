#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string>
#include "lib/Capted.h"

using namespace capted;

static PyObject *apted_compute(PyObject *self, PyObject *args)
{
    const char *str1;
    const char *str2;

    if (!PyArg_ParseTuple(args, "ss", &str1, &str2))
    {
        return NULL;
    }

    const std::string t1(str1);
    const std::string t2(str2);

    StringCostModel costModel;
    Apted<StringNodeData> algorithm(&costModel);
    BracketStringInputParser p1(t1);
    BracketStringInputParser p2(t2);
    Node<StringNodeData> *n1 = p1.getRoot();
    Node<StringNodeData> *n2 = p2.getRoot();

    float compDist = algorithm.computeEditDistance(n1, n2);
    float compDist = 0.0;

    return Py_BuildValue("s", compDist);
}

static PyMethodDef aptedmodule_methods[] = {
    {"compute", apted_compute, METH_VARARGS, "Compute the apted distance"},
    {NULL, NULL}};

static struct PyModuleDef aptedModule = {
    PyModuleDef_HEAD_INIT,
    "myModule",
    "A simple module that concatenates two strings.",
    -1,
    aptedmodule_methods};

PyMODINIT_FUNC PyInit_aptedModule(void)
{
    return PyModule_Create(&aptedModule);
}
