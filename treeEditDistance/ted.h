#ifndef TED
#define TED

#include "tree.hpp"

vector<vector<double>> adjency_matrix(Tree<int> &t1, int root1, Tree<int> &t2, int root2);
double ted(Tree<int> &t1, int root1, Tree<int> &t2, int root2);

#endif