#include <iostream>
#include "Hungarian.h"
#include "tree.hpp"
#include "ted.h"

void display_matrix(const vector<vector<double>> &matrix)
{
    for (auto row : matrix)
    {
        for (auto cell : row)
        {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
}

int main(void)
{
    Tree<int> t1;
    int root1 = t1.insert(0, 1);
    int branch1 = t1.insert(root1, 2);
    t1.insert(branch1, 1);
    t1.insert(branch1, 3);
    int branch2 = t1.insert(root1, 3);
    t1.insert(branch2, 1);

    Tree<int> t2;
    int root2 = t2.insert(0, 1);
    int branch3 = t2.insert(root2, 1);
    t2.insert(branch3, 1);
    t2.insert(branch3, 2);
    t2.insert(branch3, 3);

    std::cout << "Tree 1 " << std::endl
              << t1 << std::endl
              << "Tree 2 " << std::endl
              << t2 << std::endl;

    double cost = ted(t1, root1, t2, root2);
    std::cout << "Cost : " << cost << std::endl;

    std::cout << "Tree 1 " << std::endl
              << t1 << std::endl
              << "Tree 2 " << std::endl
              << t2 << std::endl;

    return 0;
}