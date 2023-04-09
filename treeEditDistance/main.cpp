#include <iostream>
#include "tree/tree.hh"
#include "tree/tree_util.hh"
#include "ted.hpp"
#include "tee_helper.hpp"

using namespace std;

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
    // vector<vector<double>> costMatrix = {{0, 1, 1}, {1, 1, 1}, {1, 0, 1}};
    // display_matrix(costMatrix);
    // HungarianAlgorithm HungAlgo;
    // vector<int> assignment;
    // double cost = HungAlgo.Solve(costMatrix, assignment);
    // cout << cost << endl;

    // tree<int> t1;
    // auto root = t1.set_head(1);
    // auto i2 = t1.append_child(root, 2);
    // auto i3 = t1.append_child(root, 3);
    // t1.append_child(i2, 1);
    // t1.append_child(i2, 3);
    // t1.append_child(i3, 1);

    tree<int> t1 = build_tree<int>("1(2(1,3),3(1))");
    tree<int> t2 = build_tree<int>("1(1(1,2,3))");

    // tree<int> t2;
    // auto root2 = t2.set_head(1);
    // auto i4 = t2.append_child(root2, 1);
    // t2.append_child(i4, 1);
    // t2.append_child(i4, 2);
    // t2.append_child(i4, 3);

    kptree::print_tree_bracketed<int>(t1);
    std::cout << std::endl;
    kptree::print_tree_bracketed<int>(t2);
    std::cout << std::endl;

    // HungarianAlgorithm HungAlgo;
    std::cout << TreeEditDistance(t1, t2) << std::endl;
    return 0;
}