#ifndef tee_helper
#define tee_helper

#include "lib/tree/tree.hh"
#include <string>
#include <iostream>
#include <sstream>

// Recursive function to build tree
template <typename T>
tree<T> build_tree_helper(std::stringstream &ss)
{
    T data;
    ss >> data; // read current node value

    auto t = tree<T>(data); // create new tree with current node value

    if (ss.peek() == '(')
    { // check if current node has children
        char c;
        ss >> c; // discard opening bracket

        while (true)
        {
            if (ss.peek() == ')')
            {            // check for end of children
                ss >> c; // discard closing bracket
                break;
            }
            else if (ss.peek() == ',')
            {            // check for multiple children
                ss >> c; // discard comma
            }

            // recursively parse child node
            auto child = build_tree_helper<T>(ss);

            // add child to parent node
            t.append_child(t.begin(), child.begin());
        }
    }

    return t;
}

// Function to parse string and build tree
template <typename T>
tree<T> build_tree(const std::string &s)
{
    std::stringstream ss(s);
    return build_tree_helper<T>(ss);
}

#endif