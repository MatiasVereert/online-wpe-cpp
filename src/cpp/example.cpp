#include <iostream>

int main() {
    int a = 5;
    // Pre-increment: 'a' becomes 6, then 'b' is assigned 6
    int b = ++a; 
    
    int x = 5;
    // Post-increment: 'y' is assigned 5, then 'x' becomes 6
    int y = x++; 
    
    return 0;
}