#include "cuda_runtime.h"
#include "kernel.cuh"
#include "blob.cuh"
#include <memory>

using namespace std;

int main() {
    int n = 1 << 30;
    auto A = make_shared<Blob>(n);
    float res;
    A->valueInit(1.f);
    reduceSum(A->gpu(), &res, A->size());
    cout << res << endl;
    return 0;
}