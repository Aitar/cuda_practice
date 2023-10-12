#ifndef MYNN_UTILS_H
#define MYNN_UTILS_H


#include <memory>
#include <sstream>
#include <stdexcept>


#define CUDACHECK(op) cudaCheck((op), #op, __FILE__, __LINE__)
#define NIL(ptr) checkNullptr((ptr), __FILE__, __LINE__)

#define INFO(msg) logger((msg), 0)
#define WARN(msg) logger((msg), 1)
#define ERR(msg) logger((msg), 2)


static int INFO_RANK = 0;

void logger(std::basic_string<char, std::char_traits<char>, std::allocator<char>> msg, int type) {
    if (type < INFO_RANK) return;
    switch (type) {
        case 0: // info
            std::cout << "\033[32m[INFO] " << msg << "\033[0m" << std::endl;
            break;

        case 1: // warning
            std::cout << "\033[33m[WARNING] " << msg << "\033[0m" << std::endl;
            break;

        case 2: // error
            std::cout << "\033[31m[ERROR] " << msg << "\033[0m" << std::endl;
            break;

        default:
            std::cout << msg << std::endl;
    }
}

void cudaCheck(cudaError_t code, const char *op, const char *file, int line) {
    if (code != cudaSuccess) {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("cuda runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name,
               err_message);
        throw std::runtime_error("cuda_error");
    }
}



template <typename T>
T checkNullptr(T ptr, const char *file, int line) {
    if (ptr == nullptr) {
        std::stringstream ss;
        ss << "Null printer got at file " << file << ":" << line << ".";
        ERR(ss.str());
        exit(1);
    }
    return ptr;
}

#endif //MYNN_UTILS_H
