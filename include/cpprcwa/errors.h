#pragma once

#include <stdexcept>
#include <string>

namespace cpprcwa {
namespace error {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& msg) : std::runtime_error(msg) {}
};

class SingularMatrixError : public Error {
public:
    SingularMatrixError(const std::string& fn, int info, int layer)
        : Error(fn + ": LAPACK factorization failed info=" + std::to_string(info)
                + " at layer=" + std::to_string(layer)
                + " (kp singular — consider a larger Qabs regularization)")
        , info_(info), layer_(layer) {}
    int info()  const { return info_; }
    int layer() const { return layer_; }
private:
    int info_;
    int layer_;
};

class LapackError : public Error {
public:
    LapackError(const std::string& fn, const std::string& routine, int info)
        : Error(fn + ": LAPACK " + routine + " returned info=" + std::to_string(info))
        , info_(info) {}
    int info() const { return info_; }
private:
    int info_;
};

class NotImplementedError : public Error {
public:
    explicit NotImplementedError(const std::string& what)
        : Error("not implemented: " + what) {}
};

class ConfigError : public Error {
public:
    explicit ConfigError(const std::string& what) : Error("config error: " + what) {}
};

} // namespace error
} // namespace cpprcwa
