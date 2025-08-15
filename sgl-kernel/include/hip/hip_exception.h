/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SGL_HIP_EXCEPTION_H_
#define SGL_HIP_EXCEPTION_H_

#include <exception>
#include <sstream>

#define SGL_HIP_ERROR(message) throw sgl_hip::Error(__FUNCTION__, __FILE__, __LINE__, message)

template <typename T>
void write_to_stream(std::ostringstream& oss, T&& val) {
  oss << std::forward<T>(val);
}

template <typename T, typename... Args>
void write_to_stream(std::ostringstream& oss, T&& val, Args&&... args) {
  oss << std::forward<T>(val) << " ";
  write_to_stream(oss, std::forward<Args>(args)...);
}

#define SGL_HIP_CHECK(condition, ...)    \
  if (!(condition)) {                    \
    std::ostringstream oss;              \
    write_to_stream(oss, __VA_ARGS__);   \
    std::cerr << oss.str() << std::endl; \
    SGL_HIP_ERROR(oss.str());            \
  }

namespace sgl_hip {
class Error : public std::exception {
 private:
  std::string message_;

 public:
  Error(const std::string& func, const std::string& file, int line, const std::string& message) {
    std::ostringstream oss;
    oss << "Error in function '" << func << "' "
        << "at " << file << ":" << line << ": " << message;
    message_ = oss.str();
  }

  virtual const char* what() const noexcept override {
    return message_.c_str();
  }
};

}  // namespace sgl_hip

#endif  // SGL_HIP_EXCEPTION_H_
