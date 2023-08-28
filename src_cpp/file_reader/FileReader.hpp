#ifndef FILE_READER_HPP
#define FILE_READER_HPP

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>

namespace pendulum_acrobatics {

class FileReader {
 public:
  static std::vector<Eigen::VectorXd> readNumpyTxt(
      const std::string& file_path);
  static std::vector<float> readFloatTxt(const std::string& file_path);
};

}  // namespace pendulum_acrobatics
#endif  // FILE_READER_HPP
