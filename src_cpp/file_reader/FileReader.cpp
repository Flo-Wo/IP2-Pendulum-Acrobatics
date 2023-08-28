#include "FileReader.hpp"

#include "iostream"

namespace pendulum_acrobatics {
std::vector<Eigen::VectorXd> FileReader::readNumpyTxt(
    const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "File cannot be opened." << std::endl;
  }

  std::vector<Eigen::VectorXd> data_vectors;
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    double value;
    Eigen::VectorXd data_vector;

    while (ss >> value) {
      // Read each value from the line and append it to the data_vector
      data_vector.conservativeResize(data_vector.size() + 1);
      data_vector(data_vector.size() - 1) = value;
    }

    data_vectors.push_back(data_vector);
  }

  file.close();
  return data_vectors;
};

std::vector<float> FileReader::readFloatTxt(const std::string& file_path) {
  std::ifstream file(file_path);

  std::vector<float> data;
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    float value;

    while (ss >> value) {
      // Read each value from the line and append it to the data vector
      data.push_back(value);

      // Check for a comma (CSV separator) and ignore it
      if (ss.peek() == ',') {
        ss.ignore();
      }
    }
  }

  file.close();
  return data;
};
}  // namespace pendulum_acrobatics

/*
int main() {
  pendulum_acrobatics::FileReader::readNumpyTxt("vectors.txt");
  return 0;
}
*/