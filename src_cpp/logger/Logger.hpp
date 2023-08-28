#pragma once

#include <iostream>
#include <string>

#include "Eigen/Core"

enum class LogLevel { DEBUG, INFO, WARNING, ERROR, SILENT };

class Logger {
 public:
  static void setLogLevel(LogLevel level);
  static std::string logLevelString(LogLevel level);
  static void log(LogLevel level, const std::string& message);
  static std::string matrixToString(const Eigen::MatrixXd& matrix);

 private:
  static LogLevel globalLogLevel;  // Declare static member variable
};
