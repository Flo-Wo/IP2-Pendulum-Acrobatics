#include "Logger.hpp"

LogLevel Logger::globalLogLevel;
void Logger::setLogLevel(LogLevel level) { globalLogLevel = level; }

std::string Logger::logLevelString(LogLevel level) {
  switch (level) {
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARNING:
      return "WARNING";
    case LogLevel::ERROR:
      return "ERROR";
    case LogLevel::SILENT:
      return "SILENT";
    default:
      return "UNKNOWN";
  }
}

void Logger::log(LogLevel level, const std::string& message) {
  if (level >= globalLogLevel) {
    std::cout << "[" << logLevelString(level) << "] " << message << std::endl;
  }
}
std::string Logger::matrixToString(const Eigen::MatrixXd& matrix) {
  std::stringstream oss;
  oss << matrix;
  return oss.str();
}
