#include "logger.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <ctime>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

namespace {

std::string_view level_name(LogLevel level) {
	switch (level) {
	case LogLevel::TRACE: return "TRACE";
	case LogLevel::DEBUG: return "DEBUG";
	case LogLevel::INFO: return "INFO";
	case LogLevel::WARN: return "WARN";
	case LogLevel::ERROR: return "ERROR";
	case LogLevel::FATAL: return "FATAL";
	}
	return "UNKNOWN";
}

std::tm local_time(std::time_t value) {
	std::tm result{};
#if defined(_WIN32)
	localtime_s(&result, &value);
#else
	localtime_r(&value, &result);
#endif
	return result;
}

}

Logger::Logger(std::ostream& output, LogLevel minimum_level)
	: output_(output), minimum_level_(minimum_level) {}

LogLevel Logger::parse_level(std::string_view text) {
	std::string normalized(text);
	std::transform(normalized.begin(), normalized.end(), normalized.begin(),
	               [](unsigned char value) {
		               return static_cast<char>(std::toupper(value));
	               });
	if (normalized == "TRACE") return LogLevel::TRACE;
	if (normalized == "DEBUG") return LogLevel::DEBUG;
	if (normalized == "INFO") return LogLevel::INFO;
	if (normalized == "WARN") return LogLevel::WARN;
	if (normalized == "ERROR") return LogLevel::ERROR;
	if (normalized == "FATAL") return LogLevel::FATAL;
	return LogLevel::INFO;
}

bool Logger::enabled(LogLevel level) const noexcept {
	return static_cast<int>(level) >= static_cast<int>(minimum_level_);
}

void Logger::trace(std::string_view message) { log(LogLevel::TRACE, message); }
void Logger::debug(std::string_view message) { log(LogLevel::DEBUG, message); }
void Logger::info(std::string_view message) { log(LogLevel::INFO, message); }
void Logger::warn(std::string_view message) { log(LogLevel::WARN, message); }
void Logger::error(std::string_view message) { log(LogLevel::ERROR, message); }
void Logger::fatal(std::string_view message) { log(LogLevel::FATAL, message); }

void Logger::log(LogLevel level, std::string_view message) {
	if (!enabled(level))
		return;

	const auto now = std::chrono::system_clock::now();
	const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
		now.time_since_epoch()) % 1000;
	const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
	const std::tm time = local_time(timestamp);
	std::ostringstream line;
	line << '[' << std::put_time(&time, "%Y-%m-%d %H:%M:%S") << '.'
	     << std::setfill('0') << std::setw(3) << milliseconds.count()
	     << "] [" << level_name(level) << "] " << message << '\n';

	const std::string formatted = line.str();
	std::lock_guard lock(output_mutex_);
	output_ << formatted;
	output_.flush();
}