#pragma once

#include <iosfwd>
#include <mutex>
#include <string_view>

enum class LogLevel {
	TRACE,
	DEBUG,
	INFO,
	WARN,
	ERROR,
	FATAL
};

class Logger {
public:
	explicit Logger(std::ostream& output, LogLevel minimum_level = LogLevel::INFO);

	static LogLevel parse_level(std::string_view text);
	bool enabled(LogLevel level) const noexcept;
	void trace(std::string_view message);
	void debug(std::string_view message);
	void info(std::string_view message);
	void warn(std::string_view message);
	void error(std::string_view message);
	void fatal(std::string_view message);

private:
	void log(LogLevel level, std::string_view message);
	std::ostream& output_;
	const LogLevel minimum_level_;
	std::mutex output_mutex_;
};