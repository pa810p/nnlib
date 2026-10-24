#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "logger.h"

TEST(LoggerTest, ParsesLevelsWithoutCaseSensitivity) {
	EXPECT_EQ(Logger::parse_level("debug"), LogLevel::DEBUG);
	EXPECT_EQ(Logger::parse_level("WARN"), LogLevel::WARN);
	EXPECT_EQ(Logger::parse_level("unknown"), LogLevel::INFO);
}

TEST(LoggerTest, FiltersMessagesBelowConfiguredLevel) {
	std::ostringstream output;
	Logger logger(output, LogLevel::INFO);

	logger.debug("epoch details");
	logger.info("training complete");

	EXPECT_EQ(output.str().find("epoch details"), std::string::npos);
	EXPECT_NE(output.str().find("[INFO] training complete"), std::string::npos);
}

TEST(LoggerTest, WritesDebugMessagesWhenEnabled) {
	std::ostringstream output;
	Logger logger(output, LogLevel::DEBUG);

	logger.debug("epoch=1 training_rmse=0.25");

	EXPECT_NE(output.str().find("[DEBUG] epoch=1 training_rmse=0.25"),
	          std::string::npos);
}