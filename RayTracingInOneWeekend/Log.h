#pragma once
#include <memory>
#include <spdlog/spdlog.h>

class Log
{
	inline static std::shared_ptr<spdlog::logger> s_CoreLogger;
	inline static std::shared_ptr<spdlog::logger> s_FileLogger;
public:
	static void Init();

	static void ShutDownFile();
	static void ShutDownLogger();

	inline static std::shared_ptr<spdlog::logger>& GetCoreLogger() { return s_CoreLogger; }
	inline static std::shared_ptr<spdlog::logger>& GetFileLogger() { return s_FileLogger; }
};




// Core Logging Macros
#define LOG_CORE_INFO(...) Log::GetCoreLogger()->info(__VA_ARGS__)
#define LOG_CORE_WARN(...) Log::GetCoreLogger()->warn(__VA_ARGS__)
#define LOG_CORE_ERROR(...) Log::GetCoreLogger()->error(__VA_ARGS__)

// Editor Console Logging Macros
#define LOG_FILE(...)	Log::GetFileLogger()->trace(__VA_ARGS__)

