#pragma once
#include <memory>
#include <spdlog/spdlog.h>

class Log
{
	inline static std::shared_ptr<spdlog::logger> s_CoreLogger;

  public:
	static void Init();

	static void ShutDown();

	inline static std::shared_ptr<spdlog::logger>& GetCoreLogger()
	{
		return s_CoreLogger;
	}
};

// Core Logging Macros
#define LOG_CORE_INFO(...)	Log::GetCoreLogger()->info(__VA_ARGS__)
#define LOG_CORE_WARN(...)	Log::GetCoreLogger()->warn(__VA_ARGS__)
#define LOG_CORE_ERROR(...) Log::GetCoreLogger()->error(__VA_ARGS__)
