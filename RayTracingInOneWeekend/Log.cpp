#include "pch.cuh"

#include "Log.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

void Log::Init()
{
	spdlog::sink_ptr appSinks {std::make_shared<spdlog::sinks::stdout_color_sink_mt>()};

	appSinks->set_pattern("%v%");

	s_CoreLogger = std::make_shared<spdlog::logger>("APP", appSinks);
	s_CoreLogger->set_level(spdlog::level::trace);
}

void Log::ShutDown()
{
	s_CoreLogger.reset();

	spdlog::drop_all();
}