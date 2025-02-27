#include "pch.cuh"

#include "Log.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

 void Log::Init()
{
	// Create "logs" directory if doesn't exist
	spdlog::sink_ptr fileSinks{ std::make_shared<spdlog::sinks::basic_file_sink_mt>("Image.ppm", true) };
	spdlog::sink_ptr appSinks{ std::make_shared<spdlog::sinks::stdout_color_sink_mt>() };

	fileSinks->set_pattern("%v%");
	appSinks->set_pattern("%v%");

	s_FileLogger = std::make_shared<spdlog::logger>("FILE", fileSinks);
	s_FileLogger->set_level(spdlog::level::trace);

	s_CoreLogger = std::make_shared<spdlog::logger>("APP", appSinks);
	s_CoreLogger->set_level(spdlog::level::trace);
}


void Log::ShutDownFile()
{
	s_FileLogger.reset();
}

void Log::ShutDownLogger()
{
	s_CoreLogger.reset();

	spdlog::drop_all();

}