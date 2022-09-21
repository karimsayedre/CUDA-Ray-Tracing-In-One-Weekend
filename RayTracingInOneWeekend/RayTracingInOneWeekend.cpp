#include "Log.h"

#include "Viewer.h"
#include "Renderer.h"


constexpr int WIDTH = 1280;
constexpr int HEIGHT = 720;

int main()
{
	Log::Init();

	Renderer renderer;

	renderer.Render(WIDTH, HEIGHT);
	

	//LOG_FILE(image);
	Log::ShutDownFile();
	LOG_CORE_INFO("Finished Successfully!");
	Viewer::OpenImage();

	Viewer::CloseImage();
	Log::ShutDownLogger();
}
