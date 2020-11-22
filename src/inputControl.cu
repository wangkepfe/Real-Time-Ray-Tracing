
#include "kernel.cuh"
#include <GLFW/glfw3.h>
#include <fstream>

struct InputControl
{
    float moveSpeed = 0.01f;
    float cursorMoveSpeed = 0.001f;

	double xpos = 0;
	double ypos = 0;

	bool moveW = 0;
	bool moveS = 0;
	bool moveA = 0;
	bool moveD = 0;
    bool moveC = 0;
    bool moveX = 0;

    float deltax = 0;
    float deltay = 0;
};

InputControl inputControl;

void RayTracer::keyboardUpdate(int key, int scancode, int action, int mods)
{
    if (mods == GLFW_MOD_CONTROL)
    {
        // camera save & load
        if (key == GLFW_KEY_C && action == GLFW_PRESS) { SaveCameraToFile("camera.bin"); }
        if (key == GLFW_KEY_V && action == GLFW_PRESS) { LoadCameraFromFile("camera.bin"); }
    }
    else
    {
        // movement
        if (key == GLFW_KEY_W) { if (action == GLFW_PRESS) inputControl.moveW = 1; else if (action == GLFW_RELEASE) inputControl.moveW = 0; }
        if (key == GLFW_KEY_S) { if (action == GLFW_PRESS) inputControl.moveS = 1; else if (action == GLFW_RELEASE) inputControl.moveS = 0; }
        if (key == GLFW_KEY_A) { if (action == GLFW_PRESS) inputControl.moveA = 1; else if (action == GLFW_RELEASE) inputControl.moveA = 0; }
        if (key == GLFW_KEY_D) { if (action == GLFW_PRESS) inputControl.moveD = 1; else if (action == GLFW_RELEASE) inputControl.moveD = 0; }
        if (key == GLFW_KEY_C) { if (action == GLFW_PRESS) inputControl.moveC = 1; else if (action == GLFW_RELEASE) inputControl.moveC = 0; }
        if (key == GLFW_KEY_X) { if (action == GLFW_PRESS) inputControl.moveX = 1; else if (action == GLFW_RELEASE) inputControl.moveX = 0; }

		if (key == GLFW_KEY_LEFT_SHIFT) { if (action == GLFW_PRESS) inputControl.moveSpeed = 0.001f; else if (action == GLFW_RELEASE) inputControl.moveSpeed = 0.01f;}

        // bvh debug
        if (key == GLFW_KEY_J && action == GLFW_PRESS) { cbo.bvhDebugLevel++; }
        if (key == GLFW_KEY_K && action == GLFW_PRESS) { cbo.bvhDebugLevel--; cbo.bvhDebugLevel = max(cbo.bvhDebugLevel, -1); }
        if (key == GLFW_KEY_L && action == GLFW_PRESS) { cbo.bvhDebugLevel = -1; }
    }
}

void RayTracer::cursorPosUpdate(double xpos, double ypos)
{
    static bool firstTime = true;

    if (firstTime)
    {
        firstTime = false;
        inputControl.xpos = xpos;
        inputControl.ypos = ypos;
        return;
    }

    inputControl.deltax = (float)(xpos - inputControl.xpos);
    inputControl.deltay = (float)(ypos - inputControl.ypos);

    inputControl.xpos = xpos;
    inputControl.ypos = ypos;

	// dir
	Camera& camera = cbo.camera;

	camera.yaw -= inputControl.deltax * inputControl.cursorMoveSpeed;
	camera.pitch -= inputControl.deltay * inputControl.cursorMoveSpeed;
    camera.pitch = clampf(camera.pitch, -PI_OVER_2 + 0.1f, PI_OVER_2 - 0.1f);

	camera.update();
}

void RayTracer::scrollUpdate(double xoffset, double yoffset)
{

}

void RayTracer::mouseButtenUpdate(int button, int action, int mods)
{

}

void RayTracer::InputControlUpdate()
{
	Camera& camera = cbo.camera;

    if (inputControl.moveW ||
		inputControl.moveS ||
		inputControl.moveA ||
		inputControl.moveD ||
		inputControl.moveC ||
		inputControl.moveX)
    {
        // pos
        Float3 movingDir = 0;
        Float3 strafeDir = cross(camera.dir, Float3(0, 1, 0)).normalize();

        if (inputControl.moveW) movingDir += camera.dir;
        if (inputControl.moveS) movingDir -= camera.dir;
        if (inputControl.moveA) movingDir -= strafeDir;
        if (inputControl.moveD) movingDir += strafeDir;
        if (inputControl.moveC) movingDir += Float3(0, 1, 0);
        if (inputControl.moveX) movingDir -= Float3(0, 1, 0);

        camera.pos += movingDir * deltaTime * inputControl.moveSpeed;

		camera.update();
    }
}

void RayTracer::SaveCameraToFile(const std::string &camFileName)
{
    Camera& camera = cbo.camera;
	using namespace std;
	ofstream myfile(camFileName, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
	if (myfile.is_open())
	{
		myfile.write(reinterpret_cast<char*>(&camera), sizeof(Camera));
		myfile.close();
        cout << "Successfully saved camera to file \"" << camFileName.c_str() << "\".\n";
	} else {
        cout << "Error: Failed to save camera to file \"" << camFileName.c_str() << "\".\n";
    }

}

void RayTracer::LoadCameraFromFile(const std::string &camFileName)
{
    Camera& camera = cbo.camera;
	using namespace std;
	if (camFileName.empty()){
		cout << "Error: Camera file name is not valid.\n";
        return;
	}
	ifstream infile(camFileName, std::ifstream::in | std::ifstream::binary);
	if (infile.good())
	{
		char *buffer = new char[sizeof(Camera)];
		infile.read(buffer, sizeof(Camera));
		camera = *reinterpret_cast<Camera *>(buffer);
		delete[] buffer;
		infile.close();
	} else {
		cout << "Error: Failed to read camera file \"" << camFileName.c_str() << "\".\n";
	}
}