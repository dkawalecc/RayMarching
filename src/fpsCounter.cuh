#ifndef FPS_COUNTER
#define FPS_COUNTER

#include <iomanip>
#include <iostream>

#define FRAME_SPAN 16

class FPSCounter
{
    double fps = 0.0;

    double acc = 0.0;
    int accFrames = 0;

    public:
        void registerFrame(double deltaTime)
        {
            acc += deltaTime;
            accFrames++;

            if (accFrames >= FRAME_SPAN)
            {
                fps = accFrames / acc;
                accFrames = 0;
                acc = 0;

                std::cout << std::fixed;
                std::cout << std::setprecision(2);
                std::cout << "FPS: " << fps << std::endl;
            }
        }
};

#endif //FPS_COUNTER