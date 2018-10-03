## Drone Steering Using Gestures

We use the PoseNet model as the basis to steer a drone using gestures recorded by a webcam. The tool currently supports the following gestures:

![Supported gestures](https://drone-steering.azurewebsites.net/images/summary.png)

These gestures are currently only static poses, meaning once you move your hands in the right position we detect it and send the signal. In a next step we will add dynamic gestures, e.g. performing a loop with your hands will mean the drone will also make a loop.