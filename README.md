## Drone Steering Using Gestures

We use the PoseNet model as the basis to steer a drone using gestures recorded by a webcam. The tool is available [here](https://drone-steering.azurewebsites.net/webcam.html) currently supports the following gestures:

![Supported gestures](https://drone-steering.azurewebsites.net/images/summary.png)

These gestures are currently only static poses, meaning once you move your hands in the right position we detect it and send the signal. In a next step we will add dynamic gestures, e.g. performing a loop with your hands will mean the drone will also make a loop.

To support this, we will train a machine learning model on top of the PoseNet model. This will involve streaming input data and detecting the patterns to come up with the classification (e.g. loop, land or take picture).

The architecture of the model looks as follows:

![Model architecture](https://drone-steering.azurewebsites.net/images/architecture.png)

In essence, we structure the tool technically as follows:
* The webcam films the user and feeds the video signal to the PoseNet model. The PoseNet model outputs x/y coordinates of various body parts.
* These x/y coordinates are the input into a GestureDetection model. This model translates the x/y coordinates (and patterns within these) into commands that can be understood by the drone.
* The steering module then transmits the commands through wifi to the drone.

We use pre-recorded videos with labels to generate training data. We then build and train the model in Python using Keras. Once trained, we embed this model in the JavaScript application (see [here](https://js.tensorflow.org/tutorials/import-saved-model.html) for how to do this).


### Todos
* Make Keras model work in tensorflow.js
* Research drone models (criteria: easy to steer / access video from computer)
* Generate training data for model for existing postures (stop, up, down, left, right)
* Train model such that it detects existing postures reliably
* Research about streaming machine learning and what the requirements are
  * How does a model detect sequences in a stream?
  * Does Keras support streaming ML?
  * How can we feed a stream as input to the model?
* Generate training data for streaming model
* Train streaming model