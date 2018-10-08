## Drone Steering Using Gestures

We use the PoseNet model as the basis to steer a drone using gestures recorded by a webcam. The tool is available [here](https://drone-steering.azurewebsites.net/webcam.html) currently supports the following gestures:

![Supported gestures](https://drone-steering.azurewebsites.net/images/summary.png)

These gestures are currently only static poses, meaning once you move your hands in the right position we detect it and send the signal. In a next step we will add dynamic gestures, e.g. performing a loop with your hands will mean the drone will also make a loop.

To support this, we will train a machine learning model on top of the PoseNet model. This will involve streaming input data and detecting the patterns to come up with the classification (e.g. loop, land or take picture).

The architecture of the model looks as follows:

![Model architecture](https://drone-steering.azurewebsites.net/images/architecture.png)

* The webcam films the user and feeds the video signal to the PoseNet model. The PoseNet model outputs x/y coordinates of various body parts.
* These x/y coordinates are the input into a GestureDetection model. This model translates the x/y coordinates (and patterns within these) into commands that can be understood by the drone.
* The steering module then transmits the commands through wifi to the drone.

We use pre-recorded videos with labels to generate training data. We then build and train the model in Python using Keras. Once trained, we embed this model in the JavaScript application (see [here](https://js.tensorflow.org/tutorials/import-saved-model.html) for how to do this).

### Project Evolution
In order to reduce delivery risk we divide our project in four stages and gradually add functionality (and thus complexity):
![Project evolution](https://drone-steering.azurewebsites.net/images/project_evolution.png)

* In a first stage we get the PoseNet model running in a custom website and calculate simple postures directly from the x/y coordinates. We display the inferred movements on the website.
* The second stage includes a Keras model embedded in the website. This model detects static postures (but no gestures yet - see next section for a discussion of postures vs. gestures). We train this model using webcam data enriched with manually added labels.
* In the third stage we add the more complex gesture detection model; we also add a module to actually steer the drone instead of just showing the direction on the website.
* The fourth and last stage is the highlight - we now use the video feed from the drone camera instead of the webcam. This brings additional complexity as we need to ensure a proper angle of the camera and the signal is less stable.

We are currently in the second stage, with the model already being embedded but not yet trained. In parallel we work on the third stage.

### Postures vs. Gestures
**Postures:** A posture is one single static position. It is thus possible to derive the movement from a single picture. Modelling this is straight-forward. We train the model with single position data. Once trained, the model predicts a movement from a given position.
![Posture model](https://drone-steering.azurewebsites.net/images/posture_model.png)
**Gestures:** A gesture is a sequence of interconnected positions, e.g. a circle movement done with the hand. This allows for more complex movements. Compared to the posture model the gesture model now needs a group of positions including their sequence as input.
![Gesture model](https://drone-steering.azurewebsites.net/images/gesture_model.png)

We are currently evaluating what the best way is to set up the gesture model. For simplicity we assume a maximum duration of two seconds for a gesture, with time steps of 0.05 seconds. Our initial idea was that the model input for a given point in time is an ordered set of 40 positions covering the last two seconds. The subsequent input would again be a set of 40 positions, with the oldest dropped and a new position added as the first. We thus 'outsource' the memory to the part where data is generated.

An alternative appear to be Hidden Markov Models (HMM) or Long Short-Term Memory Models (LSTM), or a hybrid of HMM and Convolutional Neural Networks (CNN) or Recurrent Neural Networks (RNN). From a superficial reading all these support time-dependent pattern (and thus gesture) recognition.


### Todos
* [Pascal] Research drone models (criteria: easy to steer / access video from computer)
* [All] Generate training data for posture model (stop, up, down, left, right)
* [Laleh] Train posture model such that it detects existing postures reliably
* [Pascal or Christian] Implement steering module
* [Christian and Pascal] Research which modelling approach is most suitable for the gesture model
* [tbd] Generate training data for gesture model
* [tbd] Implement and train gesture model
* [Christian] Integrate gesture model in JavaScript
