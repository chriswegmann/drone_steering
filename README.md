## Drone Steering Using Gestures

We use the PoseNet model as the basis to steer a drone using gestures recorded by a webcam. The tool is available [here](https://drone-steering.azurewebsites.net/predict_delta.html) and currently supports the following gestures:

![Supported gestures](https://drone-steering.azurewebsites.net/images/summary.png)

These gestures are currently only static postures, meaning once you move your hands in the right position we detect it and send the signal. We encode the movements as [stop, left, right, up, down] to [0, 1, 2, 3, 4]. In a next step we will add dynamic gestures, e.g. performing a loop with your hands will mean the drone will also make a loop; see separate section below for a more detailed discussion of this topic.

To support this, we will train a machine learning model on top of the PoseNet model. This will involve streaming input data and detecting the patterns to come up with the classification (e.g. loop, land or take picture) and a steering module to transmit signals to a drone.

![Model architecture](https://drone-steering.azurewebsites.net/images/architecture_local.png)

Let us look at the above modules a bit more in detail:
* The PoseNet model runs on a local node.js server. The user records video using the browser which connects to the server. The server then broadcasts the wireframe (i.e. the x/y coordinates of various body parts) via web socket.
* The web socket server runs in Python and receives the broadcasted wireframe data. Upon receiving, the server feeds the wireframe into a gesture detection model. This model translates the x/y coordinates (and patterns within these) into commands that can be understood by the drone.
* The Steering Module picks up the movements and translates them into the commands as required by the drone’s API.

We use pre-recorded videos with labels to generate training data. We then build and train the model in Python using Keras. Once trained, we embed this model in a JavaScript application.

### Installation / Start
You need to install _PoseNet for Installations_, available [here](https://github.com/oveddan/posenet-for-installations). 
You need to clone it to your local drive (and install _node.js_ and _yarn_ in case you don't have them yet).

Once installed, follow these steps:
* Go to the folder where you have have installed PoseNet and run ```yarn start```. This will start a node.js server (by default on port 3000) and open your browser pointed to this server.
* Run the ```websocket.py``` of this repo - this will set up a websocket server, by default on port 8081.
* Once the websocket server is running, switch back to the PoseNet website and connect to the websocket server (with the _cast_ icon) and then start the webcam and pose detection (with the other two icons).
* Voilà, you should see the model detecting your postures / gestures in the shell.

### Project Evolution
In order to reduce delivery risk we divide our project in four stages and gradually add functionality (and thus complexity):

![Project evolution](https://drone-steering.azurewebsites.net/images/project_evolution.png)

* In a first stage we get the PoseNet model running in a JavaScript application and calculate simple postures directly from the x/y coordinates. We display the inferred movements on the application.
* The second stage includes a Keras model embedded in the application. This model detects static postures (but no gestures yet - see next section for a discussion of postures vs. gestures). We train this model using webcam data enriched with manually added labels.
* In the third stage we add a more complex gesture detection model; we also add a module to actually steer the drone instead of just showing the direction in the application.
* The fourth and last stage is the highlight - we now use the video feed from the drone camera instead of the webcam. This brings additional complexity as we need to ensure a proper angle of the camera and the signal is less stable.

We are currently in the second stage, with the model already being embedded but not yet trained. In parallel we work on the third stage.

### Postures vs. Gestures
**Postures:** A posture is one single static position. It is thus possible to derive the movement from a single picture. Modelling this is straight-forward as we can train the model with lots of individual position data. Once trained, the model predicts a movement from a given position.
![Posture model](https://drone-steering.azurewebsites.net/images/posture_model.png)
**Gestures:** A gesture is a sequence of interconnected positions, e.g. a circle movement done with the hand. This allows for more complex movements. Compared to the posture model the gesture model now needs a group of positions including their sequence as input.

![Gesture model](https://drone-steering.azurewebsites.net/images/gesture_model.png)

We are currently evaluating the best approach for a gesture model. For simplicity we assume a maximum duration of two seconds for a gesture, with time steps of 0.05 seconds. Our initial idea is that the model input for a given point in time is an ordered set of 40 positions covering the last two seconds. The subsequent input would again be a set of 40 positions, with the oldest dropped and a new position added as the first. We thus outsource the 'remembering across time' from the model to the part of the application where data is generated. Schematically, this looks as follows (with the gray positions being the ones sent together as an input):

![Gesture model](https://drone-steering.azurewebsites.net/images/gesture_model_data.png)

Alternatives appear to be Hidden Markov Models (HMM), Long Short-Term Memory Models (LSTM) or a hybrid of HMM and Convolutional Neural Networks (CNN) or Recurrent Neural Networks (RNN). From a superficial reading all these support time-dependent pattern (and thus gesture) recognition.

### References
We plan to support the following six gestures:

<html>
 <table>
  <tr>
   <td><img src="https://drone-steering.azurewebsites.net/images/gesture_take_off.png"></td>
   <td><img src="https://drone-steering.azurewebsites.net/images/gesture_left.png"></td>
  </tr>
  <tr>
   <td style="background-color: #FFFFFF;"><img src="https://drone-steering.azurewebsites.net/images/gesture_land_grey.png"></td>
   <td style="background-color: #FFFFFF;"><img src="https://drone-steering.azurewebsites.net/images/gesture_right_grey.png"></td>
  </tr>
  <tr>
   <td><img src="https://drone-steering.azurewebsites.net/images/gesture_move.png"></td>
   <td><img src="https://drone-steering.azurewebsites.net/images/gesture_looping.png"></td>
  </tr>
 </table>
</html>  
  
Gestures are currently work in progress. Once a working version is available, we will make it accessible in the same way as the current posture model.

### References
* https://arxiv.org/pdf/1506.01911.pdf
* https://www.sciencedirect.com/science/article/pii/S1877750317312632
* https://arxiv.org/pdf/1707.03692.pdf
* https://arxiv.org/pdf/1802.09901.pdf
* https://arxiv.org/pdf/1712.10136.pdf
* https://github.com/udacity/CVND---Gesture-Recognition
* https://github.com/hthuwal/sign-language-gesture-recognition

### Todos
* [Christian] Implement steering module
* [Pascal and Laleh] Research which modelling approach is most suitable for the gesture model
* [All] Generate training data for gesture model
* [tbd] Implement and train the gesture model