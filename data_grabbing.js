	const multiplier = 0.5;
  const imageScaleFactor = 0.5;
  const flipHorizontal = false;
  const outputStride = 8;
	const videoWidth = 800;
	const videoHeight = 555;

	let estimator = null;
  localStorage.position = ''

	var label = [];
	var sample = [];
	var all_samples = [];
	var all_samples_json = '';


	function getCoordinatesWebcam(modelType) {
	// gets the coordinates for each snapshot of the webcam

		var i = 0;
		var interval = 200

		startCamera();	
		video = document.getElementById('video');

	  tid = setInterval(function() {
	      getCoordSnapshot(video, 0, 'webcam', modelType);
	  }, interval);

	}


	function getCoordinatesVideo() {
	// runs the video and gets the coordinates for each snapshot of a time interval as long as the video is running

		var i = 0;
		var interval = 300
	  var video = document.getElementById('video');

		var selected_video = document.getElementById("selected_video");
		var video_name = selected_video.options[selected_video.selectedIndex].value;

	  loadLabels(video_name);

		//if (purpose=='predict') {
		//	const model = tf.loadModel('model_tfjs/model.json');
		//}

	  video.src = 'videos/' + video_name + '.mp4';
		video.play();

	  tid = setInterval(function() {
	    if (!video.ended && !video.paused) {
	      document.getElementById('count').innerHTML = i;
	      getCoordSnapshot(video, i, 'video');
	      if (purpose=='preparation') {
					showLabel(i);
				}
	    	++i;
	    }
	  }, interval);
	  
	  //JSON.stringify(all_samples);
	  //console.log(all_samples);

	}
  
  
  function getCoordSnapshot(imgData, labelIndex, sourceType, modelType='none') {
	// gets to coordinates of imgData at the given time and store or display the information
	
    posenet
      .load(multiplier)
      .then(function(net) {
        if (!estimator) estimator = createEstimator(net);
        return estimator(imgData, imageScaleFactor, flipHorizontal, outputStride)
      })
      .then((resp) => {
      	
      	if (sourceType=='video') {

	      	document.getElementById("nose_x").innerHTML = resp.keypoints[0].position.x;
	      	document.getElementById("nose_y").innerHTML = resp.keypoints[0].position.y;
	      	document.getElementById("leftEye_x").innerHTML = resp.keypoints[1].position.x;
	      	document.getElementById("leftEye_y").innerHTML = resp.keypoints[1].position.y;
	      	document.getElementById("rightEye_x").innerHTML = resp.keypoints[2].position.x;
	      	document.getElementById("rightEye_y").innerHTML = resp.keypoints[2].position.y;
	      	document.getElementById("leftEar_x").innerHTML = resp.keypoints[3].position.x;
	      	document.getElementById("leftEar_y").innerHTML = resp.keypoints[3].position.y;
	      	document.getElementById("rightEar_x").innerHTML = resp.keypoints[4].position.x;
	      	document.getElementById("rightEar_y").innerHTML = resp.keypoints[4].position.y;
	      	document.getElementById("leftShoulder_x").innerHTML = resp.keypoints[5].position.x;
	      	document.getElementById("leftShoulder_y").innerHTML = resp.keypoints[5].position.y;
	      	document.getElementById("rightShoulder_x").innerHTML = resp.keypoints[6].position.x;
	      	document.getElementById("rightShoulder_y").innerHTML = resp.keypoints[6].position.y;
	      	document.getElementById("leftElbow_x").innerHTML = resp.keypoints[7].position.x;
	      	document.getElementById("leftElbow_y").innerHTML = resp.keypoints[7].position.y;
	      	document.getElementById("rightElbow_x").innerHTML = resp.keypoints[8].position.x;
	      	document.getElementById("rightElbow_y").innerHTML = resp.keypoints[8].position.y;
	      	document.getElementById("leftWrist_x").innerHTML = resp.keypoints[9].position.x;
	      	document.getElementById("leftWrist_y").innerHTML = resp.keypoints[9].position.y;
	      	document.getElementById("rightWrist_x").innerHTML = resp.keypoints[10].position.x;
	      	document.getElementById("rightWrist_y").innerHTML = resp.keypoints[10].position.y;
	      	document.getElementById("leftHip_x").innerHTML = resp.keypoints[11].position.x;
	      	document.getElementById("leftHip_y").innerHTML = resp.keypoints[11].position.y;
	      	document.getElementById("rightHip_x").innerHTML = resp.keypoints[12].position.x;
	      	document.getElementById("rightHip_y").innerHTML = resp.keypoints[12].position.y;
	      	document.getElementById("leftKnee_x").innerHTML = resp.keypoints[13].position.x;
	      	document.getElementById("leftKnee_y").innerHTML = resp.keypoints[13].position.y;
	      	document.getElementById("rightKnee_x").innerHTML = resp.keypoints[14].position.x;
	      	document.getElementById("rightKnee_y").innerHTML = resp.keypoints[14].position.y;
	      	document.getElementById("leftAnkle_x").innerHTML = resp.keypoints[15].position.x;
	      	document.getElementById("leftAnkle_y").innerHTML = resp.keypoints[15].position.y;
	      	document.getElementById("rightAnkle_x").innerHTML = resp.keypoints[16].position.x;
	      	document.getElementById("rightAnkle_y").innerHTML = resp.keypoints[16].position.y;

					sample[0] = Math.round(resp.keypoints[5].position.x, 0) / 800;
					sample[1] = Math.round(resp.keypoints[5].position.y, 0) / 800;
					sample[2] = Math.round(resp.keypoints[6].position.x, 0) / 800;
					sample[3] = Math.round(resp.keypoints[6].position.y, 0) / 800;
					sample[4] = Math.round(resp.keypoints[7].position.x, 0) / 800;
					sample[5] = Math.round(resp.keypoints[7].position.y, 0) / 800;
					sample[6] = Math.round(resp.keypoints[8].position.x, 0) / 800;
					sample[7] = Math.round(resp.keypoints[8].position.y, 0) / 800;
					sample[8] = Math.round(resp.keypoints[9].position.x, 0) / 800;
					sample[9] = Math.round(resp.keypoints[9].position.y, 0) / 800;
					sample[10] = Math.round(resp.keypoints[10].position.x, 0) / 800;
					sample[11] = Math.round(resp.keypoints[10].position.y, 0) / 800;

					sample[12] = label[labelIndex];
						
					sample_json = JSON.stringify(sample_with_label);
					all_samples_json = all_samples_json + sample_json;
	
					// store the coordinates locally in a log file (available under C:\Users\Christian\AppData\Local\Google\Chrome\User Data\Default\Local Storage\leveldb)
					//position = JSON.stringify(resp);
					//console.log(position);
					//localStorage.position = localStorage.position + position;

	      }

      	if (sourceType=='webcam') {
						
	      		var direction = 'stop.png'
      		
           	var leftShoulder_x = Math.round(resp.keypoints[5].position.x);
            var leftShoulder_y = Math.round(resp.keypoints[5].position.y);
            var rightShoulder_x = Math.round(resp.keypoints[6].position.x);
            var rightShoulder_y = Math.round(resp.keypoints[6].position.y);
            var leftElbow_x = Math.round(resp.keypoints[7].position.x);
            var leftElbow_y = Math.round(resp.keypoints[7].position.y);
            var rightElbow_x = Math.round(resp.keypoints[8].position.x);
            var rightElbow_y = Math.round(resp.keypoints[8].position.y);
            var leftWrist_x = Math.round(resp.keypoints[9].position.x);
            var leftWrist_y = Math.round(resp.keypoints[9].position.y);
            var rightWrist_x = Math.round(resp.keypoints[10].position.x);
            var rightWrist_y = Math.round(resp.keypoints[10].position.y);

						if (modelType=='deltas') {

							var leftArm_x = leftWrist_x - leftShoulder_x;
							var rightArm_x = rightShoulder_x - rightWrist_x;
							var leftArm_y = leftShoulder_y - leftWrist_y;
							var rightArm_y = rightShoulder_y - rightWrist_y;
	
							var left_ind = 'leftArm_x: ' + leftArm_x + '<br>leftWrist_x: ' + leftWrist_x + '<br>leftShoulder_x: ' + leftShoulder_x;
							var right_ind = 'rightArm_x: ' + rightArm_x + '<br>rightWrist_x: ' + rightWrist_x + '<br>rightShoulder_x: ' + rightShoulder_x;
							var stop_ind = left_ind + '<br>' + right_ind;
							var up_ind = 'leftArm_y: ' + leftArm_y + '<br>leftWrist_y: ' + leftWrist_y + '<br>leftShoulder_y: ' + leftShoulder_y + '<br>rightArm_y: ' + rightArm_y + '<br>rightWrist_y: ' + rightWrist_y + '<br>rightShoulder_y: ' + rightShoulder_y;
							var down_ind = up_ind;
	
	
							if (leftArm_x > 60) {
											left_ind = 'detected<br>' + left_ind;
											direction = 'left.png'
							}
	
							if (rightArm_x > 60) {
											right_ind = 'detected<br>' + right_ind;
											direction = 'right.png';
							}
	
							if ((leftArm_x > 60) & (rightArm_x > 60)) {
											stop_ind = 'detected<br>' + stop_ind;
											direction = 'stop.png';
							}
	
	
							if ((leftArm_y > 100) & (rightArm_y > 100))  {
											up_ind = 'detected<br>' + up_ind;
											direction = 'up.png';
							}
	
							if ((leftArm_y < -100) & (rightArm_y < -100))  {
											down_ind = 'detected<br>' + down_ind;
											direction = 'down.png';
							}
	
							document.getElementById("leftShoulder_x").innerHTML = leftShoulder_x;
							document.getElementById("leftShoulder_y").innerHTML = leftShoulder_y;
							document.getElementById("rightShoulder_x").innerHTML = rightShoulder_x;
							document.getElementById("rightShoulder_y").innerHTML = rightShoulder_y;
							document.getElementById("leftElbow_x").innerHTML = leftElbow_x;
							document.getElementById("leftElbow_y").innerHTML = leftElbow_y;
							document.getElementById("rightElbow_x").innerHTML = rightElbow_x;
							document.getElementById("rightElbow_y").innerHTML = rightElbow_y;
							document.getElementById("leftWrist_x").innerHTML = leftWrist_x;
							document.getElementById("leftWrist_y").innerHTML = leftWrist_y;
							document.getElementById("rightWrist_x").innerHTML = rightWrist_x;
							document.getElementById("rightWrist_y").innerHTML = rightWrist_y;

							document.getElementById("leftArm_x").innerHTML = leftArm_x;
							document.getElementById("rightArm_x").innerHTML = rightArm_x;
							document.getElementById("leftArm_y").innerHTML = leftArm_y;
							document.getElementById("rightArm_y").innerHTML = rightArm_y;


							document.getElementById("label_display").src = "images/" + direction;           
	
							document.getElementById("stop_calc").innerHTML = stop_ind;
							document.getElementById("left_calc").innerHTML = left_ind;
							document.getElementById("right_calc").innerHTML = right_ind;
							document.getElementById("up_calc").innerHTML = up_ind;
							document.getElementById("down_calc").innerHTML = down_ind;	

						}

	
						if (modelType=='postures') {

							sample_predict = new Array(1)
							sample_predict[0] = new Array(12)

							sample_predict[0][0] = leftShoulder_x;
							sample_predict[0][1] = leftShoulder_y;
							sample_predict[0][2] = rightShoulder_x;
							sample_predict[0][3] = rightShoulder_y;
							sample_predict[0][4] = leftElbow_x;
							sample_predict[0][5] = leftElbow_y;
							sample_predict[0][6] = rightElbow_x;
							sample_predict[0][7] = rightElbow_y;
							sample_predict[0][8] = leftWrist_x;
							sample_predict[0][9] = leftWrist_y;
							sample_predict[0][10] = rightWrist_x;
							sample_predict[0][11] = rightWrist_y;

							predictFromModel(sample_predict);
							
						}
      		
      	}

      })
  }


  function createEstimator(net) {
  // creates an estimator from PoseNet
  
    return function (imageElement, scaleFactor, flipHorizontal, outputStride) {
      return net.estimateSinglePose(imageElement, scaleFactor, flipHorizontal, outputStride);
    }
  }  


	function sleep(milliseconds) {
	// pauses the execution of the script by the passed-on milliseconds

	  var start = new Date().getTime();
	  for (var i = 0; i < 1e7; i++) {
	    if ((new Date().getTime() - start) > milliseconds){
	      break;
	    }
	  }
	}


	async function setupCamera() {
	// loads the camera to be used in the demo

	  const video = document.getElementById('video');
	  video.width = videoWidth;
	  video.height = videoHeight;

	  const stream = await navigator.mediaDevices.getUserMedia({
	    'audio': false,
	    'video': {
	      facingMode: 'user',
	      width: videoWidth,
	      height: videoHeight
	    }
	  });
	  video.srcObject = stream;

		return video;

	}

	async function startCamera() {
	// starts the webcam of the user
	
	  const video = await setupCamera();
	  video.play();

		return video;
		
	}
	

	function loadLabels(video)	{
	// loads the labels of a video into the 'label' array (with 100ms intervals)

			d3.csv("videos/" + video + ".csv").then(function(data) {

			var i;
			for (i = 0; i < data.length; i++) { 
			    label[i] = parseInt(data[i]["label"]);
			}
		  //console.log(data);

		});
		
	}


	function showLabel(interval) {
	// displays the image (left, right etc.) corresponding to a label
			
		var img_src = 'stop.png';
		
		switch(label[interval]) {
		    case 0: // stop
		        img_src = 'stop.png';
		        break;
		    case 1: // left
		        img_src = 'left.png';
		        break;
		    case 2: // right
		        img_src = 'right.png';
		        break;
		    case 3: // up
		        img_src = 'up.png';
		        break;
		    case 4: // down
		        img_src = 'down.png';
		        break;
		    default:
		        img_src = 'stop.png';
		}		
		
		document.getElementById("label_display").src = "images/" + img_src;		
		
	}
	
	
	function getFeatures() {
	// dumps all samples (features and label) at the bottom of the page
		
		var all_samples_csv = all_samples_json.replace(/\]\[/g,'<br>')
		all_samples_csv = all_samples_csv.replace('[','')
		all_samples_csv = all_samples_csv.replace(']','')
		
		document.getElementById("all_samples").innerHTML = all_samples_csv;

	}


	async function predictFromModel(sample)	{
		// predicts the direction from a sample using the model trained in keras
	
				sample_tensor = tf.tensor(sample);
				console.log(sample_tensor);
	
				const model = await tf.loadModel('model_tfjs/model.json');
				predicted = model.predict(sample_tensor)
				const values = predicted.dataSync();
				const arr = Array.from(values);
	
				document.getElementById("stop").innerHTML = arr[0];
				document.getElementById("left").innerHTML = arr[1];
				document.getElementById("right").innerHTML = arr[2];
				document.getElementById("up").innerHTML = arr[3];
				document.getElementById("down").innerHTML = arr[4];
	
		}
	