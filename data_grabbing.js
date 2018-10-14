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
	    	getCoordSnapshot(video, 'webcam', modelType);
	  	}, interval);

	}


	function getCoordinatesAllVideos() {
	// TODO: this should be made work with asynchronous functions
		var videos = [];
		videos[0] = "001";
		videos[1] = "002";

		for (var i in videos) {
			getCoordinatesVideo(videos[i]);
		}		

	}


	function getCoordinatesVideo() {
	// runs the video and gets the coordinates for each snapshot at a selected time interval as long as the video is running

		var i = 0;

		var stepTime = document.getElementById("step_time");
		var interval = stepTime.options[stepTime.selectedIndex].value;
	  	var video = document.getElementById('video');
		var selectedVideo = document.getElementById("selected_video");
		var videoId = String(selectedVideo.options[selectedVideo.selectedIndex].value);
		var videoName = "video_" + String(videoId);

		loadLabels(videoId);

	  	video.src = 'videos/' + videoName + '.mp4';
		video.play();

	  	tid = setInterval(function() {
	    	if (!video.ended && !video.paused) {
	      		document.getElementById('count').innerHTML = i;
	      		getCoordSnapshot(video, 'video');
				showLabel(video.currentTime);
	    		++i;
	    	}
	  	}, interval);
	  
	}
  
  
  function getCoordSnapshot(imgData, sourceType, modelType='none') {
	// gets to coordinates of imgData at the given time and store or display the information
	
    posenet
      .load(multiplier)
      .then(function(net) {
        if (!estimator) estimator = createEstimator(net);
        return estimator(imgData, imageScaleFactor, flipHorizontal, outputStride)
      })
      .then((resp) => {
      	
      	if (sourceType=='video') {
			
			j = 0;

			for (i=0; i < 17; i++) {
				if (document.getElementById("keypoint_" + i).checked) {
					sample[j] = Math.round(resp.keypoints[i].position.x, 0) / 800;
					document.getElementById("keypoint_" + i + "_x").innerHTML = sample[j];
					j += 1;
					sample[j] = Math.round(resp.keypoints[i].position.y, 0) / 800;
					document.getElementById("keypoint_" + i + "_y").innerHTML = sample[j];
					j += 1;
				}
			}

			sample[j] = getLabel(imgData.currentTime);
				
			sample_json = JSON.stringify(sample);
			all_samples_json = all_samples_json + sample_json;

			//store the coordinates locally in a log file (available under C:\Users\Christian\AppData\Local\Google\Chrome\User Data\Default\Local Storage\leveldb)
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

				var left_ind = '';
				var right_ind = '';
				var stop_ind = '';
				var up_ind = '';
				var down_ind = '';


				if (leftArm_x > 60) {
					left_ind = 'detected';
					direction = 'left.png'
				}

				if (rightArm_x > 60) {
					right_ind = 'detected';
					direction = 'right.png';
				}

				if ((leftArm_x > 60) & (rightArm_x > 60)) {
					stop_ind = 'detected';
					direction = 'stop.png';
				}


				if ((leftArm_y > 100) & (rightArm_y > 100))  {
					up_ind = 'detected';
					direction = 'up.png';
				}

				if ((leftArm_y < -100) & (rightArm_y < -100))  {
					down_ind = 'detected';
					direction = 'down.png';
				}

				document.getElementById("leftArm_x").innerHTML = leftArm_x;
				document.getElementById("rightArm_x").innerHTML = rightArm_x;
				document.getElementById("leftArm_y").innerHTML = leftArm_y;
				document.getElementById("rightArm_y").innerHTML = rightArm_y;

				document.getElementById("label_display").src = "images/" + direction;           

				document.getElementById("stop").innerHTML = stop_ind;
				document.getElementById("left").innerHTML = left_ind;
				document.getElementById("right").innerHTML = right_ind;
				document.getElementById("up").innerHTML = up_ind;
				document.getElementById("down").innerHTML = down_ind;	

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
	

	function getLabel(videoTime) {
		for (var i in label) {
			if ((videoTime > label[i][0] & videoTime < label[i][1])) {
				//document.getElementById("video_position").innerHTML = String(videoTime) + ': ' + String(label[i][2]) + ' | ' + String(label[i][0]) + ' | ' + String(label[i][1]);
				return label[i][2];
			}
		}

	}

	function loadLabels(videoId)	{
	// loads the labels of a video into the 'label' array

		d3.csv("videos/labels_" + videoId + ".csv").then(function(data) {

			var i;
			for (i = 0; i < data.length; i++) { 
				label[i] = [];
				label[i][0] = parseFloat(data[i]["from"]);
				label[i][1] = parseFloat(data[i]["to"]);
				label[i][2] = parseInt(data[i]["label"]);
			}

		});


		// old variant
		//d3.csv("videos/" + video + ".csv").then(function(data) {

		//	var i;
		//	for (i = 0; i < data.length; i++) { 
		//	    label[i] = parseInt(data[i]["label"]);
		//	}

		//});
		
	}


	function showLabel(videoTime) {
	// displays the image (left, right etc.) corresponding to a label
			
		var img_src = 'stop.png';
		var labelId = getLabel(videoTime)
		
		switch(labelId) {
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
	// copies the samples (features and label) to the clipboard
		
		var all_samples_clipboard = all_samples_json.replace(/\]\[/g,"\n")
		all_samples_clipboard = all_samples_clipboard.replace('[','')
		all_samples_clipboard = all_samples_clipboard.replace(']','')

		var copyText = document.getElementById("all_samples_clipboard");
		copyText.style.display = "inline";
		copyText.value = all_samples_clipboard;
		copyText.select();
		document.execCommand("copy");
		copyText.style.display = "none";
		document.getElementById("clipboard_message").innerHTML = 'Data copied to clipboard.';

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


	function enableDisableVideoSelection() {
		select = document.getElementById("selected_video");
		if (select.disabled) {
			select.disabled = 0
		}
		else {
			select.disabled = 1
		}
	}


	function showHideSeqLength() {
		//alert(document.getElementById("posture").checked);
		if (document.getElementById("posture").checked) {
			document.getElementById("seq_length").style.display = "none";
			document.getElementById("seq_length_label").innerHTML = "";
		}
		else {
			document.getElementById("seq_length").style.display = "inline";
			document.getElementById("seq_length_label").innerHTML = "Sequence Length:";
		}
	}