	const multiplier = 0.5;
  const imageScaleFactor = 0.5;
  const flipHorizontal = false;
  const outputStride = 8;
	const videoWidth = 800;
	const videoHeight = 555;
  let estimator = null;
  localStorage.position = ''
	var label = [];


	function getCoordinatesWebcam() {
	// gets the coordinates for each snapshot of the webcam

		var i = 0;
		var interval = 200

		startCamera();	
		video = document.getElementById('video');

	  tid = setInterval(function() {
	    if (i < 10) {
	    	++i;
	      document.getElementById('count').innerHTML = i;
	      getCoordSnapshot(video);
	    }
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

	  video.src = 'videos/' + video_name + '.mp4';
		video.play();

	  tid = setInterval(function() {
	    if (!video.ended && !video.paused) {
	    	++i;
	      document.getElementById('count').innerHTML = i;
	      getCoordSnapshot(video);
	      showLabel(i);
	    }
	  }, interval);

	}
  
  
  function getCoordSnapshot(imgData) {
	// gets to coordinates of imgData at the given time and stores it locally in a log file. This log file is available under
	// C:\Users\Christian\AppData\Local\Google\Chrome\User Data\Default\Local Storage\leveldb
	
    posenet
      .load(multiplier)
      .then(function(net) {
        if (!estimator) estimator = createEstimator(net);
        return estimator(imgData, imageScaleFactor, flipHorizontal, outputStride)
      })
      .then((resp) => {
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

        position = JSON.stringify(resp)
        console.log(position);
        localStorage.position = localStorage.position + position;
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
		  console.log(data);

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