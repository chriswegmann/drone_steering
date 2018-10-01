	const multiplier = 0.5;
  const imageScaleFactor = 0.5;
  const flipHorizontal = false;
  const outputStride = 8;
	const videoWidth = 800;
	const videoHeight = 555;
  let estimator = null;
  localStorage.position = ''


	function get_coordinates_webcam() {
	// gets the coordinates for each snapshot of the webcam
	// todo: replace interval-based condition with proper stop button

		var i = 0;
		var interval = 300

		startCamera();	
		video = document.getElementById('video');

	  tid = setInterval(function() {
	    if (i < 10) {
	    	++i;
	      document.getElementById('count').innerHTML = i
	      getCoordSnapshot(video)
	    }
	  }, interval);

	}


	function get_coordinates_video() {
	// runs the video and gets the coordinates for each snapshot of a time interval as long as the video is running

		var i = 0;
		var interval = 300
		//var video = startStopVideo()

	  var video = document.getElementById('video');
		video.play();

	  tid = setInterval(function() {
	    if (!video.ended && !video.paused) {
	    	++i;
	      document.getElementById('count').innerHTML = i
	      getCoordSnapshot(video)
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

	//function startStopVideo()  {
	// starts and stops the video embedded in the 'video' element	
		
	//  var button = document.getElementById('video_button').firstChild;
	//  var video = document.getElementById('video');
	//  if(button.data == 'Get coordinates') {
	//  	button.data = 'Pause'
	//		video.play();
	//  } else {
	//  	button.data = 'Get coordinates'
	//		video.pause();
	//  }
	//  return video
	//}