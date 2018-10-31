const multiplier = 0.5;
const imageScaleFactor = 0.5;
const flipHorizontal = false;
const outputStride = 8;
const videoWidth = 800;
const videoHeight = 555;

let estimator = null;
localStorage.position = ''

var label = [];
var gestureSnapshots = [];
var all_samples = [];
var all_samples_json = '';
var counterVideos = [];

posenet
.load(multiplier)
.then(function(net) {
	if (!estimator) estimator = createEstimator(net);
})
.then((resp) => {
})


function getCoordinatesWebcam(modelType) {
// gets the coordinates for each snapshot of the webcam

	var interval = 200

	startCamera();	

	video = document.getElementById('video');

	tid = setInterval(function() {
		getCoordSnapshot(sourceId=video,
						 sourceType='webcam',
						 modelType=modelType,
						 steeringType='posture'
						 );
	}, interval);

}


function getCoordinatesAllVideos() {
// loops through all videos and gets the coordinates for each of them
	
	var i = 1;

	var videos = [];
	videos[0] = "001";
	videos[1] = "002";
	videos[2] = "003";
	videos[3] = "004";
	videos[4] = "005";

	var video = document.getElementById('video');

	getCoordinatesVideo(videos[0]);

	tid = setInterval(function() {
		if (video.ended && (i < videos.length)) {
			getCoordinatesVideo(videos[i]);
			++i;
		}
	}, 1000);

}


function getCoordinatesVideo(videoId) {
// runs the video and gets the coordinates for each snapshot at a selected time interval as long as the video is running

	counterVideos[videoId] = 0;

	var stepTime = document.getElementById("step_time");
	var interval = stepTime.options[stepTime.selectedIndex].value;
	var seqLength = document.getElementById("seq_length");
	var snapshotsPerGesture = seqLength.options[seqLength.selectedIndex].value / interval;
	var steeringType = (document.getElementById("posture").checked) ? 'posture' : 'gesture';

	var video = document.getElementById('video');
	var videoName = "video_" + String(videoId);

	loadLabels(videoId);

	video.src = 'videos/' + videoName + '.mp4';
	//video.src = 'https://prometheonkickoff.sharepoint.com/drone_steering/' + videoName + '.mp4';
	video.play();

	tid = setInterval(function() {
		if (!video.ended && !video.paused) {
			document.getElementById('count').innerHTML = counterVideos[videoId];
			getCoordSnapshot(sourceId=video, 
							 sourceType='video', 
							 modelType='none', 
							 steeringType=steeringType, 
							 snapshotsPerGesture=snapshotsPerGesture);
			showLabel(video.currentTime);
			++counterVideos[videoId];
		}
	}, interval);
	
}


function getCoordSnapshot(sourceId, sourceType, modelType='none', steeringType='posture', snapshotsPerGesture = 1) {
// gets the coordinates of an image at sourceId and either stores (for sourceType='video') or estimates (for sourceType='webcam') them

	estimator(sourceId, imageScaleFactor, flipHorizontal, outputStride).then((resp) => {
		
		if (sourceType=='video') {

			var sample = [];

			for (i=0; i < 17; i++) {
				if (document.getElementById("keypoint_" + i).checked) {
					j = sample.push(Math.round(resp.keypoints[i].position.x, 0) / 800);
					document.getElementById("keypoint_" + i + "_x").innerHTML = sample[j-1];
					j = sample.push(Math.round(resp.keypoints[i].position.y, 0) / 800);
					document.getElementById("keypoint_" + i + "_y").innerHTML = sample[j-1];
				}
			}

			if (steeringType=='posture') {
				sample.push(getLabel(sourceId.currentTime));
				all_samples_json = all_samples_json + JSON.stringify(sample);
			}

			if (steeringType=='gesture') {
				if (gestureSnapshots.length>snapshotsPerGesture) {
					gestureSnapshots.splice(0, 1);               // drop oldest snapshot
				}
				gestureSnapshots.splice(-1, 1);                  // drop previous label
				gestureSnapshots.push(sample);                   // add new snapshot
				var label = [getLabel(sourceId.currentTime)]
				gestureSnapshots.push(label);                    // add new label
				if (gestureSnapshots.length>snapshotsPerGesture) {
					all_samples_json = all_samples_json + JSON.stringify(gestureSnapshots);
				}
			}
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

				sample_predict[0][0] = leftShoulder_x / 800;;
				sample_predict[0][1] = leftShoulder_y / 800;;
				sample_predict[0][2] = rightShoulder_x / 800;;
				sample_predict[0][3] = rightShoulder_y / 800;;
				sample_predict[0][4] = leftElbow_x / 800;;
				sample_predict[0][5] = leftElbow_y / 800;;
				sample_predict[0][6] = rightElbow_x / 800;;
				sample_predict[0][7] = rightElbow_y / 800;;
				sample_predict[0][8] = leftWrist_x / 800;;
				sample_predict[0][9] = leftWrist_y / 800;;
				sample_predict[0][10] = rightWrist_x / 800;;
				sample_predict[0][11] = rightWrist_y / 800;;

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


function loadVideos() {

	if(document.getElementById("all_videos").checked) {
		getCoordinatesAllVideos();
	}
	else {
		var selectedVideo = document.getElementById("selected_video");
		var videoId = String(selectedVideo.options[selectedVideo.selectedIndex].value);
		getCoordinatesVideo(videoId);
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
	
	var part = []

	part[0] = 'nose';
	part[1] = 'leftEye';
	part[2] = 'rightEye';
	part[3] = 'leftEar';
	part[4] = 'rightEar';
	part[5] = 'leftShoulder';
	part[6] = 'rightShoulder';
	part[7] = 'leftElbow';
	part[8] = 'rightElbow';
	part[9] = 'leftWrist';
	part[10] = 'rightWrist';
	part[11] = 'leftHip';
	part[12] = 'rightHip';
	part[13] = 'leftKnee';
	part[14] = 'rightKnee';
	part[15] = 'leftAnkle';
	part[16] = 'rightAnkle';

	var all_samples_clipboard = all_samples_json.replace(/\]\[/g,"\n");
	all_samples_clipboard = all_samples_clipboard.replace(/\[/g,'');
	all_samples_clipboard = all_samples_clipboard.replace(/\]/g,'');

	var all_samples_clipboard_display = all_samples_json.replace(/\]\[/g,"<br>");
	all_samples_clipboard_display = all_samples_clipboard_display.replace(/\[/g,'');
	all_samples_clipboard_display = all_samples_clipboard_display.replace(/\]/g,'');

	var header = '';

	if (document.getElementById("gesture").checked) {
		var stepTime = document.getElementById("step_time");
		var interval = stepTime.options[stepTime.selectedIndex].value;
		var seqLength = document.getElementById("seq_length");
		var snapshotsPerGesture = seqLength.options[seqLength.selectedIndex].value / interval;

		for (k=0; k<snapshotsPerGesture; k++) {
			for (i=0; i<17; i++) {
				if (document.getElementById("keypoint_" + i).checked) {
					header = header + part[i] + "_x_" + k + "," + part[i] + "_y_" + k + ",";
				}
			}		
		}
	}
	else {
		for (i=0; i<17; i++) {
			if (document.getElementById("keypoint_" + i).checked) {
				header = header + part[i] + "_x," + part[i] + "_y,";
			}
		}	
	}

	all_samples_clipboard = header + "label\n" + all_samples_clipboard;
	all_samples_clipboard_display = header + "label<br>" + all_samples_clipboard_display;

	// copy data to clipboard
	var copyText = document.getElementById("all_samples_clipboard");
	copyText.style.display = "inline";
	copyText.value = all_samples_clipboard;
	copyText.select();
	document.execCommand("copy");
	copyText.style.display = "none";

	// show data on page
	document.getElementById("clipboard_message").innerHTML = 'Data copied to clipboard. Please copy it into a text file and save it as:';
	document.getElementById("clipboard_filename").innerHTML = getFileName();
	document.getElementById("all_samples_clipboard_display").innerHTML = all_samples_clipboard_display;
	
}


function getFileName() {

	var fileName = '';
	if(document.getElementById("all_videos").checked) {
		fileName = 'all_videos_';	
	}
	else {
		var selectedVideo = document.getElementById("selected_video");
		var videoId = String(selectedVideo.options[selectedVideo.selectedIndex].value);
		var fileName = "video_" + String(videoId) + '_';
	}
	(document.getElementById("posture").checked) ? fileName += 'posture_' : fileName += 'gesture_';
	var stepTime = document.getElementById("step_time");
	fileName += 'steptime' + String(stepTime.options[stepTime.selectedIndex].value) + '_checksum' + String(getPartsChecksum());

	if (document.getElementById("gesture").checked) {
		var seqLength = document.getElementById("seq_length");
		fileName += '_seqlength' + String(seqLength.options[seqLength.selectedIndex].value);
	}

	fileName += '.csv';

	return fileName;
}

function getPartsChecksum() {
	
	var checkSum = 0;

	for (i=0; i < 17; i++) {
		if (document.getElementById("keypoint_" + i).checked) {
			checkSum += Math.pow(2, i);
		}
	}

	document.getElementById("checksum").innerHTML = checkSum;

	return checkSum
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