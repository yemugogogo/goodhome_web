/**
 * @license
 * Copyright 2018 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
//匯入(import)
import * as posenet from '@tensorflow-models/posenet';
import dat from 'dat.gui';
import Stats from 'stats.js';

import { drawBoundingBox, drawKeypoints, drawSkeleton } from './demo_util';

//前置作業
const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();
let previous_direction = null;//加了一個區域變數 後面用到

let windowNextUpdateIndex = 0;
let windowForMdeianSize = 10;
var windowForMedian = new Array(windowForMdeianSize);

function getStableKeyPoints(index, direction) {
  // There are unstability in PoseNet detection. We use a window to store
  // a size of N previous detection. Sort in that duration and get the median
  // to ensure we rule out the extreme values that might just be accidental.
  var medianToReturn = null;
  if (windowNextUpdateIndex >= windowForMdeianSize) {
    // Let's pick the median we are concerned about.

    // Getting the median of keypoints[index]
    var arrayToSort = new Array(windowForMdeianSize);
    for (let i=0; i<windowForMdeianSize; ++i)
      arrayToSort[i] = windowForMedian[i][index]["position"][direction];

    arrayToSort = arrayToSort.sort();
    medianToReturn = arrayToSort[windowForMdeianSize/2];
    // TODO: Remove the debug message
    document.getElementById('myDiv01').value = medianToReturn;
  }
  return medianToReturn;
}

function calculateTriangleArea(a, b, c, d) {
  // Reference: http://highscope.ch.ntu.edu.tw/wordpress/wp-content/uploads/2015/11/66359_p1.png
  // Reference: http://highscope.ch.ntu.edu.tw/wordpress/?p=66359  (ad-bc)
  // Reference: https://blog.xuite.net/wang620628/twblog/126094614 
  return Math.abs(a*d - b*c);
}


//判斷是否為Android - function 1
function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

//判斷是否為iOS - function 2
function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

//判斷是否為移動裝置(using function 1, 2)
function isMobile() {
  return isAndroid() || isiOS();
}

//Loads a the camera to be used in the demo
//加載要在演示中使用的相機 - function 4(using function 3)
async function setupCamera() {
  //如果加載不到相機，則產生錯誤訊息
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  //設定video的大小
  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  //如果是行動裝置(using function 3)
  const mobile = isMobile();
  //開啟流，進行行動裝置的相機設定
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  //將結果(物件)定義為上方的流
  video.srcObject = stream;

  //回傳video的結果
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

//加載影片 - function 5(using function 4)
async function loadVideo() {
  //使用相機設定(using function 4)
  const video = await setupCamera();
  video.play();

  return video;
}

//設定gui的參數(輸入什麼和輸出什麼) - function 6(using function 3)
const guiState = {
  //演算法為多姿勢
  algorithm: 'multi-pose',
  input: {
    //using function 3
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',

    //輸出步幅——必須為 32、16 或 8。
    //預設值為 16。
    //從內部來看，該引數影響神經網路中層的高度和寬度。
    //在較高層次上，它會影響姿態估計的準確率和速度。
    //輸出步幅的值越低，準確率越高，但速度越慢；
    //輸出步幅值越高，速度越快，但準確率越低。 
    //原文網址：https://itw01.com/GXQFHE6.html
    outputStride: 16,

    //影象比例因子——介於 0.2~1 的數字，預設值為 0.50。
    //用於在向網路輸送影象之前，對影象進行縮放。
    //將此數字設定得較低，以縮小影象，提高輸送至網路的速度
    //不過這是以準確率為代價的。
    //原文網址：https://itw01.com/GXQFHE6.html
    imageScaleFactor: 0.5,
  },
  //單一姿勢檢測
  singlePoseDetection: {
    //最小的信心值 - pose
    minPoseConfidence: 0.1,
    //最小的信心值 - part
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    //最小的信心值 - pose
    minPoseConfidence: 0.15,
    //最小的信心值 - part
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    //影像
    showVideo: true,
    //骨架
    showSkeleton: true,
    //點
    showPoints: true,
    //邊界框
    showBoundingBox: false,
  },
  net: null,
};

//Sets up dat.gui controller on the top-right of the window
//在窗口的右上角設置dat.gui控制器 - function 7(using function 6)
function setupGui(cameras, net) {
  guiState.net = net;

  if (cameras.length > 0) {
    guiState.camera = cameras[0].deviceId;
  }

  const gui = new dat.GUI({ width: 300 });

  // The single-pose algorithm is faster and simpler but requires only one
  // person to be in the frame or results will be innaccurate. Multi-pose works
  // for more than 1 person
  //單姿勢算法更快更簡單，但只需要一個
  //在框架中的人或結果將是不准確的。多姿勢的作品
  //超過1人
  const algorithmController =
    gui.add(guiState, 'algorithm', ['single-pose', 'multi-pose']);

  // The input parameters have the most effect on accuracy and speed of the
  //輸入參數對精度和速度的影響最大
  // network 網路
  let input = gui.addFolder('Input');

  // Architecture: there are a few PoseNet models varying in size and
  // accuracy. 1.01 is the largest, but will be the slowest. 0.50 is the
  // fastest, but least accurate.
  //架構：有一些PoseNet模型的大小不一
  //準確性。 1.01是最大的，但將是最慢的。 0.50是
  //最快，但最不准確。
  const architectureController = input.add(
    guiState.input, 'mobileNetArchitecture',
    ['1.01', '1.00', '0.75', '0.50']);

  // Output stride:  Internally, this parameter affects the height and width of
  // the layers in the neural network. The lower the value of the output stride
  // the higher the accuracy but slower the speed, the higher the value the
  // faster the speed but lower the accuracy.
  //輸出步幅：在內部，此參數會影響高度和寬度
  //神經網絡中的層。輸出步幅的值越低
  //準確度越高但速度越慢，值越高
  //速度越快但精度越低。
  input.add(guiState.input, 'outputStride', [8, 16, 32]);

  // Image scale factor: What to scale the image by before feeding it through
  // the network.
  //圖像比例因子：在通過圖像之前縮放圖像的內容
  // 網絡。
  input.add(guiState.input, 'imageScaleFactor').min(0.2).max(1.0);
  input.open();

  // Pose confidence: the overall confidence in the estimation of a person's
  // pose (i.e. a person detected in a frame)
  // Min part confidence: the confidence that a particular estimated keypoint
  // position is accurate (i.e. the elbow's position)
  //姿勢信心：對一個人估計的總體信心
  //姿勢（即在框架中檢測到的人）
  //最小部分置信度：特定估計關鍵點的置信度
  //位置準確（即肘部位置）
  //單姿勢
  let single = gui.addFolder('Single Pose Detection');
  single.add(guiState.singlePoseDetection, 'minPoseConfidence', 0.0, 1.0);
  single.add(guiState.singlePoseDetection, 'minPartConfidence', 0.0, 1.0);

  //多姿勢
  let multi = gui.addFolder('Multi Pose Detection');
  multi.add(guiState.multiPoseDetection, 'maxPoseDetections')
    .min(1)
    .max(20)
    .step(1);
  multi.add(guiState.multiPoseDetection, 'minPoseConfidence', 0.0, 1.0);
  multi.add(guiState.multiPoseDetection, 'minPartConfidence', 0.0, 1.0);

  // nms Radius: controls the minimum distance between poses that are returned
  // defaults to 20, which is probably fine for most use cases
  // nms Radius：控制返回的姿勢之間的最小距離
  //默認為20，這對大多數用例來說可能都沒問題
  multi.add(guiState.multiPoseDetection, 'nmsRadius').min(0.0).max(40.0);
  multi.open();

  let output = gui.addFolder('Output');
  output.add(guiState.output, 'showVideo');
  output.add(guiState.output, 'showSkeleton');
  output.add(guiState.output, 'showPoints');
  output.add(guiState.output, 'showBoundingBox');
  output.open();

  architectureController.onChange(function (architecture) {
    guiState.changeToArchitecture = architecture;
  });

  algorithmController.onChange(function (value) {
    switch (guiState.algorithm) {
      case 'single-pose':
        multi.close();
        single.open();
        break;
      case 'multi-pose':
        single.close();
        multi.open();
        break;
    }
  });
}

//Sets up a frames per second panel on the top-left of the window
//在窗口的左上角設置每秒幀數
function setupFPS() {
  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.body.appendChild(stats.dom);
}

//Feeds an image to posenet to estimate poses - this is where the magic
//happens. This function loops with a requestAnimationFrame method.
//為posenet提供圖像以估計姿勢 - 這就是魔法
 //發生。 此函數使用requestAnimationFrame方法循環。
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');


  // since images are being fed from a webcam
  //因為圖像是從網絡攝像頭饋送的
  const flipHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  //用以比較是否經過五秒(過去時間)
  var d = new Date();
  var before = d.getSeconds();
  var times = 0;

  async function poseDetectionFrame() {
    if (guiState.changeToArchitecture) {
      // Important to purge variables and free up GPU memory
      //清除變量和釋放GPU內存很重要
      guiState.net.dispose();

      // Load the PoseNet model weights for either the 0.50, 0.75, 1.00, or 1.01
      // version
      //加載PoseNet模型權重為0.50,0.75,1.00或1.01
      //版本
      guiState.net = await posenet.load(+guiState.changeToArchitecture);

      guiState.changeToArchitecture = null;
    }

    // Begin monitoring code for frames per second
    //開始監視每秒幀數的代碼
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will slow
    // down the GPU
    //將圖像縮小到某個因子。 太大的圖像會變慢
    //順著GPU
    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = +guiState.input.outputStride;

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;

    switch (guiState.algorithm) {
      case 'single-pose':
        const pose = await guiState.net.estimateSinglePose(
          video, imageScaleFactor, flipHorizontal, outputStride);
        poses.push(pose);

        minPoseConfidence = +guiState.singlePoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.singlePoseDetection.minPartConfidence;
        break;
      case 'multi-pose':
        poses = await guiState.net.estimateMultiplePoses(
          video, imageScaleFactor, flipHorizontal, outputStride,
          guiState.multiPoseDetection.maxPoseDetections,
          guiState.multiPoseDetection.minPartConfidence,
          guiState.multiPoseDetection.nmsRadius);

        var data = await JSON.stringify(poses, null, 3);
        // Temporarily disable it for eye detection.
        //document.getElementById('myDiv02').value = data;

        //用以比較是否經過五秒(當下時間)
        var d1 = new Date();
        var now = d1.getSeconds();

        //避免當now在56-60秒間，這樣before會有一分鐘的時間絕對大於now
        //這樣即使如此有誤差，最多僅9秒誤差(00:56 - 01:05中間只更新一次)
        if (now == 0) {
          before = 0;
        }

        //console.log('now is:' + now);
        //console.log('before is:' + before);

        //用以比較是否經過五秒
        if (now - before == 5) {
          console.log('now is:' + now);
          console.log('before is:' + before);
          times++;
          document.getElementById('myDiv02').value = data;
          document.getElementById('myDiv03').value = times;
          before = now;
          console.log('now is(new):' + now);
          console.log('before is(new):' + before);
        }

        minPoseConfidence = +guiState.multiPoseDetection.minPoseConfidence;
        minPartConfidence = +guiState.multiPoseDetection.minPartConfidence;
        break;
    }

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    if (guiState.output.showVideo) {
      ctx.save();
      ctx.scale(-1, 1);
      ctx.translate(-videoWidth, 0);
      ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      ctx.restore();
    }

    // For each pose (i.e. person) detected in an image, loop through the poses
    // and draw the resulting skeleton and keypoints if over certain confidence
    // scores
    //對於在圖像中檢測到的每個姿勢（即人），循環通過姿勢
    //如果超過一定的信心，則繪製生成的骨架和關鍵點
    //分數

    var numberOfDectection = 0;
    poses.forEach(({ score, keypoints }) => {
      if (score >= minPoseConfidence) {
        if (guiState.output.showPoints) {
          drawKeypoints(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showSkeleton) {
          drawSkeleton(keypoints, minPartConfidence, ctx);
        }
        if (guiState.output.showBoundingBox) {
          drawBoundingBox(keypoints, ctx);
        }
      }

      // There are unstability in PoseNet detection. We use a window to store
      // a size of N previous detection. Sort in that duration and get the median
      // to ensure we rule out the extreme values that might just be accidental.
      numberOfDectection++;
      if (numberOfDectection == 1) {
        // We ignore all the cases for multiple pose detected. Only first pose will be stored in array.
        if (windowNextUpdateIndex < windowForMdeianSize) {
          windowForMedian[windowNextUpdateIndex] = keypoints;
          windowNextUpdateIndex++;
        } else {
          windowForMedian.shift();
          windowForMedian[windowForMdeianSize-1] = keypoints;
        }
      }

      
      var textToDisplayForDebug = "";
      // 取得眼睛的點
      if (getStableKeyPoints(1, "x") == null) {
        textToDisplayForDebug = "Frames is not accmulated enough to do median stable selection";
      } else {
        let eyesDirection = {
          "x": getStableKeyPoints(1, "x") - getStableKeyPoints(2, "x"),
          "y": getStableKeyPoints(1, "y") - getStableKeyPoints(2, "y"),
        }
        // 算斜率
        // Increase the x a bit to avoid the chance of divided by 0
        eyesDirection.x += 0.005
        let slopeRate = eyesDirection.y / eyesDirection.x
        textToDisplayForDebug += "\nCurrent Slope = " + slopeRate;
      }

      // 取得nose[0], leftEye[1], rightEye[2] calculate the area to judge if too close to camera.
      // leftEar[3], rightEar[4], leftShoulder[5], rightShoulder[6] could be used, but it is less
      // stable (you might not always detected the ears / shoulder)
      if (getStableKeyPoints(0, "x") == null) {
        textToDisplayForDebug = "Frames is not accmulated enough to do median stable selection";
      } else {
        let area = calculateTriangleArea(
          getStableKeyPoints(0, "x") - getStableKeyPoints(1, "x"),
          getStableKeyPoints(0, "y") - getStableKeyPoints(1, "y"),
          getStableKeyPoints(0, "x") - getStableKeyPoints(2, "x"),
          getStableKeyPoints(0, "y") - getStableKeyPoints(2, "y"),
          );
          textToDisplayForDebug += "\nArea = " + area;
      }

      document.getElementById('myDiv01').value = textToDisplayForDebug;


    });

    // End monitoring code for frames per second
    //結束每秒幀數的監控代碼
    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

//Kicks off the demo by loading the posenet model, finding and loading
//available camera devices, and setting off the detectPoseInRealTime function.
//通過加載posenet模型，查找和加載來開始演示
 //可用的相機設備，並設置detectPoseInRealTime功能。
export async function bindPage() {
  // Load the PoseNet model weights with architecture 0.75
  //使用體系結構0.75加載PoseNet模型權重
  const net = await posenet.load(0.75);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }

  setupGui([], net);
  setupFPS();
  detectPoseInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
// kick off the demo
//開始演示
bindPage();
