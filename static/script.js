//login_alert
const flashContainer = document.getElementById("flash-container");
if (flashContainer) {
    const messagesData = flashContainer.dataset.messages;
    if (messagesData) {
        const messages = messagesData.split("|");
        messages.forEach(msg => {
            alert(msg);
        });
    }
}

//images_modal
document.addEventListener("DOMContentLoaded", () => {
  const modal = document.getElementById("imgModal");
  const modalImg = document.getElementById("modalImg");
  const closeBtn = document.getElementsByClassName("close")[0];

  // 이미지 클릭 이벤트
  document.querySelectorAll(".clickable-img").forEach(img => {
    img.addEventListener("click", () => {
      modal.style.display = "block";
      modalImg.src = img.src;
    });
  });

  // 모달 닫기
  closeBtn.onclick = () => { modal.style.display = "none"; };
  window.onclick = (event) => { if (event.target == modal) modal.style.display = "none"; };
});



//main_program
document.addEventListener("DOMContentLoaded", async () => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const statusEl = document.getElementById('status');
  const countdownEl = document.getElementById('countdownDisplay');
  const recordTimeEl = document.getElementById('recording-time');
  const recordBtn = document.getElementById("recordBtn");

  canvas.width = 640;
  canvas.height = 480;  

  let mediaRecorder = null;
  let recordedChunks = [];
  let recording = false;
  let timerInterval = null;
  let recordStartTime = null;

  let highlightRecording = false;
  let highlightRecorder = null;
  let highlightChunks = [];
  let highlightStartTime = null;

  let forwardHeadStartTime = null;
  const FORWARD_THRESHOLD = 20;    // CVA 임계값
  const HOLD_DURATION = 5;         // 5초 이상 지속해야 녹화 시작
  const HIGHLIGHT_POST_FRAMES = 300; // 10초 후 (30fps 기준)

  let forward_count = 0;
  let prevStatus = "Normal";

  const status = "Normal";

  const alertSound = new Audio('static/turtle.mp3');

  video.addEventListener('loadedmetadata', () => {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
});

  // =====================
  // 시간 포맷 함수
  // =====================
  function formatTime(seconds) {
    const h = String(Math.floor(seconds / 3600)).padStart(2, "0");
    const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
    const s = String(Math.floor(seconds % 60)).padStart(2, "0");
    return `${h}:${m}:${s}`;
  }

  // =====================
  // cva 계산
  // =====================
  function calculateCVA(shoulder, ear) {
    const dx = ear.x - shoulder.x;
    const dy = ear.y - shoulder.y;
    const angle = Math.atan2(dy, dx) * 180 / Math.PI;
    return Math.abs(90 - Math.abs(angle));
  }

  const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${file}`
  });

  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  pose.onResults(onResults);

  // =====================
  // 카메라 시작
  // =====================
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  video.srcObject = stream;
  await video.play();

  // =====================
  // 풀영상 녹화
  // =====================
  mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9" });
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };

  mediaRecorder.onstop = async () => {
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    recordedChunks = [];
    const dataUrl = await blobToDataURL(blob);

    const res = await fetch("/full_videos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video: dataUrl, type: "full", count: forward_count})
    });
    const json = await res.json();
    recordTimeEl.innerText = "풀영상 저장 완료 (거북목 "+ forward_count + "회)";

    forward_count = 0;
  };

  async function blobToDataURL(blob) {
    return new Promise(resolve => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    });
  }

  // =====================
  // Pose 결과 처리
  // =====================
  function onResults(results) {
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (results.poseLandmarks) {
      const lm = results.poseLandmarks;
      const shoulder_r = lm[12], ear_r = lm[8];
      const shoulder_l = lm[11], ear_l = lm[7];

      const cva_r = calculateCVA(shoulder_r, ear_r);
      const cva_l = calculateCVA(shoulder_l, ear_l);
      const cva_avg = (cva_r + cva_l) / 2;
      const status = cva_avg >= FORWARD_THRESHOLD ? "Forward Head" : "Normal";

      statusEl.innerText = `CVA: ${cva_avg.toFixed(1)}°, 상태: ${status}`;
      ctx.strokeStyle = cva_avg >= FORWARD_THRESHOLD ? "red" : "green";

      // --- 화면 색 변화 ---
      if (cva_avg >= FORWARD_THRESHOLD) {
        document.body.style.backgroundColor = "#8ED1A0";

        alertSound.play();
      } else {
        document.body.style.backgroundColor = "white";  // 정상일 때 배경을 흰색으로
      }

      const points = [shoulder_r, ear_r, shoulder_l, ear_l];
      points.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x * canvas.width, p.y * canvas.height, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "blue";
        ctx.fill();
      });

    // --- 선 연결 (귀→어깨, 어깨→어깨) ---
      const line = (a, b, color) => {
        ctx.beginPath();
        ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
        ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      };

      if (cva_avg >= FORWARD_THRESHOLD) {
        line(shoulder_r, ear_r, "red");   // 거북목 → 빨강
      } else {
        line(shoulder_r, ear_r, "green"); // 정상 → 초록
      }

      // 왼쪽 귀–어깨 선
      if (cva_avg >= FORWARD_THRESHOLD) {
        line(shoulder_l, ear_l, "red");
      } else {
        line(shoulder_l, ear_l, "green");
      }

      line(shoulder_r, shoulder_l, "white");

      // 거북목 감지 시 하이라이트 시작
      if (cva_avg >= FORWARD_THRESHOLD && recording) {
        if (!forwardHeadStartTime) {
          forwardHeadStartTime = Date.now();
        } else {
          const elapsed = (Date.now() - forwardHeadStartTime) / 1000;
          if (elapsed >= HOLD_DURATION && !highlightRecording) {
            startHighlightRecording();
          }
        }
      } else {
        if (highlightRecording) stopHighlightRecording();
        forwardHeadStartTime = null;
      }
    } else {
      statusEl.innerText = "CVA: 0.0°, 상태: 인식x";
      document.body.style.backgroundColor = "white";
      if (highlightRecording) stopHighlightRecording();
      forwardHeadStartTime = null;
    }

    ctx.restore();
  }

  async function detectPose() {
    await pose.send({ image: video });
    requestAnimationFrame(detectPose);
  }
  detectPose();

  // =====================
  // 사진 찍기
  // =====================
  document.getElementById("captureBtn").addEventListener("click", () => {
    let countdown = 3;
    const originalText = statusEl.innerText;
    const interval = setInterval(() => {
      countdownEl.innerText = `사진 저장까지 ${countdown}초`;
      countdown--;
      if (countdown < 0) {
        clearInterval(interval);
        countdownEl.innerText = "";
        const dataUrl = canvas.toDataURL("image/png");
        const cvaText = originalText.split("°")[0].replace("CVA: ", "");
        const statusText = originalText.split("상태: ")[1];
        fetch("/save_photo", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cva: parseFloat(cvaText), status: statusText, image: dataUrl })
        }).then(res => res.json()).then(() => alert("사진 저장 완료!")).catch(() => alert("저장 실패"));
      }
    }, 1000);
  });

  // =====================
  // 영상 녹화 버튼
  // =====================
  recordBtn.addEventListener("click", () => {
    if (recording) {
      mediaRecorder.stop();
      recording = false;
      clearInterval(timerInterval);
      recordBtn.innerText = "영상 촬영";
      recordTimeEl.innerText = "저장 중";
    } else {
      mediaRecorder.start();
      recording = true;
      recordStartTime = Date.now();
      recordBtn.innerText = "촬영 중지";
      timerInterval = setInterval(() => {
        let elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
        recordTimeEl.innerText = "촬영 시간: " + elapsed + "초";
      }, 1000);
    }
  });

  // =====================
  // 하이라이트 녹화 시작
  // =====================
  function startHighlightRecording() {
    highlightRecording = true;
    highlightStartTime = Date.now(); // 하이라이트 시작 시간 기록
    highlightChunks = [];

    forward_count++;

    highlightRecorder = new MediaRecorder(canvas.captureStream(30), { mimeType: "video/webm;codecs=vp9" });

    highlightRecorder.ondataavailable = (e) => { if (e.data.size > 0) highlightChunks.push(e.data); };

    highlightRecorder.onstop = async () => {
      const blob = new Blob(highlightChunks, { type: "video/webm" });
      const dataUrl = await blobToDataURL(blob);

      fetch("/save_highlight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video: dataUrl,
          duration: Math.floor((Date.now() - highlightStartTime) / 1000)
        })
      }).then(res => res.json())
        .then(json => console.log("하이라이트 저장 성공:", json))
        .catch(err => console.error("하이라이트 저장 오류:", err));

      highlightRecording = false;
    };

    highlightRecorder.start();
    console.log("하이라이트 녹화 시작");

  }
  // =====================
  // 하이라이트 녹화 종료
  // =====================
  function stopHighlightRecording() {
    if (highlightRecorder && highlightRecording) {
      highlightRecorder.stop();
      console.log("하이라이트 녹화 종료");
    }
  }
});

function goToLink(element) {
    const url = element.getAttribute("data-href");
    window.location.href = url;
  }

// Service Worker 등록
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/service-worker.js')
      .then(reg => console.log('Service Worker registered ✅', reg))
      .catch(err => console.error('Service Worker registration failed ❌', err));
  }