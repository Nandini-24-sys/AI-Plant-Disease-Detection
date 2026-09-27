const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snap = document.getElementById('snap');
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
    video.play();
  }).catch(err => {
    console.warn('Camera permission denied or no camera', err);
  });
}
snap && snap.addEventListener('click', () => {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob((blob) => {
    const fd = new FormData();
    fd.append('images', blob, 'capture.jpg');
    fetch('/upload', { method: 'POST', body: fd, credentials: 'include' })
      .then(resp => resp.text())
      .then(html => {
        document.open();
        document.write(html);
        document.close();
      }).catch(err => alert('Upload failed: ' + err));
  }, 'image/jpeg', 0.9);
});
