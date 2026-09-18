(() => {
  const $ = (selector) => document.querySelector(selector);
  const state = { blob: null, image: null, top: null, bottom: null, x: null, full: null, crop: null };
  const canvas = $('#roi-canvas');
  const ctx = canvas.getContext('2d');

  function show(selector, visible = true) { $(selector).classList.toggle('hidden', !visible); }
  function error(message) { $('#error-box').textContent = `⚠ ${message}`; show('#error-box'); show('#progress', false); }
  function status(text, percent = 0) { $('#progress-text').textContent = text; $('#progress-bar').style.width = `${percent}%`; show('#progress'); show('#error-box', false); }

  async function compress(file) {
    if (!file.type.startsWith('image/')) throw new Error('Choose an image file.');
    const image = new Image();
    image.src = URL.createObjectURL(file);
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error('The image could not be opened.')); });
    let width = image.naturalWidth, height = image.naturalHeight;
    const maxWidth = 1500;
    if (width > maxWidth) { height = Math.round(height * maxWidth / width); width = maxWidth; }
    const output = document.createElement('canvas'); output.width = width; output.height = height;
    output.getContext('2d').drawImage(image, 0, 0, width, height);
    return await new Promise((resolve, reject) => output.toBlob((blob) => {
      if (!blob) return reject(new Error('Image compression failed.'));
      resolve(blob);
    }, 'image/jpeg', blobQuality(blobSize(file))));
  }
  function blobSize(file) { return file.size; }
  function blobQuality(size) { return size > 7 * 1024 * 1024 ? .68 : .84; }

  function draw() {
    if (!state.image) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
    if (state.top !== null) line(state.top * canvas.height, '#5b9bff', 'T');
    if (state.bottom !== null) {
      line(state.bottom * canvas.height, '#35d39d', 'B');
      ctx.fillStyle = 'rgba(53,211,157,.09)';
      ctx.fillRect(0, state.top * canvas.height, canvas.width, (state.bottom - state.top) * canvas.height);
      line(state.top * canvas.height, '#5b9bff', 'T');
      line(state.bottom * canvas.height, '#35d39d', 'B');
    }
  }
  function line(y, color, label) {
    ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.setLineDash([8, 5]);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc((state.x || .5) * canvas.width, y, 9, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#071321'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center'; ctx.fillText(label, (state.x || .5) * canvas.width, y + 4);
  }
  function tap(event) {
    event.preventDefault();
    if (!state.image || (state.top !== null && state.bottom !== null)) return;
    const point = event.touches ? event.touches[0] : event;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (point.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (point.clientY - rect.top) / rect.height));
    if (state.top === null) { state.top = y; state.x = x; $('#tap-instruction').textContent = 'Tap the bottom edge'; $('#canvas-hint').textContent = 'Tap the BOTTOM of the stack'; }
    else if (y > state.top + .04) { state.bottom = y; $('#tap-instruction').textContent = 'ROI ready'; $('#canvas-hint').textContent = 'Stack marked — analyze when ready'; $('#analyze-button').disabled = false; }
    else { $('#canvas-hint').textContent = 'Bottom must be below the top'; return; }
    $('#roi-readout').textContent = state.bottom === null ? `Top ${(state.top * 100).toFixed(0)}%` : `ROI ${((state.bottom - state.top) * 100).toFixed(0)}% high`;
    draw();
  }
  canvas.addEventListener('click', tap); canvas.addEventListener('touchend', tap, { passive: false });

  async function handleFile(file) {
    if (!file) return;
    try {
      status('Compressing image…', 15);
      state.blob = await compress(file);
      state.image = new Image();
      state.image.onload = () => {
        canvas.width = state.image.naturalWidth; canvas.height = state.image.naturalHeight; draw();
        state.top = state.bottom = state.x = null; $('#analyze-button').disabled = true;
        show('#preview-area'); show('#progress', false); $('#canvas-hint').textContent = 'Tap the TOP of the stack';
      };
      state.image.src = URL.createObjectURL(state.blob);
    } catch (err) { error(err.message); }
  }
  $('#camera-input').addEventListener('change', (e) => handleFile(e.target.files[0]));
  $('#gallery-input').addEventListener('change', (e) => handleFile(e.target.files[0]));
  $('#reset-roi').addEventListener('click', () => { state.top = state.bottom = state.x = null; $('#analyze-button').disabled = true; $('#tap-instruction').textContent = 'Tap the top edge'; $('#canvas-hint').textContent = 'Tap the TOP of the stack'; draw(); });
  $('#auto-button').addEventListener('click', () => doCount(true));
  $('#analyze-button').addEventListener('click', () => doCount(false));
  $('#new-photo').addEventListener('click', () => { show('#result-content', false); show('#empty-result'); show('#metrics', false); window.scrollTo({ top: 0, behavior: 'smooth' }); });

  async function doCount(auto) {
    if (!state.blob) return error('Choose a photo first.');
    if (!auto && (state.top === null || state.bottom === null)) return error('Tap the top and bottom of the stack.');
    const form = new FormData(); form.append('photo', state.blob, 'ring-stack.jpg');
    form.append('top_frac', auto ? '.04' : state.top); form.append('bottom_frac', auto ? '.96' : state.bottom); form.append('x_frac', auto ? '.5' : state.x); form.append('auto_roi', auto ? '1' : '0');
    $('#analyze-button').disabled = true; status('Analyzing stack…', 35);
    let percent = 35; const timer = setInterval(() => { percent = Math.min(86, percent + 7); $('#progress-bar').style.width = `${percent}%`; $('#progress-text').textContent = percent > 60 ? 'Validating spacing…' : 'Detecting boundaries…'; }, 400);
    try {
      const response = await fetch('/count', { method: 'POST', body: form });
      const raw = await response.text(); let result;
      try { result = JSON.parse(raw); } catch (_) { throw new Error(`Server returned HTTP ${response.status}.`); }
      if (!response.ok || result.success === false) throw new Error(result.error || 'Counting failed.');
      state.full = result.image_data_url; state.crop = result.crop_data_url; render(result);
    } catch (err) { error(err.message); } finally { clearInterval(timer); $('#analyze-button').disabled = false; }
  }
  function render(d) {
    show('#progress', false); show('#empty-result', false); show('#result-content'); show('#metrics');
    $('#result-image').src = d.image_data_url; $('#kpi-count').textContent = d.ring_count; $('#kpi-count-sub').textContent = `${d.detected_rings} boundaries · ${d.rejected_detections} rejected`;
    $('#kpi-confidence').textContent = `${Math.round(d.confidence_score)}%`; $('#kpi-confidence-sub').textContent = `${d.confidence} evidence confidence`;
    $('#kpi-time').textContent = `${d.processing_time}s`; $('#kpi-status').textContent = d.confidence === 'Low' ? 'REVIEW' : 'STABLE'; $('#kpi-status-sub').textContent = d.quality_warning ? 'Image quality warning' : 'Result ready';
    $('#result-state').textContent = d.confidence.toUpperCase(); $('#result-state').className = `state-pill confidence-${d.confidence.toLowerCase()}`;
    metric('sharpness', d.sharpness, Math.min(100, d.sharpness / 3.5)); metric('brightness', d.brightness, d.brightness / 2.55); metric('glare', `${d.glare}%`, Math.min(100, d.glare)); metric('consistency', `${Math.round(d.consistency * 100)}%`, d.consistency * 100);
    $('#metric-detected').textContent = d.detected_rings; $('#metric-rejected').textContent = `${d.rejected_detections} / ${d.possible_missing_rings}`;
    if (d.quality_warning) { $('#quality-warning').textContent = 'Image quality may reduce reliability. Try a steadier, brighter, straighter photo with less glare.'; show('#quality-warning'); } else show('#quality-warning', false);
    loadHistory(); $('#result-content').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  function metric(name, value, percent) { $(`#metric-${name}`).textContent = value; $(`#bar-${name}`).style.width = `${Math.max(0, Math.min(100, percent))}%`; }
  document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => { document.querySelectorAll('.tab').forEach((item) => item.classList.remove('active')); tab.classList.add('active'); const crop = tab.dataset.view === 'crop'; $('#result-image').src = crop ? state.crop : state.full; $('.image-label').textContent = crop ? 'CROPPED STACK' : 'ANNOTATED'; }));
  function download(data, name) { const link = document.createElement('a'); link.href = data; link.download = name; link.click(); }
  $('#download-full').addEventListener('click', () => download(state.full, 'ringcount-annotated.jpg')); $('#download-crop').addEventListener('click', () => download(state.crop, 'ringcount-crop.jpg'));

  async function loadHistory() {
    try {
      const response = await fetch('/api/history'); const payload = await response.json(); const body = $('#history-body');
      body.innerHTML = payload.items.length ? payload.items.map((item) => `<tr><td>${new Date(item.created_at).toLocaleString()}</td><td>${escapeHtml(item.image_name)}</td><td><b>${item.ring_count}</b></td><td class="confidence-${item.confidence.toLowerCase()}">${item.confidence} · ${Math.round(item.confidence_score)}%</td><td>${item.processing_time}s</td></tr>`).join('') : '<tr><td colspan="5" class="muted">No counts yet.</td></tr>';
    } catch (_) { /* history is an enhancement; counting remains usable */ }
  }
  function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[character])); }
  $('#refresh-history').addEventListener('click', loadHistory); loadHistory();
  fetch('/status').then((response) => response.json()).then((data) => { $('#server-status').textContent = 'ONLINE'; $('#device-chip').textContent = data.device.toUpperCase(); $('#model-chip').textContent = data.model.loaded ? 'MODEL OK' : 'CV FALLBACK'; $('#system-details').textContent = `Version ${data.version} · ${data.device.toUpperCase()} · ${data.model.loaded ? 'YOLO weights loaded' : 'No YOLO weights found; classical detector active'}.`; }).catch(() => { $('#server-status').textContent = 'OFFLINE'; $('#system-details').textContent = 'The server status endpoint could not be reached.'; });
})();