const tripSelect = document.querySelector('#trip-select');
const matchButton = document.querySelector('#match-button');
const message = document.querySelector('#message');
const statusPill = document.querySelector('#api-status');
const map = document.querySelector('#map');
const mapEmpty = document.querySelector('#map-empty');
let currentTrip = null;
let currentMatches = [];
let modelReady = false;

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle('error', isError);
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value ?? 0);
}

function projectPoints(points, width = 1000, height = 520) {
  const lons = points.map(point => point.longitude);
  const lats = points.map(point => point.latitude);
  const minLon = Math.min(...lons), maxLon = Math.max(...lons);
  const minLat = Math.min(...lats), maxLat = Math.max(...lats);
  const spanLon = Math.max(maxLon - minLon, 1e-7);
  const spanLat = Math.max(maxLat - minLat, 1e-7);
  const padding = 48;
  const scale = Math.min((width - 2 * padding) / spanLon, (height - 2 * padding) / spanLat);
  const usedWidth = spanLon * scale, usedHeight = spanLat * scale;
  const offsetX = (width - usedWidth) / 2, offsetY = (height - usedHeight) / 2;
  return point => ({
    x: offsetX + (point.longitude - minLon) * scale,
    y: height - offsetY - (point.latitude - minLat) * scale,
  });
}

function renderMap() {
  if (!currentTrip?.observations?.length) return;
  const project = projectPoints(currentTrip.observations);
  const points = currentTrip.observations.map(project);
  const path = points.map((point, index) => `${index ? 'L' : 'M'}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(' ');
  const segments = currentMatches.map(match => {
    const from = project({longitude: match.matched_from_longitude, latitude: match.matched_from_latitude});
    const to = project({longitude: match.matched_to_longitude, latitude: match.matched_to_latitude});
    return `<line class="matched-segment" x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" />`;
  }).join('');
  const dots = points.map((point, index) => {
    const kind = index === 0 ? ' start-point' : index === points.length - 1 ? ' end-point' : '';
    return `<circle class="gps-point${kind}" cx="${point.x}" cy="${point.y}" r="${index % 4 === 0 ? 4.4 : 3}" />`;
  }).join('');
  map.innerHTML = `${segments}<path class="gps-path" d="${path}" />${dots}`;
  map.hidden = false;
  mapEmpty.hidden = true;
}

function renderTable(matches) {
  const body = document.querySelector('#results-body');
  const preview = matches.slice(0, 100);
  body.innerHTML = preview.map(match => `<tr>
    <td>${formatNumber(match.observation_id)}</td>
    <td>${Number(match.timestamp).toFixed(0)}</td>
    <td>${formatNumber(match.matched_arc_id)}</td>
    <td class="confidence">${(Number(match.confidence) * 100).toFixed(1)}%</td>
    <td>${Number(match.gps_latitude).toFixed(5)}, ${Number(match.gps_longitude).toFixed(5)}</td>
  </tr>`).join('') || '<tr><td colspan="5" class="empty-row">No matches returned.</td></tr>';
  document.querySelector('#results-count').textContent = matches.length > 100
    ? `Showing 100 of ${formatNumber(matches.length)} points`
    : `${formatNumber(matches.length)} points`;
}

async function loadTrip(tripId) {
  currentTrip = await requestJson(`/api/trips/${encodeURIComponent(tripId)}`);
  currentMatches = [];
  document.querySelector('#observation-count').textContent = formatNumber(currentTrip.observations.length);
  document.querySelector('#node-count').textContent = formatNumber(currentTrip.node_count);
  document.querySelector('#arc-count').textContent = formatNumber(currentTrip.arc_count);
  document.querySelector('#matched-count').textContent = '—';
  document.querySelector('#map-title').textContent = `Trip ${currentTrip.trip_id}`;
  document.querySelector('#results-count').textContent = 'No predictions yet';
  document.querySelector('#results-body').innerHTML = '<tr><td colspan="5" class="empty-row">Run route reconstruction to see model output.</td></tr>';
  renderMap();
  setMessage(`Loaded ${formatNumber(currentTrip.observations.length)} GPS observations. Run reconstruction to match them to roads.`);
  matchButton.disabled = !modelReady;
}

async function reconstruct() {
  if (!currentTrip) return;
  matchButton.disabled = true;
  setMessage('Scoring road candidates and decoding the route…');
  try {
    const response = await requestJson(`/api/trips/${encodeURIComponent(currentTrip.trip_id)}/match`, {method: 'POST'});
    currentMatches = response.matches;
    document.querySelector('#matched-count').textContent = formatNumber(currentMatches.length);
    renderMap();
    renderTable(currentMatches);
    setMessage(`Route reconstructed for ${formatNumber(currentMatches.length)} observations.`);
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    matchButton.disabled = !modelReady;
  }
}

async function start() {
  try {
    const health = await requestJson('/api/health');
    modelReady = health.model_loaded;
    statusPill.classList.add(health.model_loaded ? 'online' : 'offline');
    statusPill.innerHTML = `<i></i>${health.model_loaded ? 'Model ready' : 'Train model first'}`;
    const payload = await requestJson('/api/trips/');
    tripSelect.innerHTML = payload.trips.map(id => `<option value="${id}">Trip ${id}</option>`).join('');
    tripSelect.disabled = payload.trips.length === 0;
    if (payload.trips.length) await loadTrip(payload.trips[0]);
    else setMessage('No student-test trips were found in the configured data directory.', true);
    if (!health.model_loaded) {
      matchButton.disabled = true;
      setMessage('The API is ready, but no trained checkpoint was found. Run `python main.py train`, then refresh.', true);
    }
  } catch (error) {
    statusPill.classList.add('offline');
    statusPill.innerHTML = '<i></i>API unavailable';
    setMessage(`${error.message}. Start the API with: uvicorn backend.main:app --reload`, true);
  }
}

tripSelect.addEventListener('change', () => loadTrip(tripSelect.value).catch(error => setMessage(error.message, true)));
matchButton.addEventListener('click', reconstruct);
start();
