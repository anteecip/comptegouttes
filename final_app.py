import base64
import os
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from src.utils import predict_flow_curve_new

st.set_page_config(
    page_title="Uroflow Recorder",
    page_icon="🎙️",
    layout="centered"
)

st.markdown("""
<style>
.stApp { background-color: #0e0e0e; }
h1 { color: white; text-align: center; }

/* Cache le bouton zoom — inutilisable sur mobile */
/* button[title="View fullscreen"] {
    display: none !important;
} */
</style>
""", unsafe_allow_html=True)

# ✅ MutationObserver — surveille et réécrase en continu ce que Streamlit impose
# Nécessaire sur Android Chrome qui respecte strictement user-scalable=no et touch-action:none
st.markdown("""
<script>
(function() {
  var VIEWPORT_CONTENT = 'width=device-width, initial-scale=1.0, user-scalable=yes, minimum-scale=1.0, maximum-scale=10.0';

  function forceZoom() {
    // 1. Corrige la balise viewport
    var vp = document.querySelector('meta[name="viewport"]');
    if (vp) {
      if (vp.getAttribute('content') !== VIEWPORT_CONTENT) {
        vp.setAttribute('content', VIEWPORT_CONTENT);
      }
    } else {
      var meta = document.createElement('meta');
      meta.name = 'viewport';
      meta.content = VIEWPORT_CONTENT;
      document.head.appendChild(meta);
    }

    // 2. Supprime touch-action:none sur tous les éléments (Streamlit en injecte)
    var allElements = document.querySelectorAll('*');
    for (var i = 0; i < allElements.length; i++) {
      var el = allElements[i];
      var ta = el.style.touchAction;
      if (ta === 'none' || ta === 'pan-x' || ta === 'pan-y') {
        el.style.touchAction = 'auto';
      }
    }
  }

  // Lance une première fois
  forceZoom();

  // MutationObserver : relance à chaque modification du DOM par Streamlit
  var observer = new MutationObserver(function(mutations) {
    forceZoom();
  });
  observer.observe(document.documentElement, {
    attributes: true,
    childList: true,
    subtree: true,
    attributeFilter: ['style', 'content']
  });

  // Sécurité : relance aussi toutes les 500ms au cas où
  setInterval(forceZoom, 500);
})();
</script>
""", unsafe_allow_html=True)

st.markdown("""
<div id="top"></div>
<h1 style="color:#ffffff;">🎙️ Uroflow Meter</h1>
<p style="color:#555; text-align:center; font-size:0.85rem; margin-top:-10px;">
    powered by Compt'Gouttes
</p>
""", unsafe_allow_html=True)

# ── Placeholder graphique AVANT le composant audio (position haute) ──────────
graph_placeholder = st.empty()

# ── HTML du composant audio ───────────────────────────────────────────────────
COMPONENT_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes, minimum-scale=1.0, maximum-scale=5.0">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; touch-action: pan-x pan-y pinch-zoom; }
  body {
    background: #0e0e0e;
    color: white;
    font-family: Arial, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
    min-height: 420px;
  }
  #timer {
    font-size: 3rem;
    font-weight: bold;
    color: #ff9900;
    margin: 10px 0;
    min-height: 60px;
  }
  #record-btn {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    border: none;
    cursor: pointer;
    font-size: 1.1rem;
    font-weight: bold;
    color: white;
    background: radial-gradient(circle, #ff4444, #aa0000);
    box-shadow: 0 0 30px rgba(220,50,50,0.7);
    transition: all 0.3s;
    margin: 20px 0;
    -webkit-tap-highlight-color: transparent;
    touch-action: manipulation;
  }
  #record-btn.recording {
    background: radial-gradient(circle, #ffaa00, #cc6600);
    box-shadow: 0 0 50px rgba(255,150,0,0.9);
    animation: pulse 1s infinite;
  }
  @keyframes pulse {
    0%   { transform: scale(1); }
    50%  { transform: scale(1.08); }
    100% { transform: scale(1); }
  }
  #status {
    color: #aaa;
    font-size: 1rem;
    text-align: center;
    margin: 8px 0;
    min-height: 24px;
  }
  #warning {
    background: #1a1a00;
    border: 1px solid #ffcc00;
    border-radius: 8px;
    padding: 8px 16px;
    color: #ffcc00;
    font-size: 0.85rem;
    margin: 8px 0;
    display: none;
    text-align: center;
  }
</style>
</head>
<body>

<div id="timer">00:00</div>
<button id="record-btn" onclick="toggleRecording()">&#9210; ENREGISTRER</button>
<div id="status">Appuyez pour démarrer</div>
<div id="warning">⚠️ Arrêt automatique à 60s</div>

<script>
// ✅ Force le viewport Android depuis l'iframe
(function() {
  try {
    var vp = window.parent.document.querySelector('meta[name="viewport"]');
    if (vp) {
      vp.setAttribute('content',
        'width=device-width, initial-scale=1.0, user-scalable=yes, minimum-scale=1.0, maximum-scale=5.0'
      );
    }
  } catch(e) {}

  // Passive listeners pour ne pas bloquer le pinch sur Android
  document.addEventListener('touchstart', function(e) {}, { passive: true });
  document.addEventListener('touchmove',  function(e) {}, { passive: true });
  try {
    window.parent.document.addEventListener('touchstart', function(e) {}, { passive: true });
    window.parent.document.addEventListener('touchmove',  function(e) {}, { passive: true });
  } catch(e) {}
})();

// ── Protocole Streamlit Component ─────────────────────────────────────────────
(function () {
  function sendToStreamlit(type, data) {
    window.parent.postMessage(
      Object.assign({ isStreamlitMessage: true, type: type }, data),
      '*'
    );
  }
  window.addEventListener('load', function () {
    sendToStreamlit('streamlit:componentReady', { apiVersion: 1 });
  });
  window.Streamlit = {
    setComponentValue: function (value) {
      sendToStreamlit('streamlit:setComponentValue', { value: value, dataType: 'json' });
    }
  };
})();

// ── Logique d'enregistrement ──────────────────────────────────────────────────
var isRecording = false;
var timerInterval = null;
var seconds = 0;
var audioCtx = null;
var processor = null;
var stream = null;
var pcmChunks = [];

function formatTime(s) {
  var m = Math.floor(s / 60);
  var sec = s % 60;
  return (m < 10 ? '0' : '') + m + ':' + (sec < 10 ? '0' : '') + sec;
}

function toggleRecording() {
  if (!isRecording) startRecording();
  else stopRecording();
}

function startRecording() {
  var constraints = {
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      sampleRate: { ideal: 44100 },
      channelCount: 1
    }
  };

  navigator.mediaDevices.getUserMedia(constraints)
    .then(function (s) {
      stream = s;
      pcmChunks = [];

      try {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
      } catch (e) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }

      var source = audioCtx.createMediaStreamSource(stream);
      processor = audioCtx.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = function (e) {
        if (!isRecording) return;
        var data = e.inputBuffer.getChannelData(0);
        pcmChunks.push(new Float32Array(data));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      isRecording = true;
      seconds = 0;

      var btn = document.getElementById('record-btn');
      btn.className = 'recording';
      btn.textContent = 'STOP';
      document.getElementById('status').textContent = 'Enregistrement en cours...';
      document.getElementById('warning').style.display = 'none';
      document.getElementById('timer').textContent = '00:00';

      timerInterval = setInterval(function () {
        seconds++;
        document.getElementById('timer').textContent = formatTime(seconds);
        if (seconds >= 60) {
          document.getElementById('warning').style.display = 'block';
          stopRecording();
        }
      }, 1000);
    })
    .catch(function (err) {
      document.getElementById('status').textContent = 'Erreur micro : ' + err.message;
      console.error(err);
    });
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  clearInterval(timerInterval);

  if (processor) { processor.disconnect(); processor = null; }
  if (stream) { stream.getTracks().forEach(function (t) { t.stop(); }); }

  var btn = document.getElementById('record-btn');
  btn.className = '';
  btn.innerHTML = '&#9210; ENREGISTRER';
  document.getElementById('status').textContent = 'Encodage WAV...';

  var savedSeconds = seconds;

  setTimeout(function () {
    try {
      var sr = audioCtx ? audioCtx.sampleRate : 44100;
      var wavBlob = buildWAV(pcmChunks, sr);

      var reader = new FileReader();
      reader.onload = function (e) {
        var base64wav = e.target.result.split(',')[1];
        document.getElementById('status').textContent = '✅ ' + formatTime(savedSeconds) + ' enregistrées';
        window.Streamlit.setComponentValue(base64wav);
      };
      reader.readAsDataURL(wavBlob);

    } catch (e) {
      document.getElementById('status').textContent = 'Erreur encodage : ' + e.message;
    }
  }, 300);
}

// ── Encodage WAV (inchangé — préserve le gain maximal) ───────────────────────
function buildWAV(chunks, sampleRate) {
  var total = 0;
  for (var i = 0; i < chunks.length; i++) total += chunks[i].length;

  var merged = new Float32Array(total);
  var offset = 0;
  for (var i = 0; i < chunks.length; i++) {
    merged.set(chunks[i], offset);
    offset += chunks[i].length;
  }

  var int16 = new Int16Array(merged.length);
  for (var i = 0; i < merged.length; i++) {
    var s = Math.max(-1, Math.min(1, merged[i]));
    int16[i] = s < 0 ? s * 32768 : s * 32767;
  }

  var numChannels   = 1;
  var bitsPerSample = 16;
  var byteRate      = sampleRate * numChannels * bitsPerSample / 8;
  var blockAlign    = numChannels * bitsPerSample / 8;
  var dataSize      = int16.byteLength;

  var buf  = new ArrayBuffer(44 + dataSize);
  var view = new DataView(buf);

  function ws(off, str) {
    for (var i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i));
  }

  ws(0,  'RIFF');
  view.setUint32( 4, 36 + dataSize, true);
  ws(8,  'WAVE');
  ws(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1,  true);
  view.setUint16(22, numChannels,   true);
  view.setUint32(24, sampleRate,    true);
  view.setUint32(28, byteRate,      true);
  view.setUint16(32, blockAlign,    true);
  view.setUint16(34, bitsPerSample, true);
  ws(36, 'data');
  view.setUint32(40, dataSize, true);

  new Int16Array(buf, 44).set(int16);

  return new Blob([buf], { type: 'audio/wav' });
}
</script>
</body>
</html>
"""

# ── Écriture du composant sur disque ─────────────────────────────────────────
component_dir = Path(__file__).parent / "audio_component"
component_dir.mkdir(exist_ok=True)
(component_dir / "index.html").write_text(COMPONENT_HTML, encoding="utf-8")

# ── Déclaration et rendu du composant ────────────────────────────────────────
audio_recorder = components.declare_component("audio_recorder", path=str(component_dir))
wav_base64 = audio_recorder(key="recorder", default=None, height=500)

# ── Chargement du modèle (mis en cache) ──────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "models" / "final_model_RandomForest_best_init.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return joblib.load(f)

model = load_model()

# ── Analyse dès réception du WAV ─────────────────────────────────────────────
if wav_base64:
    wav_bytes = base64.b64decode(wav_base64)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        # ✅ Affiche "Analyse en cours" dans le placeholder pendant le calcul
        with graph_placeholder.container():
            st.markdown("""
            <div style="text-align:center; padding: 30px 0;">
                <p style="color:#ff9900; font-size:1.2rem;">🔬 Analyse en cours...</p>
                <p style="color:#555; font-size:0.85rem;">Veuillez patienter</p>
            </div>
            """, unsafe_allow_html=True)

        times_final, debits_final, metrics, masque_miction, debits = predict_flow_curve_new(
            model,
            tmp_path,
            seuil_score=0.6,
            window_length=0
        )

        # ✅ Rendu matplotlib → base64 PNG pour l'afficher dans components.html
        #    (iframe = contrôle total sur touch-action → zoom pinch Android garanti)
        import io
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')

        ax.fill_between(times_final, debits_final, alpha=0.25, color="steelblue")
        ax.plot(times_final, debits_final, color="steelblue", linewidth=2, label="Débit (mL/s)")
        ax.axhline(
            metrics["debit_max_mL_s"], color="red", linestyle="--",
            linewidth=1, label=f"Qmax = {metrics['debit_max_mL_s']} mL/s"
        )
        ax.set_xlabel("Temps (s)", color="white")
        ax.set_ylabel("Débit (mL/s)", color="white")
        ax.set_title("Uroflowmétrie acoustique", color="white")
        ax.legend(facecolor="#2a2a2a", labelcolor="white")
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        textstr = (
            f"Qmax  = {metrics['debit_max_mL_s']} mL/s\n"
            f"T mic = {metrics['duree_s']} s\n"
            f"Vol   = {metrics['volume_total_mL']} mL"
        )
        ax.text(
            0.98, 0.95, textstr, transform=ax.transAxes,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="#2a2a2a", alpha=0.9, edgecolor="#555"),
            fontsize=9, family="monospace", color="white"
        )
        plt.tight_layout()

        # Sauvegarde en mémoire → base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        # ✅ Graphique + métriques dans le même components.html
        #    → scroll libre, zoom pinch, pas de gap entre graphique et métriques
        qmax   = metrics['debit_max_mL_s']
        duree  = metrics['duree_s']
        volume = metrics['volume_total_mL']

        # ✅ Scroll automatique vers le haut dès que le graphique est prêt
        st.markdown('<script>window.parent.document.getElementById("top").scrollIntoView({behavior:"smooth"});</script>', unsafe_allow_html=True)

        with graph_placeholder.container():
            components.html(f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes, minimum-scale=1.0, maximum-scale=10.0">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ background:#0e0e0e; font-family:Arial,sans-serif; width:100%; }}

  /* ── Barre supérieure ── */
  #topbar {{ display:flex; justify-content:space-between; align-items:center;
             padding:4px 8px; }}
  #hint   {{ color:#555; font-size:0.72rem; }}
  #reset  {{ background:#2a2a2a; color:#ccc; border:1px solid #555;
             border-radius:6px; padding:4px 10px; font-size:0.72rem;
             cursor:pointer; display:none; touch-action:manipulation; }}

  /* ── Zone graphique zoomable ── */
  #wrap {{ overflow:hidden; width:100%; touch-action:none; }}
  #graph {{ width:100%; display:block; transform-origin:0 0;
            user-select:none; will-change:transform; }}

  /* ── Métriques collées sous le graphique ── */
  #metrics {{ display:flex; justify-content:space-around;
              padding:10px 8px 8px 8px; touch-action:pan-y; }}
  .metric-box {{
    flex:1; margin:0 4px;
    background:#1a1a1a; border:1px solid #555; border-radius:10px;
    padding:10px 6px; text-align:center;
  }}
  .metric-label {{ color:#888; font-size:0.72rem; margin-bottom:4px; }}
  .metric-value {{ color:#ffffff; font-size:1.3rem; font-weight:bold; }}
  .metric-unit  {{ color:#aaa;   font-size:0.7rem; }}
</style>
</head>
<body>

<div id="topbar">
  <span id="hint">👌 Pincez pour zoomer · 1 doigt déplace · 2× tap reset</span>
  <button id="reset" onclick="resetZoom()">↩ Reset</button>
</div>

<div id="wrap">
  <img id="graph" src="data:image/png;base64,{img_b64}" draggable="false">
</div>

<div id="metrics">
  <div class="metric-box">
    <div class="metric-label">⚡ Qmax</div>
    <div class="metric-value">{qmax}</div>
    <div class="metric-unit">mL/s</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">⏱️ Durée</div>
    <div class="metric-value">{duree}</div>
    <div class="metric-unit">s</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">💧 Volume</div>
    <div class="metric-value">{volume}</div>
    <div class="metric-unit">mL</div>
  </div>
</div>

<script>
var img      = document.getElementById('graph');
var wrap     = document.getElementById('wrap');
var resetBtn = document.getElementById('reset');
var scale = 1, transX = 0, transY = 0;
var lastScale = 1, startDist = 0;
var pinchOX = 0, pinchOY = 0;
var isPanning = false;
var panStartX = 0, panStartY = 0, panStartTX = 0, panStartTY = 0;
var lastTap = 0;

function resetZoom() {{
  scale = 1; transX = 0; transY = 0;
  img.style.transform = 'translate(0px,0px) scale(1)';
  resetBtn.style.display = 'none';
  wrap.style.touchAction = 'pan-y';   // ✅ re-autorise le scroll page
}}

function clamp(v, mn, mx) {{ return Math.min(mx, Math.max(mn, v)); }}

function applyTransform() {{
  var baseW = wrap.offsetWidth;
  var baseH = img.offsetHeight > 0 ? img.offsetHeight : baseW * 0.4;
  transX = clamp(transX, Math.min(0, baseW - baseW * scale), 0);
  transY = clamp(transY, Math.min(0, baseH - baseH * scale), 0);
  img.style.transform = 'translate(' + transX + 'px,' + transY + 'px) scale(' + scale + ')';
  var zoomed = scale > 1.05;
  resetBtn.style.display  = zoomed ? 'block' : 'none';
  wrap.style.touchAction  = zoomed ? 'none'  : 'pan-y';  // ✅ scroll libre si pas zoomé
}}

function touchDist(t) {{
  var dx = t[0].clientX - t[1].clientX;
  var dy = t[0].clientY - t[1].clientY;
  return Math.sqrt(dx*dx + dy*dy);
}}

wrap.addEventListener('touchstart', function(e) {{
  var now = Date.now();
  if (e.touches.length === 1 && (now - lastTap) < 300) {{ resetZoom(); return; }}
  lastTap = now;
  if (e.touches.length === 2) {{
    e.preventDefault();
    isPanning = false;
    startDist = touchDist(e.touches);
    lastScale = scale;
    var rect = wrap.getBoundingClientRect();
    var midX = (e.touches[0].clientX + e.touches[1].clientX) / 2 - rect.left;
    var midY = (e.touches[0].clientY + e.touches[1].clientY) / 2 - rect.top;
    pinchOX = (midX - transX) / scale;
    pinchOY = (midY - transY) / scale;
  }} else if (e.touches.length === 1 && scale > 1.05) {{
    e.preventDefault();
    isPanning = true;
    panStartX = e.touches[0].clientX; panStartY = e.touches[0].clientY;
    panStartTX = transX; panStartTY = transY;
  }}
}}, {{passive: false}});

wrap.addEventListener('touchmove', function(e) {{
  if (e.touches.length === 2) {{
    e.preventDefault();
    var newScale = clamp(lastScale * (touchDist(e.touches) / startDist), 1, 6);
    transX = transX + pinchOX * (scale - newScale);
    transY = transY + pinchOY * (scale - newScale);
    scale = newScale;
    applyTransform();
  }} else if (e.touches.length === 1 && isPanning) {{
    e.preventDefault();
    transX = panStartTX + (e.touches[0].clientX - panStartX);
    transY = panStartTY + (e.touches[0].clientY - panStartY);
    applyTransform();
  }}
}}, {{passive: false}});

wrap.addEventListener('touchend', function() {{
  isPanning = false;
  if (scale < 1.05) resetZoom();
}});
</script>
</body>
</html>""", height=480)

    except Exception as e:
        graph_placeholder.empty()
        st.error(f"Erreur lors de l'analyse : {e}")
    finally:
        os.unlink(tmp_path)
