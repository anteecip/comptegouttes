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

st.markdown("""
<h1>🎙️ Uroflow Meter</h1>
<p style="color:#555; text-align:center; font-size:0.85rem; margin-top:-10px;">
    powered by Compt'Gouttes
</p>
""", unsafe_allow_html=True)

# ── HTML du composant audio ───────────────────────────────────────────────────
COMPONENT_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
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

      // Conversion Blob → base64 puis envoi à Python via Streamlit
      var reader = new FileReader();
      reader.onload = function (e) {
        var base64wav = e.target.result.split(',')[1]; // retire le préfixe data:audio/wav;base64,
        document.getElementById('status').textContent =
          '✅ ' + formatTime(savedSeconds) + ' — Analyse en cours...';
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

  var numChannels  = 1;
  var bitsPerSample = 16;
  var byteRate  = sampleRate * numChannels * bitsPerSample / 8;
  var blockAlign = numChannels * bitsPerSample / 8;
  var dataSize  = int16.byteLength;

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
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate,  true);
  view.setUint32(28, byteRate,    true);
  view.setUint16(32, blockAlign,  true);
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
        with st.spinner("🔬 Analyse en cours..."):
            times_final, debits_final, metrics, masque_miction, debits = predict_flow_curve_new(
                model,
                tmp_path,
                seuil_score=0.6,
                window_length=0
            )

        # ── Graphique ─────────────────────────────────────────────────────────
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
        st.pyplot(fig)
        plt.close(fig)

        # ── Métriques ─────────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("⚡ Qmax",    f"{metrics['debit_max_mL_s']} mL/s")
        col2.metric("⏱️ Durée",   f"{metrics['duree_s']} s")
        col3.metric("💧 Volume",  f"{metrics['volume_total_mL']} mL")

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
    finally:
        os.unlink(tmp_path)