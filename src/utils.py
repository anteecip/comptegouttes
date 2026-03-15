import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import logging
import pickle


''' 
ensemble des fonctions utilitaires pour le projet de prédiction du débit de miction à partir de l'audio 
vectorisation des audios.wac (librosa)
estimation de la fentre utile pour la prédiction (détection de l'impact du jet sur l'eau)
prédictions de débit à partir des features extraites

'''


def extract_features(y, sr):
    """
    Extrait un vecteur de features acoustiques depuis un signal audio.
    
    Normalsiation: Toutes les features d'énergie/amplitude sont transformées via log1p
    (ln(1+x)) avant agrégation pour réduire l'asymétrie des distributions
    et améliorer la robustesse des modèles de régression (SVR, Ridge, KNN).
    Random Forest est insensible à cette transformation mais elle ne lui nuit pas.
    
    Paramètres
    ----------
    y  : np.ndarray  — signal audio (float32, mono)
    sr : int         — fréquence d'échantillonnage (ex: 44100 Hz)
    
    Retourne
    --------
    np.ndarray de 42 features (float64)
    """

    # -------------------------
    # 0️⃣ Pré-traitement robuste micro
    # -------------------------
    
    # Filtre passe-haut du premier ordre (coeff ~0.97) qui atténue les très basses
    # fréquences (<80 Hz) : bruits de ventilation, vibrations de la pièce, grondements.
    # Désactivé ici car son effet sur la bande 1–4 kHz est négligeable,
    # et il peut légèrement déformer les MFCCs bas (coeff 1–3).
    # À réactiver si le bruit de fond basse fréquence est un problème constaté.
    # y = librosa.effects.preemphasis(y)

    # =========================
    # 1️⃣ MFCCs — 26 features (13 mean + 13 std)
    # Capturent la "texture" globale du spectre via le cepstre mel.
    # Les 13 coefficients résument la forme de l'enveloppe spectrale :
    # MFCC1 ≈ énergie globale, MFCC2–4 ≈ forme large bande,
    # MFCC5–13 ≈ détails fins de timbre.
    # n_fft=2048 → résolution fréquentielle de ~21 Hz à 44100 Hz
    # hop_length=512 → frame toutes les ~11.6 ms
    # =========================
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512
    )

    # CMVN (Cepstral Mean and Variance Normalization) : soustrait la moyenne
    # par coefficient pour neutraliser les effets de la réponse en fréquence
    # du microphone. Utile si les enregistrements proviennent de téléphones
    # très différents. Désactivé car peut effacer des informations utiles
    # sur le débit si les MFCCs ont une moyenne globalement corrélée au débit.
    # mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / \
    #        (np.std(mfcc, axis=1, keepdims=True) + 1e-8)

    # Version allégée : soustraction de la moyenne uniquement (sans normalisation
    # de la variance). Compromis entre robustesse inter-micros et conservation
    # de l'information de débit. Également désactivé pour les mêmes raisons.
    # mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

    # Les MFCCs peuvent être négatifs → pas de log1p applicable.
    # On agrège directement par mean et std sur l'axe temporel (axis=1).
    features = []
    features.extend(np.mean(mfcc, axis=1))  # 13 valeurs
    features.extend(np.std(mfcc, axis=1))   # 13 valeurs

    # =========================
    # 2️⃣ Features spectrales — 6 features (3 x mean+std)
    # Décrivent la distribution de l'énergie dans le domaine fréquentiel.
    # Toutes en Hz → valeurs larges et asymétriques → log1p appliqué
    # sur le vecteur de frames AVANT agrégation (mean/std cohérentes).
    # =========================

    # Centroïde spectral : "centre de gravité" du spectre en Hz.
    # Monte avec le débit car un jet puissant génère plus d'énergie
    # dans les hautes fréquences.
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid = np.log1p(spec_centroid)
    features.append(np.mean(spec_centroid))
    features.append(np.std(spec_centroid))

    # Bandwidth spectrale : étalement du spectre autour du centroïde en Hz.
    # Un impact fort génère un spectre plus large → bandwidth élevée.
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_bandwidth = np.log1p(spec_bandwidth)
    features.append(np.mean(spec_bandwidth))
    features.append(np.std(spec_bandwidth))

    # Rolloff spectral : fréquence en Hz en dessous de laquelle se concentre
    # 85% de l'énergie. Discrimine les débits faibles (rolloff bas, énergie
    # concentrée dans les graves) des débits forts (rolloff élevé).
    # roll_percent=0.85 est le seuil standard, ajustable à 0.90 si besoin.
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff = np.log1p(rolloff)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # =========================
    # 3️⃣ Spectral flatness — 2 features (mean + std)
    # Rapport géométrique/arithmétique du spectre, borné entre 0 et 1.
    # 0 → signal tonal (énergie concentrée sur quelques fréquences)
    # 1 → bruit blanc (énergie uniformément répartie)
    # Le bruit d'impact du jet est large bande → flatness élevée.
    # Utile pour détecter et distinguer le son du jet du bruit de fond ambiant
    # (voix, ventilation) qui aura une flatness différente et plus stable.
    # Pas de log1p car déjà borné entre 0 et 1 → pas d'asymétrie problématique.
    # =========================
    spec_flatness = librosa.feature.spectral_flatness(y=y)
    features.append(np.mean(spec_flatness))
    features.append(np.std(spec_flatness))

    # =========================
    # 4️⃣ RMS Energy — 2 features (mean + std)
    # Mesure la puissance moyenne du signal par frame.
    # Feature la plus directement corrélée au débit : un jet fort
    # génère un impact sonore plus puissant → RMS plus élevé.
    # log1p appliqué car valeurs très proches de 0 et très asymétriques
    # (silences avant/après miction tirent la distribution vers le bas).
    # =========================
    rms = librosa.feature.rms(y=y)
    rms = np.log1p(rms)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # =========================
    # 5️⃣ Zero-Crossing Rate — 2 features (mean + std)
    # Nombre de fois que le signal change de signe par frame, normalisé.
    # Proxy de la rugosité temporelle et de la richesse en hautes fréquences.
    # Utile pour distinguer le silence (ZCR faible et stable) du son d'impact
    # (ZCR plus élevé et variable). Borné entre 0 et 1 → pas de log1p.
    # =========================
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # =========================
    # 6️⃣ Flux spectral — 2 features (mean + std)
    # Mesure la variation du spectre entre frames successives.
    # Un jet turbulent (débit élevé) génère un spectre instable → flux élevé.
    # Un jet laminaire (débit faible) a un spectre stable → flux bas.
    # Feature dynamique complémentaire aux features statiques ci-dessus.
    # Normalisé par S.mean() * S.shape[0] pour être indépendant du volume
    # global → robustesse inter-téléphones.
    # log1p car valeurs positives et asymétriques.
    # Même n_fft/hop_length que les MFCCs pour cohérence temporelle.
    # =========================
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0)) / (S.mean() * S.shape[0])
    flux = np.log1p(flux)
    features.append(np.mean(flux))
    features.append(np.std(flux))

    # =========================
    # 7️⃣ Énergie bande 1–4 kHz — 2 features (mean + std)
    # Zone fréquentielle la plus informative pour le bruit d'impact
    # d'un jet liquide sur l'eau (documenté en acoustique des jets).
    # Avantage clé : les microphones de smartphones sont plus homogènes
    # dans cette bande qu'en dehors → robustesse inter-appareils.
    # Calculé sur les puissances (S²) plutôt que les amplitudes (S)
    # pour une mesure d'énergie physiquement correcte.
    # log1p car valeurs de puissance très asymétriques.
    # =========================
    # S = np.abs(librosa.stft(y))  # ⚠️ identique au S du bloc 6️⃣ → peut être mutualisé
    freqs = librosa.fft_frequencies(sr=sr)
    band = np.logical_and(freqs >= 1000, freqs <= 4000)
    band_energy = (S[band, :] ** 2).mean(axis=0)
    band_energy = np.log1p(band_energy)

    # Alternative normalisée (ratio énergie bande / énergie totale) :
    # encore plus robuste au volume global mais perd l'information d'énergie absolue.
    # À tester si la variabilité inter-téléphones reste un problème après entraînement.
    # band_energy_ratio = band_energy / (S.mean(axis=0) + 1e-8)

    features.append(np.mean(band_energy))
    features.append(np.std(band_energy))

    # Vecteur final : 26 (MFCCs) + 6 (spectrales) + 2 (flatness) +
    #                  2 (RMS) + 2 (ZCR) + 2 (flux) + 2 (bande) = 42 features
    return np.array(features)


def score_miction(y_frame, sr, rms_global=None):
    """
    Score combiné pour discriminer son de miction vs bruit parasite.
    Retourne un score entre 0 (bruit) et 1 (miction probable).
    
    Critères :
        1. fréquence avec la plus grande amplitude de la frame entre 0.1 à 8KHz (ex 1-4 kHz)
        2. Flatness spectrale (non conservé)
        3. RMS log-normalisé en % de l'amplitude globale de l'enregistrement (rms_frame / rms_global)
        4. Amplitude relative à l'enregistrement complet (%) (non conservé)

        doucble condition: 
        1) la fréquence dominante dooit se situer entre 2 valeurs
        2) le RMS relatif de chaque frame doit ête > à seuil_score
    """
    S     = np.abs(librosa.stft(y_frame)) # S= spectrogramme de magnitude de la frame. STFT = Short-Time Fourier Transform
    freqs = librosa.fft_frequencies(sr=sr) # vecteur des fréquences [0, 15.6, 31.2, ..., 16000] Hz  — toujours le même vecteur

    # critère 1 : Fréquence dominante = fréquence avec l'énergie maximale de la frame
    freq_dominante = freqs[np.argmax(S.mean(axis=1))]
    # print(f"Fréquence dominante : {freq_dominante:.1f} Hz")

    # Critère 2 : flatness
    flatness    = librosa.feature.spectral_flatness(y=y_frame)[0].mean()

    # # Critère 3 : RMS amplitude relative à l'enregistrement complet:rms_frame / rms_global
    rms         = librosa.feature.rms(y=y_frame)[0].mean()

    if rms_global is not None and rms_global > 0:
        amplitude_relative = np.clip(rms / rms_global, 0, 1) # np.clip borne une valeur entre un minimum et un maximum — tout ce qui dépasse les bornes est ramené à la borne.
    else:
        amplitude_relative = 0.0

    # on exclut les fréquences hors bornes typiques du jet d'urine (1-4 kHz) pour éviter les faux positifs sur les bruits de fond très graves ou très aigus"
    if freq_dominante < 300 or freq_dominante > 8000: 
        score = 0.0
    else:    
        # Score combiné
        score = (
                0.0 * (1 - flatness) +
                1.0 * amplitude_relative + #rms
                0.0 * freq_dominante # pour test à l'affichage uniquement, normalement entre 1K et 4 KHz mais pas dans les faits
                )

    return score

def predict_flow_curve_new(
    model_prev,                  # model utilsé pour les prévisions de débit
    wav_path,
    seuil_score            = 0.5,   # score minimum pour considérer frame valide
    nb_frame_silence_debut = 2,     # nb de frames à forcer True avant le début
    nb_frame_silence_fin   = 1,     # nb de frames à forcer True après la fin
    window_length          = 5,     # (5)fenêtre de lissage Savitzky-Golay← plus grand = plus lissé, plus petit = moins lissé, 0 = pas de lissage
    polyorder              = 2,     # (2)ordre du polynôme Savitzky-Golay← plus grand = suit mieux les pics, moins lissant
    frame_duration         = 0.5,   # durée d'une frame d'analyse en secondes (résolution temporelle de la courbe)
    overlap                = 0,   # chevauchement entre frames (0.5 = 50% → un point toutes les 0.25s)
    sr                     = 22050  # fréquence d'échantillonnage en Hz (standard librosa, à aligner avec l'entraînement)
):
    y, sr = librosa.load(wav_path, sr=sr)

    #test normalisation désactivé le 15 mars 26
    # y = librosa.util.normalize(y)

    # Calcul du RMS global de tout l'enregistrement (une seule fois)
    rms_global = librosa.feature.rms(y=y)[0].mean()

    frame_size = int(frame_duration * sr)
    hop_size   = int(frame_size * (1 - overlap))

    # ── 1. Prédictions brutes + score de miction sur toutes les frames ─
    times  = []
    debits = []
    scores = []

    for start in range(0, len(y) - frame_size, hop_size):
        frame    = y[start:start + frame_size]

        # Prédiction du débit
        features = extract_features(frame, sr)
        # choix du modele de prédiction : model = RandomForest, model2 = KNN, model3 = RandomForest_best
        debit    = model_prev.predict(features.reshape(1, -1))[0]
        debit    = max(debit, 0.0)

        # Score de confiance miction vs bruit parasite
        score    = score_miction(frame, sr, rms_global=rms_global)

        t = (start + frame_size / 2) / sr
        times.append(t)
        debits.append(debit)
        scores.append(score)

    times  = np.array(times)
    debits = np.array(debits)
    scores = np.array(scores)

    print("Scores de miction (0=bruit, 1=miction) :")
    print(scores)

    # ── 2. Construction du masque_miction ──────────────────────────────

    # Condition a : True si score > seuil_score
    masque_miction = scores > seuil_score

    # Condition b : forcer True les nb_frame_silence_debut frames
    #               précédant le premier True
    if masque_miction.any():
        idx_premier_true = int(np.argmax(masque_miction))
        idx_debut_ext    = max(0, idx_premier_true - nb_frame_silence_debut)
        masque_miction[idx_debut_ext:idx_premier_true] = True

    # Condition c : forcer True les nb_frame_silence_fin frames
    #               suivant le dernier True
    if masque_miction.any():
        idx_dernier_true = int(len(masque_miction) - np.argmax(masque_miction[::-1]) - 1)
        idx_fin_ext      = min(len(masque_miction) - 1, idx_dernier_true + nb_frame_silence_fin)
        masque_miction[idx_dernier_true:idx_fin_ext + 1] = True

    print(f"  Frames totales         : {len(debits)}")
    print(f"  Frames filtrées (bruit): {(~masque_miction).sum()}")
    print(f"  Frames conservées      : {masque_miction.sum()}")

    # ── 3. Mise à 0 des frames hors masque avant lissage ───────────────
    debits_filtered = debits.copy()
    debits_filtered[~masque_miction] = 0.0

    # ── 4. Lissage Savitzky-Golay sur le signal filtré ─────────────────
    if window_length >= 3 and len(debits_filtered) >= window_length:
        debits_smooth = savgol_filter(debits_filtered,
                                    window_length=window_length,
                                    polyorder=polyorder)
        debits_smooth = np.clip(debits_smooth, 0, None)
    else:
        # window_length=0 ou trop peu de frames → pas de lissage
        debits_smooth = debits_filtered.copy()

    # ── 5. Application du masque ───────────────────────────────────────
    times_miction  = times[masque_miction]
    debits_miction = debits_smooth[masque_miction]

    # ── 6. Calcul des métriques cliniques sur la miction uniquement ─────
    if len(debits_miction) > 1:
        dt           = np.mean(np.diff(times_miction))
        debit_max    = float(np.max(debits_miction))
        duree        = float(times_miction[-1] - times_miction[0])
        volume_total = float(np.sum(debits_miction) * dt)
        debit_moyen  = float(np.mean(debits_miction))
    else:
        dt = debit_max = duree = volume_total = debit_moyen = 0.0

    metrics = {
        "debit_max_mL_s"  : round(debit_max,   2),
        "duree_s"         : round(duree,        2),
        "volume_total_mL" : round(volume_total, 1),
        "debit_moyen_mL_s": round(debit_moyen,  2),
    }

    print("=" * 35)
    print(f"  Débit max       : {metrics['debit_max_mL_s']} mL/s")
    print(f"  Temps de miction: {metrics['duree_s']} s")
    print(f"  Volume total    : {metrics['volume_total_mL']} mL")
    print(f"  débit moyen     : {metrics['debit_moyen_mL_s']} mL/s")
    print("=" * 35)

    

    return times_miction, debits_miction, metrics, masque_miction, debits
