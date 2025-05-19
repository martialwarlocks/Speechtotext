from flask import Flask, request, jsonify, render_template, send_from_directory
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from matplotlib.figure import Figure
import os
import matplotlib
from scipy.signal import hilbert, butter, lfilter
from scipy import signal

matplotlib.use("Agg")  # Required for headless operation

app = Flask(__name__)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)


# Create placeholder waveform image
def create_placeholder_waveform():
    fig = Figure(figsize=(10, 4))
    
    # Add waveform subplot
    ax1 = fig.add_subplot(2, 1, 1)
    x = np.linspace(0, 10, 1000)
    y = np.zeros(1000)
    ax1.plot(x, y, color="#e0e0e0")
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_xlim(0, 10)
    ax1.set_title("Waveform")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    # Add spectrogram subplot
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(np.zeros((100, 100)), aspect='auto', cmap='viridis')
    ax2.set_title("Spectrogram")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    
    fig.tight_layout()
    fig.savefig(
        "static/placeholder-waveform.png",
        bbox_inches="tight",
        transparent=True,
        dpi=100,
    )
    plt.close(fig)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


@app.route("/visualize-waveform", methods=["POST"])
def visualize_waveform():
    data = request.json
    audio_data = data.get("audioData", [])
    transcript = data.get("transcript", "")
    sample_rate = data.get("sampleRate", 44100)  # Default to 44.1kHz if not provided

    # Generate waveform visualization
    waveform_url = create_waveform_visualization(audio_data, transcript, sample_rate)

    return jsonify({"waveformUrl": waveform_url})


def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a bandpass filter for voice frequencies"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a bandpass filter to the data"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def normalize_audio(audio_data):
    """Properly normalize audio data to range [-1, 1]"""
    if isinstance(audio_data[0], int):
        # If data is integer format (e.g., 8-bit or 16-bit PCM)
        max_possible_value = 128.0  # Assuming 8-bit audio; adjust for 16-bit if needed
        normalized = [(x / max_possible_value) - 1 for x in audio_data]
    else:
        # If data is already float format
        max_val = max(abs(min(audio_data)), abs(max(audio_data)))
        if max_val > 0:  # Avoid division by zero
            normalized = [x / max_val for x in audio_data]
        else:
            normalized = audio_data
    return normalized


def get_signal_envelope(signal_data, num_points=1000):
    """Calculate the envelope of the signal using Hilbert transform"""
    # Get the envelope from the analytic signal
    analytic_signal = hilbert(signal_data)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Downsample the envelope if needed
    if len(amplitude_envelope) > num_points:
        indices = np.linspace(0, len(amplitude_envelope) - 1, num_points, dtype=int)
        amplitude_envelope = amplitude_envelope[indices]
    
    return amplitude_envelope


def create_waveform_visualization(audio_data, transcript, sample_rate=44100):
    if not audio_data:
        return "/static/placeholder-waveform.png"

    # Normalize audio data
    normalized_data = normalize_audio(audio_data)
    
    # Apply bandpass filter for voice frequencies (typically 300-3400 Hz)
    if len(normalized_data) > 20:  # Only filter if we have enough data points
        filtered_data = butter_bandpass_filter(normalized_data, 300, 3400, sample_rate)
    else:
        filtered_data = normalized_data
    
    # Calculate time axis in seconds
    duration = len(normalized_data) / sample_rate
    time_axis = np.linspace(0, duration, len(normalized_data))
    
    # Create a figure with two subplots: waveform and spectrogram
    fig = Figure(figsize=(10, 6))
    
    # Waveform plot
    ax1 = fig.add_subplot(2, 1, 1)
    
    # Plot both raw waveform and filtered data for comparison
    ax1.plot(time_axis, normalized_data, color="#3498db", linewidth=0.5, alpha=0.5, label="Raw")
    ax1.plot(time_axis, filtered_data, color="#e74c3c", linewidth=0.8, label="Filtered")
    
    # Calculate and plot the envelope
    if len(filtered_data) > 20:  # Only calculate envelope if we have enough data
        envelope = get_signal_envelope(filtered_data)
        envelope_time = np.linspace(0, duration, len(envelope))
        ax1.plot(envelope_time, envelope, color="#2ecc71", linewidth=1.5, alpha=0.7, label="Envelope")
    
    # Add legend and labels
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Voice Waveform")
    ax1.set_ylabel("Amplitude")
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_xlim(0, duration)
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis to show time in seconds
    ax1.set_xlabel("Time (s)")
    
    # Spectrogram plot (very useful for voice analysis)
    ax2 = fig.add_subplot(2, 1, 2)
    
    if len(normalized_data) > 256:  # Need enough data for a spectrogram
        # Compute spectrogram
        nperseg = min(256, len(normalized_data) // 8)  # Window size
        if nperseg < 16:  # Ensure minimum window size
            nperseg = 16
            
        f, t, Sxx = signal.spectrogram(
            filtered_data, 
            fs=sample_rate, 
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling='spectrum'
        )
        
        # Plot spectrogram (focus on lower frequencies where voice usually exists)
        max_freq_idx = np.searchsorted(f, 5000)  # Display up to 5kHz
        max_freq_idx = max(max_freq_idx, 1)  # Ensure at least one frequency bin
        
        spec = ax2.pcolormesh(
            t, 
            f[:max_freq_idx], 
            10 * np.log10(Sxx[:max_freq_idx] + 1e-10),  # Convert to dB scale
            shading='gouraud', 
            cmap='viridis'
        )
        fig.colorbar(spec, ax=ax2, label='Power/Frequency (dB/Hz)')
    else:
        # If not enough data, create an empty spectrogram
        ax2.text(0.5, 0.5, "Not enough data for spectrogram", 
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes)
    
    ax2.set_title("Spectrogram")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    
    # Add transcript as text in a text box if it exists
    if transcript:
        # Truncate transcript if too long
        display_text = transcript[:100] + "..." if len(transcript) > 100 else transcript
        fig.text(0.5, 0.01, f"Transcript: {display_text}",
                 horizontalalignment="center",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
                 fontsize=8)
    
    fig.tight_layout()
    
    # Add a bit more space at the bottom for the transcript
    if transcript:
        fig.subplots_adjust(bottom=0.15)
    
    # Save to a base64 string
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=100
    )
    buf.seek(0)
    img_str = "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_str


@app.route("/advanced-analysis", methods=["POST"])
def advanced_analysis():
    """Perform more advanced audio analysis"""
    data = request.json
    audio_data = data.get("audioData", [])
    full_transcript = data.get("fullTranscript", "")
    sample_rate = data.get("sampleRate", 44100)
    
    if not audio_data or len(audio_data) < 100:
        return jsonify({
            "status": "error",
            "message": "Not enough audio data for analysis"
        })
    
    try:
        # Normalize and filter the audio data
        normalized_data = normalize_audio(audio_data)
        filtered_data = butter_bandpass_filter(normalized_data, 300, 3400, sample_rate)
        
        # Calculate basic audio statistics
        rms = np.sqrt(np.mean(np.square(filtered_data)))
        peak = max(abs(min(filtered_data)), abs(max(filtered_data)))
        crest_factor = peak / (rms + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Calculate zero-crossing rate (useful for voice activity detection)
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(filtered_data))))
        zcr = zero_crossings / len(filtered_data)
        
        # Calculate spectral centroid (indicator of "brightness" of sound)
        if len(filtered_data) > 256:
            nperseg = min(512, len(filtered_data) // 4)
            f, pxx = signal.welch(filtered_data, fs=sample_rate, nperseg=nperseg)
            spectral_centroid = np.sum(f * pxx) / (np.sum(pxx) + 1e-10)
        else:
            spectral_centroid = 0
            
        # Generate analysis results
        results = {
            "status": "success",
            "audioStats": {
                "duration": len(audio_data) / sample_rate,
                "rmsLevel": float(rms),
                "peakLevel": float(peak),
                "crestFactor": float(crest_factor),
                "zeroCrossingRate": float(zcr),
                "spectralCentroid": float(spectral_centroid)
            },
            "audioQuality": {
                "signalLevel": "Good" if rms > 0.1 else "Low",
                "dynamicRange": "Good" if crest_factor > 4 else "Compressed",
                "clarity": "Clear" if spectral_centroid > 1000 else "Muddy"
            }
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error in analysis: {str(e)}"
        })


if __name__ == "__main__":
    # Create placeholder image on startup
    create_placeholder_waveform()
    app.run(debug=True)
