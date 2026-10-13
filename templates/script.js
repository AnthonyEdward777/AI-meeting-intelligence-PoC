async function analyze() {
    const fileInput = document.getElementById("audio");
    const loading = document.getElementById("loading");
    const results = document.getElementById("results");

    if (!fileInput.files.length) {
        alert("Please select an audio file first.");
        return;
    }

    loading.classList.remove("hidden");
    results.classList.add("hidden");

    try {
        const formData = new FormData();
        formData.append("audio", fileInput.files[0]);

        const res = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || "Analysis failed.");
        }

        // Display the meeting summary.
        document.getElementById("summary").textContent =
            data.summary || "No summary returned.";

        // Display the transcript.
        const transcriptDiv = document.getElementById("transcript");
        transcriptDiv.replaceChildren();

        const transcriptLines =
            data.speaker_transcript?.length
                ? data.speaker_transcript
                : (data.transcript || []);

        transcriptLines.forEach(line => {
            const card = document.createElement("div");
            card.className =
                "p-3 bg-gray-50 border rounded-lg shadow-sm";

            const speaker = document.createElement("strong");
            speaker.className = "text-blue-700";
            speaker.textContent = `${line.speaker || "UNKNOWN"}: `;

            const text = document.createElement("p");
            text.className = "text-gray-800 mt-1";
            text.textContent = line.text || "";

            card.appendChild(speaker);

            if (
                Number.isFinite(Number(line.start)) &&
                Number.isFinite(Number(line.end)) &&
                line.start !== undefined &&
                line.end !== undefined
            ) {
                const timing = document.createElement("p");
                timing.className = "text-sm text-gray-500 mt-1";
                timing.textContent =
                    `${Number(line.start).toFixed(2)}s – ` +
                    `${Number(line.end).toFixed(2)}s`;

                card.appendChild(timing);
            }

            card.appendChild(text);
            transcriptDiv.appendChild(card);
        });

        if (data.alignment_error) {
            const warning = document.createElement("p");
            warning.className = "text-amber-700 text-sm mt-3";
            warning.textContent =
                "Word alignment failed; showing the original transcript. " +
                data.alignment_error;
            transcriptDiv.appendChild(warning);
        }

        if (data.alignment_warning) {
            const warning = document.createElement("p");
            warning.className = "text-amber-700 text-sm mt-3";
            warning.textContent = data.alignment_warning;
            transcriptDiv.appendChild(warning);
        }

        // Display pyannote's detected speaker turns.
        const segmentsDiv = document.getElementById("speaker_segments");
        

        results.classList.remove("hidden");
    } catch (error) {
        alert("Error analyzing meeting: " + error.message);
        console.error(error);
    } finally {
        loading.classList.add("hidden");
    }
}