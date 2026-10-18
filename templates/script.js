// Initialize Lucide Icons
lucide.createIcons();

// --- View Router Logic ---
const pageTitles = {
    'dashboard': 'Dashboard',
    'analyze': 'Analyze Meeting',
    'history': 'Meeting History',
    'settings': 'Settings',
    'loading': 'Analyzing Meeting...',
    'results': 'Intelligence Report'
};

function switchView(viewId) {
    document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
    const target = document.getElementById(`view-${viewId}`);
    if (target) target.classList.add('active');
    
    document.getElementById('page-title').textContent = pageTitles[viewId] || 'Majlis';

    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('bg-white/10', 'text-white');
        btn.classList.add('text-slate-300');
    });
    
    const activeBtn = document.getElementById(`nav-${viewId}`);
    if (activeBtn) {
        activeBtn.classList.add('bg-white/10', 'text-white');
        activeBtn.classList.remove('text-slate-300');
    }
}

// --- Drag and Drop File Handling ---
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('audio');
const filePreview = document.getElementById('file-preview');
const uploadPrompt = document.getElementById('upload-prompt');
const btnAnalyze = document.getElementById('btn-analyze');
let selectedFile = null;

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.add('border-majlisTeal', 'bg-slate-50');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => {
        dropZone.classList.remove('border-majlisTeal', 'bg-slate-50');
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
        fileInput.files = files; 
        handleFileSelect({ target: fileInput });
    }
}, false);

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('file-size').textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
        
        uploadPrompt.classList.add('hidden');
        filePreview.classList.remove('hidden');
        filePreview.classList.add('flex');
        
        btnAnalyze.disabled = false;
    }
}

function clearFile(event) {
    event.preventDefault(); 
    event.stopPropagation();
    
    fileInput.value = '';
    selectedFile = null;
    
    filePreview.classList.add('hidden');
    filePreview.classList.remove('flex');
    uploadPrompt.classList.remove('hidden');
    
    btnAnalyze.disabled = true;
}

// --- Analysis Engine ---

const speakerColors = [
    'text-teal-600 bg-teal-50 border-teal-100',
    'text-indigo-600 bg-indigo-50 border-indigo-100',
    'text-rose-600 bg-rose-50 border-rose-100',
    'text-amber-600 bg-amber-50 border-amber-100',
    'text-sky-600 bg-sky-50 border-sky-100'
];

function formatSpeakerName(rawName) {
    if (!rawName || rawName === "UNKNOWN") return { name: "Unknown Speaker", index: 0 };
    const match = rawName.match(/SPEAKER_(\d+)/i);
    if (match) {
        const num = parseInt(match[1], 10);
        return { name: `Speaker ${num + 1}`, index: num };
    }
    return { name: rawName, index: 0 };
}

let statusInterval;

function startLoadingSimulation() {
    switchView('loading');
    
    const statuses = [
        "Aligning audio vectors...",
        "Running Cohere transcription engine...",
        "Identifying distinct speakers via Pyannote...",
        "Executing Qwen LLM for executive summarization...",
        "Finalizing intelligence report..."
    ];
    let i = 0;
    
    document.getElementById('step-transcribe').classList.replace('text-slate-400', 'text-majlisTeal');
    
    statusInterval = setInterval(() => {
        document.getElementById('loading-status').textContent = statuses[i % statuses.length];
        
        if (i === 3) {
            const stepSum = document.getElementById('step-summary');
            stepSum.classList.remove('opacity-50');
            stepSum.innerHTML = '<i data-lucide="loader-2" class="w-4 h-4 mr-2 animate-spin"></i> Generating Executive Summary';
            lucide.createIcons();
        }
        i++;
    }, 4000);
}

function stopLoadingSimulation() {
    clearInterval(statusInterval);
    document.getElementById('step-transcribe').classList.replace('text-majlisTeal', 'text-slate-400');
    const stepSum = document.getElementById('step-summary');
    stepSum.classList.add('opacity-50');
    stepSum.innerHTML = '<i data-lucide="circle" class="w-4 h-4 mr-2"></i> Generating Executive Summary';
}

async function analyze() {
    if (!selectedFile) {
        showToast("Please select an audio file first.", true);
        return;
    }

    startLoadingSimulation();

    try {
        const formData = new FormData();
        formData.append("audio", selectedFile);
        formData.append("num_speakers", document.getElementById("num_speakers").value);

        const res = await fetch("/analyze", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || "Analysis failed.");
        }

        renderResults(data);
        stopLoadingSimulation();
        switchView('results');
        showToast("Analysis complete");

    } catch (error) {
        stopLoadingSimulation();
        switchView('analyze'); 
        showToast(error.message, true);
        console.error(error);
    }
}

function renderResults(data) {
    const summaryEl = document.getElementById("summary");
    summaryEl.innerHTML = (data.summary || "No summary returned.")
        .split('\n')
        .map(p => p.trim() ? `<p class="mb-4">${p}</p>` : '')
        .join('');

    const transcriptDiv = document.getElementById("transcript");
    transcriptDiv.replaceChildren();

    const transcriptLines = data.speaker_transcript?.length 
        ? data.speaker_transcript 
        : (data.transcript || []);

    transcriptLines.forEach(line => {
        const speakerInfo = formatSpeakerName(line.speaker);
        const colorClasses = speakerColors[speakerInfo.index % speakerColors.length];

        const card = document.createElement("div");
        card.className = "transcript-card group flex flex-col hover:bg-white p-4 rounded-xl transition-colors duration-200";

        const header = document.createElement("div");
        header.className = "flex items-center justify-between mb-2";

        const speakerBadge = document.createElement("div");
        speakerBadge.className = `speaker-badge px-3 py-1 rounded-full text-sm font-semibold border font-sans tracking-wide ${colorClasses}`;
        speakerBadge.textContent = speakerInfo.name;
        speakerBadge.setAttribute('data-original-text', speakerInfo.name);

        const timing = document.createElement("span");
        timing.className = "text-xs font-sans text-slate-400 group-hover:text-slate-500 transition-colors";
        if (Number.isFinite(Number(line.start)) && Number.isFinite(Number(line.end))) {
            const formatTime = (secs) => {
                const m = Math.floor(secs / 60);
                const s = Math.floor(secs % 60).toString().padStart(2, '0');
                return `${m}:${s}`;
            };
            timing.textContent = `${formatTime(line.start)} – ${formatTime(line.end)}`;
        }

        header.appendChild(speakerBadge);
        header.appendChild(timing);

        const text = document.createElement("p");
        text.className = "transcript-text text-slate-700 leading-relaxed pl-2";
        text.textContent = line.text || "";
        text.setAttribute('data-original-text', line.text || "");

        card.appendChild(header);
        card.appendChild(text);
        transcriptDiv.appendChild(card);
    });

    if (data.alignment_error) {
        showToast("Word alignment failed; showing raw transcript.", true);
    }
}

// --- Utilities ---
function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    const msgEl = document.getElementById('toast-msg');
    
    const existingIcon = toast.querySelector('svg') || toast.querySelector('i');
    if (existingIcon) {
        existingIcon.remove();
    }
    
    const newIcon = document.createElement('i');
    newIcon.setAttribute('data-lucide', isError ? 'alert-circle' : 'check-circle');
    newIcon.className = isError ? 'w-5 h-5 text-red-400 mr-3' : 'w-5 h-5 text-green-400 mr-3';
    
    toast.insertBefore(newIcon, msgEl);
    
    msgEl.textContent = message;
    lucide.createIcons();
    
    toast.classList.remove('translate-y-20', 'opacity-0');
    
    setTimeout(() => {
        toast.classList.add('translate-y-20', 'opacity-0');
    }, 4000);
}

function copyText(elementId) {
    const text = document.getElementById(elementId).innerText;
    navigator.clipboard.writeText(text).then(() => {
        showToast("Copied to clipboard");
    });
}

// --- Smart Search Filter ---
function filterTranscript() {
    const rawQuery = document.getElementById('transcript-search').value.toLowerCase().trim();
    // Split the query into individual words (Google-style smart search)
    const searchTerms = rawQuery.split(/\s+/).filter(term => term.length > 0);
    
    const transcriptDiv = document.getElementById('transcript');
    const cards = transcriptDiv.getElementsByClassName('transcript-card');

    for (let i = 0; i < cards.length; i++) {
        const card = cards[i];
        const speakerBadge = card.querySelector('.speaker-badge');
        const textEl = card.querySelector('.transcript-text');

        // Grab the original, unmodified text we saved earlier
        const originalSpeaker = speakerBadge.getAttribute('data-original-text');
        const originalText = textEl.getAttribute('data-original-text');
        
        const fullCardText = (originalSpeaker + " " + originalText).toLowerCase();

        // Check if *all* typed words appear anywhere in the card
        const matches = searchTerms.every(term => fullCardText.includes(term));

        if (matches || searchTerms.length === 0) {
            card.style.display = ''; // Show card
            
            if (searchTerms.length > 0) {
                // Highlight the matching words in yellow
                speakerBadge.innerHTML = highlightText(originalSpeaker, searchTerms);
                textEl.innerHTML = highlightText(originalText, searchTerms);
            } else {
                // Reset to normal if search is empty
                speakerBadge.textContent = originalSpeaker;
                textEl.textContent = originalText;
            }
        } else {
            card.style.display = 'none'; // Hide card
        }
    }
}

// Helper to wrap matching words in a highlighted <mark> tag
function highlightText(text, terms) {
    if (!text) return "";
    let highlighted = text;
    // Escape regex characters so special characters don't break the search
    const escapedTerms = terms.map(t => t.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&'));
    const regex = new RegExp(`(${escapedTerms.join('|')})`, 'gi');
    return highlighted.replace(regex, '<mark class="bg-yellow-200 text-majlisNavy rounded-sm px-1">$1</mark>');
}