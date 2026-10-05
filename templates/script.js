async function analyze() {
            // Show loading text, hide old results
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').classList.add('hidden');


            let fileInput = document.getElementById('audio');
            let formData = new FormData();
            formData.append('audio', fileInput.files[0]);

            let res = await fetch('/analyze', { method: 'POST', body: formData });
            let data = await res.json();

            document.getElementById('summary').innerText = data.summary;

            document.getElementById('transcript').innerHTML = ''; // clear old text
            data.transcript.forEach(line => {
                document.getElementById('transcript').innerHTML += `
                    <div class="p-3 bg-gray-50 border rounded-lg shadow-sm">
                        <strong class="text-blue-700">${line.speaker}:</strong>
                        <span class="text-gray-800">${line.text}</span>
                    </div>
                `;
            });

            // Hide loading text, show results
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('results').classList.remove('hidden');
        }