document.getElementById('analysisForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const form = e.target;
    const fileInput = document.getElementById('audioFile');
    const resultsDiv = document.getElementById('results');
    
    if (fileInput.files.length === 0) {
        alert('Please select a WAV file to analyze');
        return;
    }
    
    const formData = new FormData();
    formData.append('audio_file', fileInput.files[0]);
    
    try {
        // Show loading state
        form.querySelector('button').disabled = true;
        form.querySelector('button').innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
        
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(await response.text());
        }
        
        const data = await response.json();
        displayResults(data);
        resultsDiv.style.display = 'block';
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing file: ' + error.message);
    } finally {
        // Reset form
        form.querySelector('button').disabled = false;
        form.querySelector('button').textContent = 'Analyze';
    }
});

function displayResults(data) {
    // Update risk score
    const riskScore = document.getElementById('riskScore');
    riskScore.textContent = Math.round(data.risk_score);
    
    // Set risk circle color based on score
    const riskCircle = document.querySelector('.risk-circle');
    if (data.risk_score >= 70) {
        riskCircle.style.background = 'linear-gradient(135deg, #dc3545, #ff6b6b)';
    } else if (data.risk_score >= 40) {
        riskCircle.style.background = 'linear-gradient(135deg, #ffc107, #ffdd59)';
    } else {
        riskCircle.style.background = 'linear-gradient(135deg, #28a745, #51cf66)';
    }
    
    // Update risk description
    const riskDescription = document.getElementById('riskDescription');
    if (data.risk_score >= 70) {
        riskDescription.textContent = 'High risk of cognitive markers detected. Consider professional evaluation.';
    } else if (data.risk_score >= 40) {
        riskDescription.textContent = 'Moderate risk detected. Monitor for changes over time.';
    } else {
        riskDescription.textContent = 'Low risk detected. No significant cognitive markers found.';
    }
    
    // Update transcription
    document.getElementById('transcription').textContent = data.transcription || 'No transcription available';
    
    // Update metrics table
    const metricsTable = document.getElementById('metricsTable');
    metricsTable.innerHTML = '';
    
    for (const [metric, value] of Object.entries(data.key_indicators)) {
        const row = document.createElement('tr');
        
        const metricCell = document.createElement('td');
        metricCell.textContent = metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        
        const valueCell = document.createElement('td');
        valueCell.textContent = typeof value === 'number' ? value.toFixed(2) : value;
        
        row.appendChild(metricCell);
        row.appendChild(valueCell);
        metricsTable.appendChild(row);
    }
    
    // Create indicators chart
    createIndicatorsChart(data.key_indicators);
}

function createIndicatorsChart(indicators) {
    const ctx = document.getElementById('indicatorsChart').getContext('2d');
    
    // Prepare data for radar chart
    const labels = Object.keys(indicators).map(key => 
        key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    );
    const values = Object.values(indicators);
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Speech Metrics',
                data: values,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    angleLines: {
                        display: true
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            }
        }
    });
}