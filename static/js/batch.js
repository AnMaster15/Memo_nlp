document.getElementById('batchForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const form = e.target;
    const fileInput = document.getElementById('audioFiles');
    const batchStatusDiv = document.getElementById('batchStatus');
    const batchResultsDiv = document.getElementById('batchResults');
    
    if (fileInput.files.length === 0) {
        alert('Please select one or more WAV files to analyze');
        return;
    }
    
    const formData = new FormData();
    for (const file of fileInput.files) {
        formData.append('files', file);
    }
    
    try {
        // Show loading state
        form.querySelector('button').disabled = true;
        form.querySelector('button').innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        batchStatusDiv.style.display = 'block';
        
        const response = await fetch('/analyze-batch', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(await response.text());
        }
        
        const { batch_id } = await response.json();
        monitorBatchProgress(batch_id);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Error processing batch: ' + error.message);
    } finally {
        // Reset form
        form.querySelector('button').disabled = false;
        form.querySelector('button').textContent = 'Analyze Batch';
    }
});

async function monitorBatchProgress(batchId) {
    const progressBar = document.getElementById('progressBar');
    const statusMessage = document.getElementById('statusMessage');
    const batchResultsDiv = document.getElementById('batchResults');
    
    let attempts = 0;
    const maxAttempts = 30; // ~5 minutes with 10s intervals
    
    const checkInterval = setInterval(async () => {
        attempts++;
        
        try {
            const response = await fetch(`/batch-results/${batchId}`);
            if (!response.ok) {
                throw new Error(await response.text());
            }
            
            const data = await response.json();
            
            if (data.status === 'completed') {
                clearInterval(checkInterval);
                progressBar.style.width = '100%';
                progressBar.classList.remove('progress-bar-animated');
                statusMessage.textContent = 'Analysis complete!';
                displayBatchResults(batchId, data.results);
                batchResultsDiv.style.display = 'block';
            } else if (data.status === 'error') {
                clearInterval(checkInterval);
                statusMessage.textContent = 'Error processing batch';
                progressBar.style.width = '100%';
                progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
                progressBar.classList.add('bg-danger');
            } else {
                // Update progress (simple linear progress for demo)
                const progress = Math.min(90, attempts * 3); // Cap at 90% until complete
                progressBar.style.width = `${progress}%`;
                statusMessage.textContent = `Processing ${data.sample_count} samples...`;
            }
        } catch (error) {
            console.error('Error checking batch status:', error);
            if (attempts >= maxAttempts) {
                clearInterval(checkInterval);
                statusMessage.textContent = 'Batch processing timed out';
                progressBar.style.width = '100%';
                progressBar.classList.remove('progress-bar-striped', 'progress-bar-animated');
                progressBar.classList.add('bg-warning');
            }
        }
    }, 10000); // Check every 10 seconds
}

async function displayBatchResults(batchId, results) {
    // Set report link
    document.getElementById('reportLink').href = `/batch-report/${batchId}`;
    
    // Create risk distribution chart
    createRiskDistributionChart(results);
    
    // Create anomaly chart
    createAnomalyChart(results);
    
    // Populate insights
    const insightsList = document.getElementById('insightsList');
    insightsList.innerHTML = '';
    
    // Simple insights for demo - in a real app these would come from the API
    const highRiskCount = results.filter(r => r.risk_score >= 70).length;
    const anomalyCount = results.filter(r => r.is_anomaly).length;
    const avgRisk = results.reduce((sum, r) => sum + r.risk_score, 0) / results.length;
    
    const insights = [
        `${highRiskCount} samples (${Math.round(highRiskCount/results.length*100)}%) show high risk patterns`,
        `${anomalyCount} samples (${Math.round(anomalyCount/results.length*100)}%) were identified as anomalies`,
        `Average risk score across all samples: ${avgRisk.toFixed(1)}`
    ];
    
    insights.forEach(insight => {
        const li = document.createElement('li');
        li.className = 'list-group-item';
        li.textContent = insight;
        insightsList.appendChild(li);
    });
    
    // Populate samples table
    const samplesTable = document.getElementById('samplesTable');
    samplesTable.innerHTML = '';
    
    results.slice(0, 10).forEach(sample => { // Show first 10 for demo
        const row = document.createElement('tr');
        
        const sampleCell = document.createElement('td');
        sampleCell.textContent = sample.sample_id || 'Sample';
        
        const riskCell = document.createElement('td');
        riskCell.textContent = sample.risk_score.toFixed(1);
        
        const anomalyCell = document.createElement('td');
        anomalyCell.textContent = sample.is_anomaly ? 'Yes' : 'No';
        
        const detailsCell = document.createElement('td');
        const detailsBtn = document.createElement('button');
        detailsBtn.className = 'btn btn-sm btn-outline-primary';
        detailsBtn.textContent = 'View';
        detailsBtn.addEventListener('click', () => {
            // In a real app, this would show more details
            alert(`Sample ${sample.sample_id}\nRisk: ${sample.risk_score.toFixed(1)}\nAnomaly: ${sample.is_anomaly}`);
        });
        detailsCell.appendChild(detailsBtn);
        
        row.appendChild(sampleCell);
        row.appendChild(riskCell);
        row.appendChild(anomalyCell);
        row.appendChild(detailsCell);
        samplesTable.appendChild(row);
    });
}

function createRiskDistributionChart(results) {
    const ctx = document.getElementById('riskDistributionChart').getContext('2d');
    
    // Prepare data for histogram
    const riskScores = results.map(r => r.risk_score);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0-20', '21-40', '41-60', '61-80', '81-100'],
            datasets: [{
                label: 'Number of Samples',
                data: [
                    riskScores.filter(s => s <= 20).length,
                    riskScores.filter(s => s > 20 && s <= 40).length,
                    riskScores.filter(s => s > 40 && s <= 60).length,
                    riskScores.filter(s => s > 60 && s <= 80).length,
                    riskScores.filter(s => s > 80).length
                ],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.7)',
                    'rgba(40, 167, 69, 0.7)',
                    'rgba(255, 193, 7, 0.7)',
                    'rgba(220, 53, 69, 0.7)',
                    'rgba(220, 53, 69, 0.7)'
                ],
                borderColor: [
                    'rgba(40, 167, 69, 1)',
                    'rgba(40, 167, 69, 1)',
                    'rgba(255, 193, 7, 1)',
                    'rgba(220, 53, 69, 1)',
                    'rgba(220, 53, 69, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Samples'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Risk Score Range'
                    }
                }
            }
        }
    });
}

function createAnomalyChart(results) {
    const ctx = document.getElementById('anomalyChart').getContext('2d');
    
    // Prepare data for pie chart
    const anomalyCount = results.filter(r => r.is_anomaly).length;
    const normalCount = results.length - anomalyCount;
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal Samples', 'Anomalies'],
            datasets: [{
                data: [normalCount, anomalyCount],
                backgroundColor: [
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 99, 132, 0.7)'
                ],
                borderColor: [
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = Math.round((value / total) * 100);
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}