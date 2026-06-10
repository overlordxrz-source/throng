// Setup Chart.js defaults for dark theme
Chart.defaults.color = '#8b949e';
Chart.defaults.font.family = "'Outfit', sans-serif";
Chart.defaults.scale.grid.color = 'rgba(33, 38, 45, 0.8)';
Chart.defaults.scale.grid.borderColor = '#30363d';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(22, 27, 34, 0.9)';
Chart.defaults.plugins.tooltip.titleColor = '#e6edf3';
Chart.defaults.plugins.tooltip.bodyColor = '#e6edf3';
Chart.defaults.plugins.tooltip.borderColor = '#30363d';
Chart.defaults.plugins.tooltip.borderWidth = 1;

// Global Chart Instances
let popChart, fitChart, miChart, linChart;

function initCharts() {
    // 1. Population & Energy (Dual Axis)
    const ctxPop = document.getElementById('populationChart').getContext('2d');
    popChart = new Chart(ctxPop, {
        type: 'line',
        data: { labels: [], datasets: [
            {
                label: 'Population',
                yAxisID: 'y',
                data: [],
                borderColor: '#58a6ff',
                backgroundColor: 'rgba(88, 166, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 0
            },
            {
                label: 'Mean Energy',
                yAxisID: 'y1',
                data: [],
                borderColor: '#f0b429',
                borderWidth: 1.5,
                borderDash: [4, 4],
                fill: false,
                tension: 0.3,
                pointRadius: 0
            }
        ]},
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { ticks: { maxTicksLimit: 8 } },
                y: { type: 'linear', display: true, position: 'left', title: { display: true, text: 'Agents' } },
                y1: { type: 'linear', display: true, position: 'right', title: { display: true, text: 'Energy' }, grid: { drawOnChartArea: false } }
            },
            plugins: { legend: { position: 'top', align: 'start' } }
        }
    });

    // 2. Fitness
    const ctxFit = document.getElementById('fitnessChart').getContext('2d');
    fitChart = new Chart(ctxFit, {
        type: 'line',
        data: { labels: [], datasets: [
            {
                label: 'Blue Entropy',
                data: [],
                borderColor: '#3fb950',
                backgroundColor: 'rgba(63, 185, 80, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 0
            },
            {
                label: 'Red VQ Loss',
                data: [],
                borderColor: '#e3b341',
                borderWidth: 1.5,
                borderDash: [4, 4],
                fill: false,
                tension: 0.3,
                pointRadius: 0
            }
        ]},
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: { ticks: { maxTicksLimit: 8 } },
                y: { title: { display: true, text: 'Score' } }
            },
            plugins: { legend: { position: 'top', align: 'start' } }
        }
    });

    // 3. Mutual Information (Heatmap via Chart.js Matrix or Custom)
    // For simplicity without pulling in chartjs-chart-matrix plugin, we will use a custom canvas drawing for MI
    const miCanvas = document.getElementById('miChart');
    const miCtx = miCanvas.getContext('2d');
    // We handle resize and custom draw in updateMIHeatmap

    // 4. Lineage Timeline
    const ctxLin = document.getElementById('lineageChart').getContext('2d');
    linChart = new Chart(ctxLin, {
        type: 'bar',
        data: { labels: [], datasets: [
            {
                label: 'Alive',
                data: [], // will store [start, end] tuples
                backgroundColor: '#3fb950',
                barPercentage: 0.6
            },
            {
                label: 'Extinct',
                data: [],
                backgroundColor: '#f85149',
                barPercentage: 0.6
            }
        ]},
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: { title: { display: true, text: 'Simulation Step' } },
                y: { stacked: true }
            },
            plugins: { legend: { position: 'bottom', align: 'end' } }
        }
    });
}

function updateMIHeatmap(canvas, miData) {
    if (!miData || !miData.length) return;
    
    // Auto-resize canvas for sharp rendering
    const rect = canvas.parentNode.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const featNames = ["resource", "neighbors", "energy", "dist_red"];
    const rows = miData.length;
    const cols = featNames.length;
    
    // Find global max
    let maxVal = 0.01;
    for (let r=0; r<rows; r++) {
        for (let c=0; c<cols; c++) {
            if (miData[r][c] > maxVal) maxVal = miData[r][c];
        }
    }
    
    // Margins
    const marginLeft = 40;
    const marginBottom = 30;
    const cellW = (canvas.width - marginLeft) / cols;
    const cellH = (canvas.height - marginBottom) / rows;
    
    // Draw cells
    for (let r=0; r<rows; r++) {
        for (let c=0; c<cols; c++) {
            const val = miData[r][c];
            const intensity = Math.min(1.0, val / maxVal);
            
            // Magma-ish colormap approximation
            // low: black/purple #100b20, mid: orange #f85149, high: yellow #f0b429
            const rC = Math.floor(16 + intensity * 224);
            const gC = Math.floor(11 + intensity * 169);
            const bC = Math.floor(32 - intensity * 10);
            
            ctx.fillStyle = `rgb(${rC}, ${gC}, ${bC})`;
            ctx.fillRect(marginLeft + c*cellW, r*cellH, cellW-1, cellH-1);
        }
    }
    
    // Labels
    ctx.fillStyle = '#8b949e';
    ctx.font = '11px Outfit';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    
    // Row labels
    for (let r=0; r<rows; r++) {
        ctx.fillText(`d${r}`, marginLeft - 5, r*cellH + cellH/2);
    }
    
    // Col labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let c=0; c<cols; c++) {
        ctx.fillText(featNames[c], marginLeft + c*cellW + cellW/2, canvas.height - marginBottom + 5);
    }
}

function updateLineageData(currentStep, lineages) {
    if (!lineages || !lineages.length) return;
    
    const labels = [];
    const aliveData = [];
    const deadData = [];
    
    lineages.forEach((lin, i) => {
        labels.push(`L${lin.lineage_id !== undefined ? lin.lineage_id : i}`);
        const start = lin.birth_step || 0;
        const end = lin.last_seen_step || currentStep;
        
        if (lin.is_alive) {
            aliveData.push([start, end]);
            deadData.push(null);
        } else {
            aliveData.push(null);
            deadData.push([start, end]);
        }
    });
    
    linChart.data.labels = labels;
    linChart.data.datasets[0].data = aliveData;
    linChart.data.datasets[1].data = deadData;
    linChart.update();
}

async function fetchTelemetry() {
    try {
        const response = await fetch('/api/data');
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        
        document.getElementById('step-counter').textContent = `Step ${data.step.toLocaleString()}`;
        
        // Update Pop Chart
        popChart.data.labels = data.step_history;
        popChart.data.datasets[0].data = data.population_history;
        popChart.data.datasets[1].data = data.mean_energy_hist;
        popChart.update();
        
        // Update Fitness Chart
        fitChart.data.labels = data.step_history;
        fitChart.data.datasets[0].data = data.mean_fitness_hist;
        fitChart.data.datasets[1].data = data.max_fitness_hist;
        fitChart.update();
        
        // Update MI Heatmap
        if (data.mi_history && data.mi_history.length > 0) {
            const latestMi = data.mi_history[data.mi_history.length - 1].mi_matrix;
            updateMIHeatmap(document.getElementById('miChart'), latestMi);
        }
        
        // Update Lineage
        updateLineageData(data.step, data.top_lineages);
        
    } catch (error) {
        console.error('Error fetching telemetry:', error);
    }
}

// Initialize and start polling
window.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchTelemetry();
    setInterval(fetchTelemetry, 5000); // Poll every 5s
    
    // Handle window resize for custom canvas
    window.addEventListener('resize', () => {
        fetchTelemetry(); // Redraw custom canvas
    });
});
