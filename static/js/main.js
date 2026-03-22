/* ==========================================================================
   main.js — UI interactions & Plotly chart rendering
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    initCharCounter();
    initFileUpload();
    initTopicSlider();
    initFormSubmit();
});

/* ---------------------------------------------------------------------------
   Character Counter
   --------------------------------------------------------------------------- */
function initCharCounter() {
    const textarea = document.getElementById('text-input');
    const counter  = document.getElementById('char-count');
    if (!textarea || !counter) return;

    textarea.addEventListener('input', () => {
        const len = textarea.value.length;
        counter.textContent = `${len.toLocaleString()} characters`;
    });
}

/* ---------------------------------------------------------------------------
   File Upload UX
   --------------------------------------------------------------------------- */
function initFileUpload() {
    const area      = document.getElementById('file-upload-area');
    const input     = document.getElementById('file-input');
    const selected  = document.getElementById('file-selected');
    const nameSpan  = document.getElementById('file-name');
    const removeBtn = document.getElementById('file-remove');
    const content   = area ? area.querySelector('.file-upload-content') : null;

    if (!area || !input) return;

    // Drag & drop visual feedback
    ['dragenter', 'dragover'].forEach(evt => {
        area.addEventListener(evt, e => {
            e.preventDefault();
            area.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(evt => {
        area.addEventListener(evt, e => {
            e.preventDefault();
            area.classList.remove('dragover');
        });
    });

    // Show selected file name
    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            nameSpan.textContent = input.files[0].name;
            selected.style.display = 'flex';
            if (content) content.style.display = 'none';
        }
    });

    // Remove selected file
    if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            input.value = '';
            selected.style.display = 'none';
            if (content) content.style.display = 'block';
        });
    }
}

/* ---------------------------------------------------------------------------
   Topic Slider
   --------------------------------------------------------------------------- */
function initTopicSlider() {
    const slider = document.getElementById('n-topics');
    const display = document.getElementById('n-topics-value');
    if (!slider || !display) return;

    slider.addEventListener('input', () => {
        display.textContent = slider.value;
    });
}

/* ---------------------------------------------------------------------------
   Form Submit — show loading state
   --------------------------------------------------------------------------- */
function initFormSubmit() {
    const form = document.getElementById('analysis-form');
    const btn  = document.getElementById('analyze-btn');
    if (!form || !btn) return;

    form.addEventListener('submit', () => {
        const btnText   = btn.querySelector('.btn-text');
        const btnLoader = btn.querySelector('.btn-loader');
        if (btnText)   btnText.style.display   = 'none';
        if (btnLoader) btnLoader.style.display  = 'inline-flex';
        btn.disabled = true;
        btn.style.opacity = '0.7';
    });
}

/* ---------------------------------------------------------------------------
   Plotly Chart Rendering
   --------------------------------------------------------------------------- */
const PLOTLY_LAYOUT_DEFAULTS = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font: {
        family: 'Inter, sans-serif',
        color:  '#94a3b8',
        size:   12,
    },
    margin: { t: 30, r: 20, b: 40, l: 50 },
    hoverlabel: {
        bgcolor: '#1e293b',
        bordercolor: '#334155',
        font: { color: '#f1f5f9', family: 'Inter, sans-serif' },
    },
};

const PLOTLY_CONFIG = {
    displayModeBar: false,
    responsive: true,
};

function renderCharts(data) {
    if (!data) return;

    renderSentimentDonut(data.sentiment_dist);
    renderCompoundBars(data.compound_bars);
    renderHeatmap(data.heatmap);
}

/* Sentiment Distribution — Donut Chart */
function renderSentimentDonut(dist) {
    const el = document.getElementById('sentiment-donut-chart');
    if (!el || !dist) return;

    const trace = {
        labels: dist.labels,
        values: dist.values,
        type: 'pie',
        hole: 0.55,
        marker: {
            colors: dist.colors,
            line: { color: '#0a0e17', width: 2 },
        },
        textinfo: 'label+percent',
        textfont: { color: '#f1f5f9', size: 13 },
        hoverinfo: 'label+value+percent',
        sort: false,
    };

    const layout = {
        ...PLOTLY_LAYOUT_DEFAULTS,
        showlegend: false,
        annotations: [{
            text: `${dist.values.reduce((a, b) => a + b, 0)}<br>docs`,
            showarrow: false,
            font: { size: 18, color: '#f1f5f9', family: 'Inter' },
            x: 0.5, y: 0.5,
        }],
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

/* Per-Topic Average Compound Score — Bar Chart */
function renderCompoundBars(bars) {
    const el = document.getElementById('compound-bar-chart');
    if (!el || !bars) return;

    const colors = bars.values.map(v =>
        v >= 0.05 ? '#00e676' : v <= -0.05 ? '#ff5252' : '#ffd740'
    );

    const trace = {
        x: bars.labels,
        y: bars.values,
        type: 'bar',
        marker: {
            color: colors,
            line: { color: 'rgba(255,255,255,0.1)', width: 1 },
            opacity: 0.85,
        },
        hovertemplate: '<b>%{x}</b><br>Avg Compound: %{y:.4f}<extra></extra>',
    };

    const layout = {
        ...PLOTLY_LAYOUT_DEFAULTS,
        xaxis: {
            tickfont: { size: 10 },
            tickangle: -30,
            gridcolor: 'rgba(255,255,255,0.03)',
        },
        yaxis: {
            title: 'Avg VADER Compound',
            gridcolor: 'rgba(255,255,255,0.05)',
            zeroline: true,
            zerolinecolor: 'rgba(255,255,255,0.15)',
        },
        bargap: 0.3,
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}

/* Topic × Sentiment Heatmap */
function renderHeatmap(hm) {
    const el = document.getElementById('heatmap-chart');
    if (!el || !hm) return;

    const trace = {
        x: hm.x,
        y: hm.y,
        z: hm.z,
        type: 'heatmap',
        colorscale: [
            [0,   '#0a0e17'],
            [0.25,'#312e81'],
            [0.5, '#6366f1'],
            [0.75,'#a855f7'],
            [1,   '#ec4899'],
        ],
        hovertemplate: '<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>',
        showscale: true,
        colorbar: {
            title: { text: '%', font: { color: '#94a3b8' } },
            tickfont: { color: '#94a3b8' },
            outlinewidth: 0,
        },
    };

    const layout = {
        ...PLOTLY_LAYOUT_DEFAULTS,
        xaxis: { side: 'top', tickfont: { size: 12 } },
        yaxis: { tickfont: { size: 10 }, autorange: 'reversed' },
        margin: { t: 50, r: 80, b: 20, l: 160 },
    };

    Plotly.newPlot(el, [trace], layout, PLOTLY_CONFIG);
}
