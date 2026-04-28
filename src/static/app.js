// API Configuration
const API_BASE_URL = '/api/v1';
let currentAnalysisId = null;
let progressInterval = null;

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadSystemStatus();
    loadRecentAnalyses();
    loadAvailableModels();
    setInterval(loadSystemStatus, 5000); // Refresh status every 5 seconds
    setInterval(loadRecentAnalyses, 10000); // Refresh recent analyses every 10 seconds
});

// Event Listeners
function setupEventListeners() {
    document.getElementById('analysis-form').addEventListener('submit', handleAnalysisSubmit);
    document.getElementById('code').addEventListener('input', updateCharCount);
}

// Load available models from API
async function loadAvailableModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/models`);
        const data = await response.json();

        const modelSelect = document.getElementById('model');
        const modelInfo = document.getElementById('model-info');

        if (data.models && data.models.length > 0) {
            // Clear existing options except the first one (auto-select)
            modelSelect.innerHTML = '<option value="">Auto-select (agent decides)</option>';

            // Add available models
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });

            // Update info text
            if (data.agentic_selection) {
                modelInfo.textContent = `✓ ${data.count} models available. ${data.agentic_selection}`;
            } else {
                modelInfo.textContent = `✓ ${data.count} models available for inference`;
            }
        } else {
            modelInfo.textContent = 'No models available. Make sure Ollama is running.';
        }
    } catch (error) {
        console.error('Error loading models:', error);
        document.getElementById('model-info').textContent = 'Could not load available models. Make sure Ollama is running.';
    }
}

// Update character count
function updateCharCount() {
    const code = document.getElementById('code').value;
    document.getElementById('char-count').textContent = `${code.length} characters`;
}

// Handle form submission
async function handleAnalysisSubmit(e) {
    e.preventDefault();

    const language = document.getElementById('language').value;
    const code = document.getElementById('code').value;
    const fileName = document.getElementById('file-name').value || 'code_snippet';
    const includeRag = document.getElementById('include-rag').checked;
    const timeout = parseInt(document.getElementById('timeout').value);
    const model = document.getElementById('model').value || null; // null = auto-select

    if (!language) {
        showToast('Error', 'Please select a programming language', 'danger');
        return;
    }

    if (code.trim().length < 10) {
        showToast('Error', 'Code snippet must be at least 10 characters', 'danger');
        return;
    }

    // Disable button
    const submitBtn = document.getElementById('submit-btn');
    const originalText = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="bi bi-hourglass-split spinner-animation"></i> Analyzing...';

    try {
        const requestBody = {
            code,
            language,
            file_name: fileName,
            include_rag: includeRag,
            timeout
        };

        // Add model if specified (omit if null for auto-selection)
        if (model) {
            requestBody.model = model;
        }

        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (response.status === 202) {
            const data = await response.json();
            currentAnalysisId = data.analysis_id;

            // Clear form
            document.getElementById('analysis-form').reset();
            document.getElementById('char-count').textContent = '0 characters';

            // Show progress section
            document.getElementById('no-results').style.display = 'none';
            document.getElementById('progress-section').style.display = 'block';
            document.getElementById('results-display').style.display = 'none';
            document.getElementById('error-display').style.display = 'none';

            // Reset agent graph to initial state
            resetAgentGraph();
            updateAgentGraph('queued');

            // Switch to results tab
            const resultsTab = new bootstrap.Tab(document.getElementById('results-tab'));
            resultsTab.show();

            const modelInfo = model ? `using ${model}` : 'with agentic model selection';
            showToast('Success', `Analysis submitted successfully ${modelInfo}!`, 'success');

            // Start polling for progress
            startProgressPolling();
        } else {
            const error = await response.json();
            showToast('Error', error.detail || 'Failed to submit analysis', 'danger');
        }
    } catch (error) {
        showToast('Error', `Network error: ${error.message}`, 'danger');
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerHTML = originalText;
    }
}

// Poll for progress
function startProgressPolling() {
    if (progressInterval) clearInterval(progressInterval);

    // Reset agent graph to clean state
    resetAgentGraph();
    // Immediately show the first node as active
    updateAgentGraph('queued');

    progressInterval = setInterval(async () => {
        if (!currentAnalysisId) {
            clearInterval(progressInterval);
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/progress/${currentAnalysisId}`);
            const data = await response.json();

            updateProgressBar(data.percentage, data.status);

            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(progressInterval);

                // Fetch final results
                if (data.status === 'completed') {
                    fetchAnalysisResults(currentAnalysisId);
                } else {
                    showError('Analysis failed. Please try again.');
                }
            }
        } catch (error) {
            console.error('Error fetching progress:', error);
        }
    }, 500); // Poll every 500ms for smoother transitions
}

// Update progress bar
function updateProgressBar(percentage, status) {
    const progressBar = document.getElementById('progress-bar');
    const progressPercent = document.getElementById('progress-percent');
    const progressStep = document.getElementById('progress-step');

    progressBar.style.width = percentage + '%';
    progressPercent.textContent = percentage + '%';

    const steps = {
        'queued': 'Queued for processing...',
        'processing': 'Processing...',
        'parsing': 'Parsing code...',
        'rag_retrieval': 'Retrieving similar examples...',
        'inference': 'Running LLM inference...',
        'validation': 'Validating results...',
        'completed': 'Analysis complete!',
        'failed': 'Analysis failed'
    };

    progressStep.textContent = steps[status] || status;
    updateAgentGraph(status);
}

// ─── Agent Transition Graph ──────────────────────────────────────────────────
// Maps API status strings to the ordered list of graph node ids.
const AGENT_NODE_ORDER = [
    'gnode-start',
    'gnode-parse',
    'gnode-model',
    'gnode-chunk',
    'gnode-rag',
    'gnode-detect',
    'gnode-validate',
    'gnode-aggregate',
    'gnode-end',
];

// Maps actual API status values to active node IDs
// These statuses are returned from /api/v1/progress/{analysis_id}
const STATUS_TO_ACTIVE_NODE = {
    'queued':        'gnode-start',
    'processing':    'gnode-parse',      // Initial processing state
    'parsing':       'gnode-parse',
    'rag_retrieval': 'gnode-rag',
    'inference':     'gnode-detect',
    'validation':    'gnode-validate',
    'completed':     'gnode-end',
    'failed':        null,
};

function updateAgentGraph(status) {
    const isFailed = status === 'failed';
    const activeId = STATUS_TO_ACTIVE_NODE[status];
    const activeIdx = AGENT_NODE_ORDER.indexOf(activeId);

    AGENT_NODE_ORDER.forEach((id, idx) => {
        const el = document.getElementById(id);
        if (!el) return;

        el.classList.remove('state-pending', 'state-active', 'state-done', 'state-error');

        if (isFailed) {
            // Mark all nodes up to and including current as error, rest pending
            if (activeIdx < 0 || idx <= activeIdx) {
                el.classList.add('state-error');
            } else {
                el.classList.add('state-pending');
            }
        } else if (activeIdx < 0) {
            // Unknown status – leave pending
            el.classList.add('state-pending');
        } else if (idx < activeIdx) {
            // Completed nodes
            el.classList.add('state-done');
        } else if (idx === activeIdx) {
            // Currently active node – shows pulsing animation
            el.classList.add('state-active');
        } else if (idx === activeIdx + 1) {
            // Next node – show as pending but slightly highlighted
            el.classList.add('state-pending');
        } else {
            // Future nodes
            el.classList.add('state-pending');
        }
    });
}

function resetAgentGraph() {
    AGENT_NODE_ORDER.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.remove('state-pending', 'state-active', 'state-done', 'state-error');
        }
    });
}
// ────────────────────────────────────────────────────────────────────────────

// Fetch analysis results
async function fetchAnalysisResults(analysisId) {
    try {
        const response = await fetch(`${API_BASE_URL}/results/${analysisId}`);

        if (response.ok) {
            const result = await response.json();
            displayResults(result);
        } else if (response.status === 404) {
            showError('Analysis not found');
        } else {
            showError('Failed to fetch results');
        }
    } catch (error) {
        showError(`Error fetching results: ${error.message}`);
    }
}

// Display results
function displayResults(result) {
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('error-display').style.display = 'none';
    document.getElementById('results-display').style.display = 'block';

    // Update metrics
    // H3: Defensively coerce findings to an array and drop malformed entries
    // so a single null/undefined field from the LLM can't crash the UI.
    const rawFindings = Array.isArray(result.findings) ? result.findings : [];
    const findings = rawFindings.filter(
        (f) => f && (f.smell_type || f.type) && (f.location || f.line || f.line_number)
    );
    document.getElementById('findings-count').textContent = findings.length;

    const maxSeverity = findings.length > 0
        ? findings.reduce((max, f) => {
            const severity = (f.severity || '').toLowerCase();
            const severityValue = { 'critical': 4, 'high': 3, 'medium': 2, 'low': 1 }[severity] || 0;
            return severityValue > ({ 'critical': 4, 'high': 3, 'medium': 2, 'low': 1 }[max] || 0) ? severity : max;
        }, (findings[0].severity || '').toLowerCase())
        : 'N/A';

    document.getElementById('max-severity').textContent = maxSeverity.charAt(0).toUpperCase() + maxSeverity.slice(1);

    const duration = result.analysis_time_ms ? result.analysis_time_ms / 1000 : 0;
    document.getElementById('analysis-time').textContent = duration.toFixed(2) + 's';

    // Display AI metrics (Model used, F1 score, etc.)
    const modelUsed = result.model_used || 'Unknown';

    // Helper: format metric value, showing N/A for null/undefined
    const fmtMetric = (val) => {
        if (val === null || val === undefined) return 'N/A';
        const num = parseFloat(val);
        return isNaN(num) ? 'N/A' : num.toFixed(3);
    };

    const f1Score = fmtMetric(result.f1_score);
    const precision = fmtMetric(result.precision);
    const recall = fmtMetric(result.recall);
    const hasGroundTruth = result.ground_truth_count && result.ground_truth_count > 0;

    document.getElementById('model-used').textContent = modelUsed;

    // F1/Confidence score element was removed from the HTML; skip if absent
    const f1Element = document.getElementById('f1-score');
    if (f1Element) {
        f1Element.textContent = f1Score;
        if (f1Score === 'N/A') {
            f1Element.title = 'No ground truth available for this snippet. F1 requires benchmark data.';
            f1Element.style.opacity = '0.6';
        } else {
            f1Element.title = '';
            f1Element.style.opacity = '1';
        }
    }

    // Precision / Recall element was removed from the HTML; skip if absent
    const prElement = document.getElementById('precision-recall');
    if (prElement) {
        if (precision !== 'N/A' && recall !== 'N/A') {
            prElement.textContent = `${precision} / ${recall}`;
            prElement.title = 'Precision / Recall against ground truth';
            prElement.style.opacity = '1';
        } else if (precision !== 'N/A') {
            prElement.textContent = `${precision} (conf)`;
            prElement.title = 'Average confidence score (no ground truth available)';
            prElement.style.opacity = '0.85';
        } else {
            prElement.textContent = 'N/A';
            prElement.title = 'No findings detected or no ground truth available';
            prElement.style.opacity = '0.6';
        }
    }

    // Display model reasoning if available
    // L4: Always show reasoning when the backend returns any non-empty string,
    // whether the model was auto-picked by the agent OR user-specified. The
    // backend populates "User-specified model: <name>" for manual picks, so
    // hiding that row made the UI silently inconsistent across the two paths.
    const reasoning = typeof result.model_reasoning === 'string' ? result.model_reasoning.trim() : '';
    const reasoningRow = document.getElementById('model-reasoning-row');
    const reasoningContent = document.getElementById('model-reasoning-content');
    if (reasoning) {
        reasoningRow.style.display = 'block';
        reasoningContent.textContent = reasoning;
        // Adjust the tag so it's accurate for both auto and manual selection.
        const tagEl = reasoningRow.querySelector('.section-tag');
        if (tagEl) {
            const isUserPicked = /^user[- ]specified/i.test(reasoning);
            tagEl.textContent = isUserPicked ? 'User Selection' : 'Agentic Decision';
        }
    } else {
        reasoningRow.style.display = 'none';
    }

    // Display findings
    const findingsList = document.getElementById('findings-list');
    findingsList.innerHTML = '';

    if (findings.length === 0) {
        findingsList.innerHTML = '<p class="text-muted text-center py-3">No code smells detected! ✓</p>';
    } else {
        findings.forEach(finding => {
            const item = createFindingItem(finding);
            findingsList.appendChild(item);
        });
    }

    // Display metrics
    const metrics = result.metrics || {};
    const metricsList = document.getElementById('metrics-list');
    metricsList.innerHTML = '';

    Object.entries(metrics).forEach(([key, value]) => {
        const col = document.createElement('div');
        col.className = 'col-md-4 mb-2';
        col.innerHTML = `
            <div class="metric-card">
                <div class="metric-label">${key.replace(/_/g, ' ')}</div>
                <div class="metric-value">${formatMetricValue(value)}</div>
            </div>
        `;
        metricsList.appendChild(col);
    });

    showToast('Success', 'Analysis completed!', 'success');
}

// Create finding item element
function createFindingItem(finding) {
    const severityKey = (finding.severity || 'low').toLowerCase();
    const locationText = finding.location !== null && typeof finding.location === 'object'
        ? (finding.location.line ? `Line ${finding.location.line}` : 'Unknown')
        : finding.location;
    const title = finding.smell_type || finding.name || 'Finding';
    const explanation = finding.explanation || finding.description || '';
    const suggestion = finding.suggested_refactoring || finding.refactoring || finding.suggestion || '';

    const item = document.createElement('div');
    item.className = `finding-item ${severityKey}`;

    const severityColors = {
        'critical': { bg: '#fecaca', text: '#7f1d1d', label: 'CRITICAL' },
        'high': { bg: '#fee2e2', text: '#991b1b', label: 'HIGH' },
        'medium': { bg: '#fef3c7', text: '#92400e', label: 'MEDIUM' },
        'low': { bg: '#cffafe', text: '#164e63', label: 'LOW' }
    };

    const severity = severityColors[severityKey] || severityColors['low'];

    item.innerHTML = `
        <div class="finding-header">
            <div class="finding-title">${escapeHtml(title)}</div>
            <span class="finding-severity ${severityKey}" style="background-color: ${severity.bg}; color: ${severity.text};">
                ${severity.label}
            </span>
        </div>
        ${locationText ? `<div class="finding-location"><strong>Location:</strong> ${escapeHtml(locationText)}</div>` : ''}
        ${explanation ? `<div class="finding-explanation">${escapeHtml(explanation)}</div>` : ''}
        ${suggestion ? `<div class="finding-suggestion"><strong>💡 Suggestion:</strong> ${escapeHtml(suggestion)}</div>` : ''}
    `;

    return item;
}

// Format metric values
function formatMetricValue(value) {
    if (typeof value === 'number') {
        return value.toFixed(2);
    }
    return value;
}

// Show error message
function showError(message) {
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('results-display').style.display = 'none';
    document.getElementById('error-display').style.display = 'block';
    document.getElementById('error-message').textContent = message;

    showToast('Error', message, 'danger');
}

// Load system status
async function loadSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        if (response.ok) {
            const status = await response.json();

            // Update uptime
            const uptime = formatUptime(status.uptime_seconds);
            document.getElementById('uptime').textContent = uptime;

            // Update analysis counts
            document.getElementById('active-analyses').textContent = status.active_analyses || 0;
            document.getElementById('completed-analyses').textContent = status.completed_analyses || 0;

            // Update cache size
            const cacheSizeMB = (status.cache_size_bytes / 1024 / 1024).toFixed(2);
            document.getElementById('cache-size').textContent = cacheSizeMB + ' MB';

            // Update health badges
            updateHealthBadge(status.health_checks);
        }
    } catch (error) {
        console.error('Error loading system status:', error);
    }
}

// Update health badges
function updateHealthBadge(healthChecks) {
    if (!healthChecks) return;

    const serviceMap = {
        'ollama': 'service-ollama',
        'chromadb': 'service-chromadb',
        'database': 'service-database'
    };

    Object.entries(healthChecks).forEach(([service, isHealthy]) => {
        const element = document.getElementById(serviceMap[service]);
        if (element) {
            element.className = `service-badge ${isHealthy ? 'bg-success' : 'bg-danger'}`;
        }
    });

    // Update main health badge
    const allHealthy = Object.values(healthChecks).every(h => h);
    const badge = document.getElementById('health-badge');
    badge.className = `badge status-pill ${allHealthy ? 'bg-success' : 'bg-warning'}`;
    badge.innerHTML = `<i class="bi bi-circle-fill"></i> ${allHealthy ? 'Healthy' : 'Degraded'}`;
}

// Load recent analyses
async function loadRecentAnalyses() {
    try {
        const response = await fetch(`${API_BASE_URL}/analyses/active`);
        if (response.ok) {
            const analyses = await response.json();
            updateRecentAnalysesTable(analyses);
        }
    } catch (error) {
        console.error('Error loading recent analyses:', error);
    }
}

// Update recent analyses table
function updateRecentAnalysesTable(analyses) {
    const tbody = document.getElementById('analyses-table-body');

    if (!analyses || analyses.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-muted text-center">No analyses yet</td></tr>';
        return;
    }

    tbody.innerHTML = '';
    analyses.slice(0, 10).forEach(analysis => {
        const row = document.createElement('tr');
        const createdAt = new Date(analysis.created_at).toLocaleTimeString();
        const statusBadge = getStatusBadge(analysis.status);

        row.innerHTML = `
            <td><small>${analysis.analysis_id.substring(0, 8)}...</small></td>
            <td>${getLanguagePill(analysis.language)}</td>
            <td>${statusBadge}</td>
            <td><small>${createdAt}</small></td>
        `;

        tbody.appendChild(row);
    });
}

// Get status badge
function getStatusBadge(status) {
    const normalizedStatus = status || 'queued';
    return `<span class="run-status run-status--${normalizedStatus}">${normalizedStatus.charAt(0).toUpperCase() + normalizedStatus.slice(1)}</span>`;
}

function getLanguagePill(language) {
    return `<span class="language-pill">${escapeHtml(language || 'unknown')}</span>`;
}

// Show toast notification
function showToast(title, message, type = 'info') {
    const toast = document.getElementById('toast');
    const toastTitle = document.getElementById('toast-title');
    const toastMessage = document.getElementById('toast-message');

    toastTitle.textContent = title;
    toastMessage.textContent = message;
    toast.dataset.tone = type;

    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
}

// Format uptime
function formatUptime(seconds) {
    if (!seconds) return '0s';

    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${secs}s`;
    return `${secs}s`;
}

// Escape HTML special characters
function escapeHtml(text) {
    if (!text) return '';
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Handle keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl+Enter or Cmd+Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const form = document.getElementById('analysis-form');
        if (document.activeElement.closest('form') === form) {
            form.dispatchEvent(new Event('submit'));
        }
    }
});
