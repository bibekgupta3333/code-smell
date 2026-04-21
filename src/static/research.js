/**
 * Research Experiment Management Module
 *
 * Implements frontend for RQ1-RQ4 research experiments:
 * - RQ1: LLM vs Static Tools accuracy comparison
 * - RQ2: RAG effectiveness analysis
 * - RQ3: Per-smell-type performance breakdown
 * - RQ4: Computational resource requirements
 */

// ============================================================================
// State Management
// ============================================================================

const researchState = {
  experiments: [],
  selectedExperiment: null,
  activeRQ: null,
  pollingIntervals: {},
};

// ============================================================================
// Experiment Management
// ============================================================================

/**
 * Start a new research experiment
 */
async function startExperiment(experimentType, options = {}) {
  const button = document.querySelector(`[data-experiment="${experimentType}"]`);
  if (button) button.disabled = true;

  try {
    const response = await fetch("/api/v1/research/experiments/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        experiment_type: experimentType,
        model_names: options.models || ["llama3:8b"],
        dataset_split: options.split || "test",
        include_baselines: options.baselines !== false,
        enable_rag: options.rag !== false,
        notes: options.notes || null,
      }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    showToast(`Experiment started: ${data.experiment_id}`, "success");

    // Load experiments list and select the new one
    await loadExperimentsList();
    selectExperiment(data.experiment_id);

    // Start polling progress
    pollExperimentProgress(data.experiment_id);
  } catch (error) {
    console.error("Failed to start experiment:", error);
    showToast(`Failed to start experiment: ${error.message}`, "danger");
  } finally {
    if (button) button.disabled = false;
  }
}

/**
 * Load list of all experiments
 */
async function loadExperimentsList() {
  try {
    const response = await fetch("/api/v1/research/experiments?limit=50");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    researchState.experiments = data.experiments;

    // Render experiments table
    renderExperimentsTable(data.experiments);
  } catch (error) {
    console.error("Failed to load experiments:", error);
    showToast("Failed to load experiments list", "danger");
  }
}

/**
 * Select an experiment and display its details
 */
async function selectExperiment(experimentId) {
  try {
    // Get status
    const statusResponse = await fetch(`/api/v1/research/experiments/${experimentId}/status`);
    if (!statusResponse.ok) throw new Error(`HTTP ${statusResponse.status}`);

    const status = await statusResponse.json();
    researchState.selectedExperiment = experimentId;

    // Render status
    renderExperimentStatus(status);

    // If completed, fetch results
    if (status.status === "completed") {
      fetchExperimentResults(experimentId);
    }
  } catch (error) {
    console.error("Failed to select experiment:", error);
  }
}

/**
 * Poll experiment progress
 */
function pollExperimentProgress(experimentId) {
  // Clear existing interval if any
  if (researchState.pollingIntervals[experimentId]) {
    clearInterval(researchState.pollingIntervals[experimentId]);
  }

  // Poll every 2 seconds
  researchState.pollingIntervals[experimentId] = setInterval(async () => {
    try {
      const response = await fetch(
        `/api/v1/research/experiments/${experimentId}/status`
      );
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const status = await response.json();

      // Update UI
      updateExperimentProgressUI(status);

      // If completed, fetch results and stop polling
      if (status.status === "completed") {
        clearInterval(researchState.pollingIntervals[experimentId]);
        delete researchState.pollingIntervals[experimentId];
        fetchExperimentResults(experimentId);
      }
    } catch (error) {
      console.error("Polling error:", error);
    }
  }, 2000);
}

/**
 * Fetch completed experiment results
 */
async function fetchExperimentResults(experimentId) {
  try {
    const response = await fetch(
      `/api/v1/research/experiments/${experimentId}/results`
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const results = await response.json();

    // Render results based on experiment type
    renderExperimentResults(results);

    showToast("Results ready", "success");
  } catch (error) {
    if (error.message.includes("202")) {
      // Still processing
      console.log("Results not yet available");
    } else {
      console.error("Failed to fetch results:", error);
      showToast("Failed to fetch results", "danger");
    }
  }
}

// ============================================================================
// UI Rendering
// ============================================================================

/**
 * Render experiments table
 */
function renderExperimentsTable(experiments) {
  const tbody = document.getElementById("experiments-tbody");
  if (!tbody) return;

  tbody.innerHTML = experiments.map((exp) => `
    <tr class="experiment-row" onclick="selectExperiment('${exp.experiment_id}')">
      <td>
        <code class="exp-id">${exp.experiment_id.substring(0, 20)}...</code>
      </td>
      <td>
        <span class="badge bg-info">${exp.experiment_type}</span>
      </td>
      <td>
        <span class="status-badge status-${exp.status}">
          ${exp.status}
        </span>
      </td>
      <td>
        <div class="progress" style="height: 6px;">
          <div class="progress-bar" style="width: ${exp.progress_percent}%"></div>
        </div>
        <small>${exp.progress_percent.toFixed(0)}%</small>
      </td>
      <td>
        <small>${new Date(exp.created_at).toLocaleString()}</small>
      </td>
    </tr>
  `).join("");
}

/**
 * Render experiment status UI
 */
function renderExperimentStatus(status) {
  const statusPanel = document.getElementById("experiment-status-panel");
  if (!statusPanel) return;

  statusPanel.innerHTML = `
    <div class="status-detail">
      <h3>Experiment ${status.experiment_type.toUpperCase()}</h3>
      <p class="exp-id-display">ID: <code>${status.experiment_id}</code></p>

      <div class="status-grid">
        <article>
          <span>Status</span>
          <strong class="status-${status.status}">${status.status}</strong>
        </article>
        <article>
          <span>Progress</span>
          <strong>${status.progress_percent.toFixed(0)}%</strong>
        </article>
        <article>
          <span>Processed</span>
          <strong>${status.processed_samples}/${status.total_samples}</strong>
        </article>
        <article>
          <span>Current Step</span>
          <strong>${status.current_step || "Initializing..."}</strong>
        </article>
      </div>

      <div class="progress mt-3" style="height: 10px;">
        <div class="progress-bar bg-success" style="width: ${status.progress_percent}%"></div>
      </div>
    </div>
  `;
}

/**
 * Update experiment progress in UI
 */
function updateExperimentProgressUI(status) {
  const progressBar = document.querySelector(".progress-bar");
  const currentStepEl = document.querySelector(".status-detail .status-grid [role='article']:last-child strong");

  if (progressBar) {
    progressBar.style.width = `${status.progress_percent}%`;
  }

  if (currentStepEl) {
    currentStepEl.textContent = status.current_step || "Initializing...";
  }
}

/**
 * Render experiment results
 */
function renderExperimentResults(results) {
  const resultsPanel = document.getElementById("experiment-results-panel");
  if (!resultsPanel) return;

  let html = `<div class="results-container">`;

  // RQ1: Baseline comparison
  if (results.rq1_metrics) {
    html += renderRQ1Results(results);
  }

  // RQ2: RAG effectiveness
  if (results.rq2_metrics) {
    html += renderRQ2Results(results);
  }

  // RQ3: Per-smell analysis
  if (results.rq3_per_smell_performance) {
    html += renderRQ3Results(results);
  }

  // RQ4: Performance metrics
  if (results.rq4_latency_breakdown) {
    html += renderRQ4Results(results);
  }

  html += `</div>`;
  resultsPanel.innerHTML = html;
}

/**
 * Render RQ1 Results: LLM vs Static Tools
 */
function renderRQ1Results(results) {
  const metrics = results.rq1_metrics;
  const comparison = results.rq1_tool_comparison || {};

  return `
    <section class="rq-results-section">
      <h4><i class="bi bi-bar-chart"></i> RQ1: LLM vs Static Tools Accuracy</h4>
      <p class="rq-description">
        How accurately do LLMs detect code smells compared to traditional static analysis tools?
      </p>

      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Precision</div>
          <div class="metric-value">${(metrics.llm_precision * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Recall</div>
          <div class="metric-value">${(metrics.llm_recall * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">F1-Score</div>
          <div class="metric-value">${(metrics.llm_f1_score * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Accuracy</div>
          <div class="metric-value">${(metrics.llm_accuracy * 100).toFixed(1)}%</div>
        </div>
      </div>

      <div class="comparison-table">
        <h5>Tool Comparison</h5>
        <table class="table table-sm">
          <thead>
            <tr>
              <th>Tool</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
            </tr>
          </thead>
          <tbody>
            ${Object.entries(comparison).map(([tool, metrics]) => `
              <tr>
                <td><strong>${tool}</strong></td>
                <td>${(metrics.precision * 100).toFixed(1)}%</td>
                <td>${(metrics.recall * 100).toFixed(1)}%</td>
                <td>${(metrics.f1 * 100).toFixed(1)}%</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </section>
  `;
}

/**
 * Render RQ2 Results: RAG Effectiveness
 */
function renderRQ2Results(results) {
  const metrics = results.rq2_metrics;

  return `
    <section class="rq-results-section">
      <h4><i class="bi bi-graph-up"></i> RQ2: RAG Effectiveness</h4>
      <p class="rq-description">
        Does retrieval-augmented generation improve detection accuracy compared to vanilla prompting?
      </p>

      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Vanilla Accuracy</div>
          <div class="metric-value">${(metrics.vanilla_accuracy * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">RAG Accuracy</div>
          <div class="metric-value">${(metrics.rag_accuracy * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-card highlight">
          <div class="metric-label">Improvement</div>
          <div class="metric-value">+${metrics.improvement_percent.toFixed(1)}%</div>
        </div>
        <div class="metric-card highlight">
          <div class="metric-label">FP Reduction</div>
          <div class="metric-value">-${metrics.false_positive_reduction_percent.toFixed(1)}%</div>
        </div>
      </div>

      <p class="insight">
        <strong>Finding:</strong> RAG improves accuracy by ${metrics.improvement_percent.toFixed(1)}% and reduces false positives by ${metrics.false_positive_reduction_percent.toFixed(1)}%.
      </p>
    </section>
  `;
}

/**
 * Render RQ3 Results: Per-Smell Analysis
 */
function renderRQ3Results(results) {
  const perSmell = results.rq3_per_smell_performance || {};
  const errorAnalysis = results.rq3_error_analysis || {};

  return `
    <section class="rq-results-section">
      <h4><i class="bi bi-bug"></i> RQ3: Per-Smell-Type Performance</h4>
      <p class="rq-description">
        Which code smell types are detected most/least accurately, and what factors influence performance?
      </p>

      <div class="comparison-table">
        <h5>Performance by Smell Type</h5>
        <table class="table table-sm">
          <thead>
            <tr>
              <th>Smell Type</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
            </tr>
          </thead>
          <tbody>
            ${Object.entries(perSmell).map(([smell, metrics]) => `
              <tr>
                <td><strong>${smell.replace(/_/g, " ")}</strong></td>
                <td>${(metrics.precision * 100).toFixed(1)}%</td>
                <td>${(metrics.recall * 100).toFixed(1)}%</td>
                <td>${(metrics.f1 * 100).toFixed(1)}%</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>

      <div class="error-analysis">
        <h5>Error Analysis</h5>
        <p>
          <strong>Most Difficult:</strong> ${errorAnalysis.most_difficult_smell || "N/A"} <br>
          <strong>Most Accurate:</strong> ${errorAnalysis.most_accurate_smell || "N/A"}
        </p>
        ${errorAnalysis.common_misclassifications ? `
          <p><strong>Common Misclassifications:</strong></p>
          <ul>
            ${Object.entries(errorAnalysis.common_misclassifications).map(([error, count]) =>
              `<li>${error}: ${count} instances</li>`
            ).join("")}
          </ul>
        ` : ""}
      </div>
    </section>
  `;
}

/**
 * Render RQ4 Results: Performance & Latency
 */
function renderRQ4Results(results) {
  const latency = results.rq4_latency_breakdown || {};
  const resource = results.rq4_resource_usage || {};
  const throughput = results.rq4_throughput;

  return `
    <section class="rq-results-section">
      <h4><i class="bi bi-speedometer"></i> RQ4: Computational Requirements</h4>
      <p class="rq-description">
        What are the computational resource requirements and latency characteristics for practical deployment?
      </p>

      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-label">Total Latency</div>
          <div class="metric-value">${(latency.total_ms || 0).toFixed(0)}ms</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Memory</div>
          <div class="metric-value">${(resource.avg_memory_mb || 0).toFixed(0)}MB</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">CPU Usage</div>
          <div class="metric-value">${(resource.avg_cpu_percent || 0).toFixed(0)}%</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Throughput</div>
          <div class="metric-value">${(throughput || 0).toFixed(0)}/hr</div>
        </div>
      </div>

      <div class="latency-breakdown">
        <h5>Latency Breakdown</h5>
        <table class="table table-sm">
          <tr>
            <td>Embedding</td>
            <td><strong>${latency.embedding_ms || 0}ms</strong></td>
          </tr>
          <tr>
            <td>Retrieval</td>
            <td><strong>${latency.retrieval_ms || 0}ms</strong></td>
          </tr>
          <tr>
            <td>Inference</td>
            <td><strong>${latency.inference_ms || 0}ms</strong></td>
          </tr>
          <tr>
            <td>Parsing</td>
            <td><strong>${latency.parsing_ms || 0}ms</strong></td>
          </tr>
          <tr>
            <td>Overhead</td>
            <td><strong>${latency.overhead_ms || 0}ms</strong></td>
          </tr>
        </table>
      </div>

      <div class="resource-details">
        <h5>Resource Usage</h5>
        <p>
          Peak Memory: <strong>${(resource.peak_memory_mb || 0).toFixed(0)}MB</strong> <br>
          GPU Utilization: <strong>${(resource.gpu_utilization_percent || 0).toFixed(0)}%</strong> <br>
          GPU Memory: <strong>${(resource.gpu_memory_mb || 0).toFixed(0)}MB</strong>
        </p>
      </div>
    </section>
  `;
}

// ============================================================================
// RQ Summary Endpoints
// ============================================================================

/**
 * Load RQ1 Summary
 */
async function loadRQ1Summary() {
  try {
    const response = await fetch("/api/v1/research/rq1/summary");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    displayRQSummary("rq1", data);
  } catch (error) {
    console.error("Failed to load RQ1 summary:", error);
    showToast("Failed to load RQ1 summary", "danger");
  }
}

/**
 * Load RQ2 Summary
 */
async function loadRQ2Summary() {
  try {
    const response = await fetch("/api/v1/research/rq2/summary");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    displayRQSummary("rq2", data);
  } catch (error) {
    console.error("Failed to load RQ2 summary:", error);
    showToast("Failed to load RQ2 summary", "danger");
  }
}

/**
 * Load RQ3 Summary
 */
async function loadRQ3Summary() {
  try {
    const response = await fetch("/api/v1/research/rq3/summary");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    displayRQSummary("rq3", data);
  } catch (error) {
    console.error("Failed to load RQ3 summary:", error);
    showToast("Failed to load RQ3 summary", "danger");
  }
}

/**
 * Load RQ4 Summary
 */
async function loadRQ4Summary() {
  try {
    const response = await fetch("/api/v1/research/rq4/summary");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    displayRQSummary("rq4", data);
  } catch (error) {
    console.error("Failed to load RQ4 summary:", error);
    showToast("Failed to load RQ4 summary", "danger");
  }
}

/**
 * Display RQ summary in UI
 */
function displayRQSummary(rqNum, data) {
  const summaryPanel = document.getElementById("rq-summary-panel");
  if (!summaryPanel) return;

  summaryPanel.innerHTML = `
    <div class="rq-summary">
      <h3>Research Question ${rqNum.toUpperCase()}</h3>
      <p class="rq-question">${data.research_question}</p>
      <pre>${JSON.stringify(data, null, 2)}</pre>
    </div>
  `;
}

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize research dashboard
 */
function initResearchDashboard() {
  // Load experiments list on page load
  loadExperimentsList();

  // Add event listeners for experiment buttons
  document.querySelectorAll("[data-start-experiment]").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const type = btn.dataset.startExperiment;
      startExperiment(type);
    });
  });

  // Add event listeners for RQ summary buttons
  document.getElementById("btn-load-rq1")?.addEventListener("click", loadRQ1Summary);
  document.getElementById("btn-load-rq2")?.addEventListener("click", loadRQ2Summary);
  document.getElementById("btn-load-rq3")?.addEventListener("click", loadRQ3Summary);
  document.getElementById("btn-load-rq4")?.addEventListener("click", loadRQ4Summary);
}

// Auto-initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("research-dashboard")) {
    initResearchDashboard();
  }
});
