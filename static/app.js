document.addEventListener('DOMContentLoaded', () => {
    // --- State & Selectors ---
    const trainForm = document.getElementById('train-form');
    const predictForm = document.getElementById('predict-form');
    const btnRefreshLogs = document.getElementById('btn-refresh-logs');
    const logsTest = document.getElementById('logs-test');

    // --- Train Model Logic ---
    if (trainForm) {
        trainForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('btn-train');
            const resultBox = document.getElementById('train-result');
            
            const payload = {
                country: document.getElementById('train-country').value,
                test: document.getElementById('train-test').checked
            };

            setLoading(btn, true);
            resultBox.classList.add('hidden');
            resultBox.className = 'result-box hidden'; 

            try {
                const response = await fetch('/train', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Training failed');

                // Displaying enhanced metrics from the new response structure
                const m = data.metrics;
                resultBox.innerHTML = `
                    <div class="success-header"><strong>Success!</strong> Model trained in ${data.runtime.toFixed(2)}s</div>
                    <div class="metrics-grid" style="display: grid; grid-template-columns: 1-fr 1fr; gap: 10px; margin-top: 10px; font-size: 0.9em;">
                        <span><strong>MAE %:</strong> ${m.mae_pct}%</span>
                        <span><strong>RMSE %:</strong> ${m.rmse_pct}%</span>
                        <span><strong>MAPE:</strong> ${m.mape}</span>
                        <span><strong>Model:</strong> ${m.model_type.toUpperCase()}</span>
                        <span><strong>Train Size:</strong> ${m.train_size}</span>
                        <span><strong>Features:</strong> ${m.n_features}</span>
                    </div>
                `;
                resultBox.classList.remove('hidden');
                resultBox.classList.add('success-msg');
            } catch (error) {
                resultBox.innerHTML = `<strong>Error:</strong> ${error.message}`;
                resultBox.classList.remove('hidden');
                resultBox.classList.add('error-msg');
            } finally {
                setLoading(btn, false);
                loadLogs();
            }
        });
    }

    // --- Predict Revenue Logic ---
    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('btn-predict');
            const resultBox = document.getElementById('predict-result');
            const valueSpan = document.getElementById('predict-value');

            const payload = {
                country: document.getElementById('predict-country').value,
                date: document.getElementById('predict-date').value,
                test: document.getElementById('predict-test').checked
            };

            setLoading(btn, true);
            resultBox.classList.add('hidden');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Prediction failed');

                const formatter = new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: 'USD',
                });

                // Mapping the specific key: predicted_revenue_30_days
                valueSpan.textContent = formatter.format(data.predicted_revenue_30_days);
                resultBox.classList.remove('hidden');
            } catch (error) {
                console.error(error);
                alert(`Prediction Error: ${error.message}`);
            } finally {
                setLoading(btn, false);
                loadLogs();
            }
        });
    }

    // --- Logs Logic ---
    async function loadLogs() {
        const tbody = document.getElementById('logs-body');
        if (!tbody) return;

        const isTest = logsTest ? logsTest.checked : true;
        tbody.innerHTML = '<tr><td colspan="6" class="status-msg">Fetching logs...</td></tr>';

        try {
            const response = await fetch(`/logs?test=${isTest}`);
            const data = await response.json();

            if (!response.ok) throw new Error(data.error || 'Failed to load logs');

            tbody.innerHTML = '';
            const logs = [...(data.logs || [])].reverse().slice(0, 50);

            if (logs.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" class="status-msg">No entries found.</td></tr>';
                return;
            }

            logs.forEach(log => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${new Date(log.timestamp).toLocaleString()}</td>
                    <td><span class="badge">${log.country.toUpperCase()}</span></td>
                    <td>${log.target_date || 'N/A'}</td>
                    <td class="revenue-cell">${log.y_pred ? '$' + parseFloat(log.y_pred).toLocaleString() : 'N/A'}</td>
                    <td>${parseFloat(log.runtime).toFixed(3)}s</td>
                    <td><small>${log.model_version || 'v1.0'}</small></td>
                `;
                tbody.appendChild(tr);
            });
        } catch (error) {
            tbody.innerHTML = `<tr><td colspan="6" class="status-msg error-text">Error: ${error.message}</td></tr>`;
        }
    }

    // --- UI Helpers ---
    function setLoading(btn, isLoading) {
        if (!btn) return;
        const text = btn.querySelector('.btn-text');
        const loader = btn.querySelector('.loader');

        btn.disabled = isLoading;
        if (isLoading) {
            if (text) text.style.display = 'none';
            if (loader) loader.classList.remove('hidden');
        } else {
            if (text) text.style.display = 'block';
            if (loader) loader.classList.add('hidden');
        }
    }

    if (btnRefreshLogs) btnRefreshLogs.addEventListener('click', loadLogs);
    if (logsTest) logsTest.addEventListener('change', loadLogs);

    loadLogs();
});