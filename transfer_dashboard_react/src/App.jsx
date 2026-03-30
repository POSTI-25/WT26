import { useEffect, useMemo, useRef, useState } from "react";

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "-";
  const mb = bytes / (1024 * 1024);
  return `${mb.toFixed(2)} MB`;
}

function nowTime() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

function parseErrorMessage(error, fallback) {
  if (!error) return fallback;
  if (typeof error === "string") return error;
  if (error.detail) return error.detail;
  if (error.message) return error.message;
  return fallback;
}

export default function App() {
  const [currentPage, setCurrentPage] = useState("receiver");

  const [host, setHost] = useState("127.0.0.1");
  const [port, setPort] = useState(50051);
  const [timeoutSeconds, setTimeoutSeconds] = useState(30);
  const [chunkSize, setChunkSize] = useState(1024 * 1024);

  const [apiHealthy, setApiHealthy] = useState(null);
  const [gpuReport, setGpuReport] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [transferResult, setTransferResult] = useState(null);

  const [gpuLoading, setGpuLoading] = useState(false);
  const [sendLoading, setSendLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [theme, setTheme] = useState("light");

  const [logLines, setLogLines] = useState([
    { level: "info", text: "Console ready.", time: nowTime() },
    { level: "ok", text: "Waiting for probe or transfer commands.", time: nowTime() }
  ]);

  const fileInputRef = useRef(null);

  const lastHistoryItem = history.length > 0 ? history[history.length - 1] : null;

  const averageThroughput = useMemo(() => {
    if (history.length === 0) return 0;
    const total = history.reduce((sum, item) => sum + Number(item.throughput_mb_s || 0), 0);
    return total / history.length;
  }, [history]);

  const gpuList = gpuReport?.ok ? gpuReport.gpus || [] : [];
  const activeNodes = gpuList.length;
  const busyNodes = gpuList.filter((gpu) => Number(gpu.utilization_gpu_percent || 0) >= 65).length;
  const idleNodes = gpuList.filter((gpu) => Number(gpu.utilization_gpu_percent || 0) <= 5).length;

  const totalGpuUsage =
    gpuList.length === 0
      ? 0
      : Math.round(
          gpuList.reduce((sum, gpu) => sum + Number(gpu.utilization_gpu_percent || 0), 0) /
            gpuList.length
        );

  const runningJobs = sendLoading ? 1 : Math.min(2, busyNodes);
  const queueRows = history.slice().reverse().slice(0, 8);

  const utilizationBars = useMemo(() => {
    if (gpuList.length === 0) {
      return [40, 65, 85, 45, 30, 70, 95, 55, 25];
    }

    const base = gpuList.map((gpu) => Number(gpu.utilization_gpu_percent || 0));
    const expanded = [];
    while (expanded.length < 9) {
      expanded.push(...base);
    }

    return expanded.slice(0, 9).map((value, idx) => {
      const swing = idx % 3 === 0 ? 5 : idx % 3 === 1 ? -4 : 2;
      return Math.max(5, Math.min(100, value + swing));
    });
  }, [gpuList]);

  const nodeCards =
    gpuList.length > 0
      ? gpuList.slice(0, 3).map((gpu, index) => {
          const util = Number(gpu.utilization_gpu_percent || 0);
          const memTotal = Number(gpu.memory_total_mb || 1);
          const memUsed = Number(gpu.memory_used_mb || 0);

          let status = "IDLE";
          if (util >= 65) status = "BUSY";
          else if (util > 5) status = "FREE";

          return {
            id: gpu.uuid || `${gpu.index}-${index}`,
            title: `Node ${String(index + 1).padStart(2, "0")}`,
            model: gpu.name || "Unknown GPU",
            util,
            mem: Math.max(0, Math.min(100, Math.round((memUsed / Math.max(memTotal, 1)) * 100))),
            status,
            task: status === "BUSY" ? `gpu_probe_${gpu.index}` : "Ready for assignment"
          };
        })
      : [
          {
            id: "placeholder-1",
            title: "Node 01",
            model: "A100 Tensor Core",
            util: 0,
            mem: 0,
            status: "IDLE",
            task: "Run GPU Probe to load live data"
          },
          {
            id: "placeholder-2",
            title: "Node 02",
            model: "A100 Tensor Core",
            util: 0,
            mem: 0,
            status: "IDLE",
            task: "Run GPU Probe to load live data"
          },
          {
            id: "placeholder-3",
            title: "Node 03",
            model: "H100 Hopper",
            util: 0,
            mem: 0,
            status: "IDLE",
            task: "Run GPU Probe to load live data"
          }
        ];

  async function readJson(response) {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw payload;
    return payload;
  }

  function appendLog(level, text) {
    setLogLines((prev) => [...prev, { level, text, time: nowTime() }].slice(-22));
  }

  async function checkApiHealth() {
    try {
      const response = await fetch("/api/health");
      const payload = await readJson(response);
      setApiHealthy(Boolean(payload.ok));
      appendLog("ok", "API health check passed.");
    } catch {
      setApiHealthy(false);
      appendLog("error", "API health check failed.");
    }
  }

  async function fetchHistory() {
    try {
      const response = await fetch("/api/history?limit=100");
      const payload = await readJson(response);
      setHistory(Array.isArray(payload.history) ? payload.history : []);
      appendLog("info", "History synchronized.");
    } catch (error) {
      setStatusMessage(parseErrorMessage(error, "Unable to load transfer history."));
      appendLog("error", "History sync failed.");
    }
  }

  async function handleProbeGpu() {
    setGpuLoading(true);
    setStatusMessage("Fetching GPU status from receiver...");
    appendLog("info", `Probing GPU on ${host}:${port} ...`);

    try {
      const response = await fetch("/api/gpu-probe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          host,
          port: Number(port),
          timeout_seconds: Number(timeoutSeconds)
        })
      });

      const payload = await readJson(response);
      setGpuReport(payload.report);
      setStatusMessage("GPU probe completed.");
      appendLog("ok", "GPU probe completed.");
    } catch (error) {
      setGpuReport(null);
      setStatusMessage(parseErrorMessage(error, "GPU probe failed."));
      appendLog("error", "GPU probe failed.");
    } finally {
      setGpuLoading(false);
    }
  }

  async function handleSendFile() {
    if (!selectedFile) {
      setStatusMessage("Select a file first.");
      appendLog("error", "Send failed: no file selected.");
      return;
    }

    setSendLoading(true);
    setStatusMessage(`Transferring ${selectedFile.name}...`);
    appendLog("info", `Transfer started: ${selectedFile.name}`);

    const form = new FormData();
    form.append("host", host);
    form.append("port", String(Number(port)));
    form.append("timeout_seconds", String(Number(timeoutSeconds)));
    form.append("chunk_size", String(Number(chunkSize)));
    form.append("file", selectedFile);

    try {
      const response = await fetch("/api/send-file", {
        method: "POST",
        body: form
      });

      const payload = await readJson(response);
      setTransferResult(payload.result);
      setStatusMessage("Transfer completed successfully.");
      appendLog("ok", `Transfer completed: ${selectedFile.name}`);
      await fetchHistory();
    } catch (error) {
      setTransferResult(null);
      setStatusMessage(parseErrorMessage(error, "Transfer failed."));
      appendLog("error", `Transfer failed: ${selectedFile.name}`);
    } finally {
      setSendLoading(false);
    }
  }

  useEffect(() => {
    const syncFromHash = () => {
      const value = window.location.hash.replace("#/", "").trim();
      if (value === "sender" || value === "receiver") {
        setCurrentPage(value);
      }
    };

    syncFromHash();
    window.addEventListener("hashchange", syncFromHash);
    return () => window.removeEventListener("hashchange", syncFromHash);
  }, []);

  useEffect(() => {
    const hash = `#/${currentPage}`;
    if (window.location.hash !== hash) {
      window.location.hash = hash;
    }
  }, [currentPage]);

  useEffect(() => {
    checkApiHealth();
    fetchHistory();
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  return (
    <div className={`dashboard-shell ${theme === "dark" ? "theme-dark" : "theme-light"}`}>
      <header className="topbar">
        <div className="topbar-left">
          <div className="brand-icon">hub</div>
          <div className="topbar-title-wrap">
            <h1>Distributed Compute Dashboard</h1>
            <p className="topbar-subtitle">
              {currentPage === "receiver" ? "Receiver page" : "Sender page"}
            </p>
          </div>
        </div>

        <div className="topbar-actions">
          <button
            className="icon-btn"
            onClick={() => {
              checkApiHealth();
              fetchHistory();
            }}
          >
            refresh
          </button>

          <button
            className="icon-btn"
            onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
          >
            dark_mode
          </button>

          <button
            className="pill-btn"
            onClick={currentPage === "receiver" ? handleProbeGpu : handleSendFile}
            disabled={gpuLoading || sendLoading}
          >
            {currentPage === "receiver"
              ? gpuLoading
                ? "Probing..."
                : "Probe Receiver"
              : sendLoading
                ? "Sending..."
                : "Send File"}
          </button>
        </div>
      </header>

      <aside className="sidebar">
        <div className="sidebar-brand">
          <h2>Compute Engine</h2>
          <p>V-1.0.4</p>
        </div>

        <nav className="side-nav">
          <button
            className={currentPage === "receiver" ? "active" : ""}
            onClick={() => setCurrentPage("receiver")}
          >
            radio_tower Receiver
          </button>
          <button
            className={currentPage === "sender" ? "active" : ""}
            onClick={() => setCurrentPage("sender")}
          >
            upload_file Sender
          </button>
        </nav>

        <div className="control-block">
          <p className="control-title">Connection</p>

          <label>Host</label>
          <input value={host} onChange={(e) => setHost(e.target.value)} />

          <div className="row-2">
            <div>
              <label>Port</label>
              <input
                type="number"
                min={1}
                max={65535}
                value={port}
                onChange={(e) => setPort(e.target.value)}
              />
            </div>
            <div>
              <label>Timeout</label>
              <input
                type="number"
                min={1}
                max={120}
                value={timeoutSeconds}
                onChange={(e) => setTimeoutSeconds(e.target.value)}
              />
            </div>
          </div>

          <label>Chunk Size</label>
          <input
            type="number"
            min={65536}
            max={33554432}
            step={65536}
            value={chunkSize}
            onChange={(e) => setChunkSize(e.target.value)}
          />

          {currentPage === "sender" && (
            <>
              <label>Select File</label>
              <input
                ref={fileInputRef}
                type="file"
                onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
              />
              {selectedFile && (
                <div className="selected-file">
                  <span>{selectedFile.name}</span>
                  <small>{formatBytes(selectedFile.size)}</small>
                </div>
              )}
              <button className="solid" onClick={handleSendFile} disabled={sendLoading || gpuLoading}>
                {sendLoading ? "Sending..." : "Send File"}
              </button>
            </>
          )}

          {currentPage === "receiver" && (
            <button className="solid" onClick={handleProbeGpu} disabled={gpuLoading || sendLoading}>
              {gpuLoading ? "Probing..." : "Probe Receiver"}
            </button>
          )}

          <button className="ghost" onClick={fetchHistory} disabled={gpuLoading || sendLoading}>
            Refresh History
          </button>

          <p className="status-note">{statusMessage || "System ready."}</p>
        </div>

        <div className="sidebar-user">
          <div className="avatar">person</div>
          <div>
            <p>System Admin</p>
            <span>{apiHealthy ? "Online" : "Offline"}</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        {currentPage === "receiver" ? (
          <>
            <section className="overview-grid">
              <article className="overview-card">
                <p>Active Nodes</p>
                <h3>{activeNodes}</h3>
                <div className="pill success">trending_up {apiHealthy ? "API Reachable" : "API Offline"}</div>
              </article>
              <article className="overview-card primary">
                <p>Running Jobs</p>
                <h3>{runningJobs}</h3>
                <div className="pill">sync Processing...</div>
              </article>
              <article className="overview-card">
                <p>Idle Nodes</p>
                <h3>{idleNodes}</h3>
                <div className="pill neutral">sleep Standby</div>
              </article>
              <article className="overview-card">
                <p>Total GPU Usage</p>
                <h3>
                  {totalGpuUsage}
                  <span>%</span>
                </h3>
                <div className="progress-track">
                  <div style={{ width: `${Math.max(0, Math.min(100, totalGpuUsage))}%` }} />
                </div>
              </article>
            </section>

            <section className="bento-grid">
              <div className="node-grid-panel">
                <div className="section-head">
                  <h2>Node Status Grid</h2>
                  <span>Real-time update</span>
                </div>

                <div className="node-cards">
                  {nodeCards.map((node) => {
                    const statusClass =
                      node.status === "BUSY" ? "busy" : node.status === "FREE" ? "free" : "idle";

                    return (
                      <article key={node.id} className="node-card">
                        <div className="node-top">
                          <div>
                            <h4>{node.title}</h4>
                            <p>{node.model}</p>
                          </div>
                          <div className={`node-status ${statusClass}`}>{node.status}</div>
                        </div>

                        <div className="metric-row">
                          <div className="metric-label">
                            <span>GPU Load</span>
                            <span>{node.util}%</span>
                          </div>
                          <div className="bar"><div style={{ width: `${node.util}%` }} /></div>
                        </div>

                        <div className="metric-row">
                          <div className="metric-label">
                            <span>Memory</span>
                            <span>{node.mem}%</span>
                          </div>
                          <div className="bar"><div style={{ width: `${node.mem}%` }} /></div>
                        </div>

                        <div className="task-block">
                          <p>Current Task</p>
                          <code>{node.task}</code>
                        </div>
                      </article>
                    );
                  })}
                </div>
              </div>

              <div className="live-panel">
                <div className="section-head compact">
                  <h2>Live Utilization</h2>
                </div>

                <div className="live-card">
                  <p>Real-time Flow</p>
                  <h4>GPU Performance</h4>

                  <div className="bar-spark">
                    {utilizationBars.map((value, idx) => (
                      <span key={`spark-${idx}`} style={{ height: `${Math.max(10, Math.min(100, value))}%` }} />
                    ))}
                  </div>

                  <div className="live-footer">
                    <div>
                      <p>Last Throughput</p>
                      <strong>
                        {lastHistoryItem
                          ? `${Number(lastHistoryItem.throughput_mb_s).toFixed(2)} MB/s`
                          : "0.00 MB/s"}
                      </strong>
                    </div>
                    <div>
                      <p>Avg Throughput</p>
                      <strong>{averageThroughput.toFixed(2)} MB/s</strong>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </>
        ) : (
          <section className="sender-layout">
            <div className="overview-grid">
              <article className="overview-card primary">
                <p>Selected File</p>
                <h3>{selectedFile ? formatBytes(selectedFile.size) : "0 MB"}</h3>
                <div className="pill">upload_file {selectedFile ? selectedFile.name : "No file selected"}</div>
              </article>
              <article className="overview-card">
                <p>Last Send Throughput</p>
                <h3>
                  {transferResult ? Number(transferResult.throughput_mbps).toFixed(2) : "0.00"}
                  <span>MB/s</span>
                </h3>
                <div className="pill neutral">bolt Transfer metric</div>
              </article>
              <article className="overview-card">
                <p>Total Transfers</p>
                <h3>{history.length}</h3>
                <div className="pill success">check_circle History tracked</div>
              </article>
              <article className="overview-card">
                <p>Receiver Snapshot</p>
                <h3>{gpuList.length}</h3>
                <div className="pill neutral">memory GPUs detected</div>
              </article>
            </div>

            <div className="jobs-panel">
              <div className="jobs-grid">
                <div className="queue-side">
                  <div className="queue-tabs">
                    <button className="active">Sender Queue ({queueRows.length})</button>
                  </div>

                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Job ID</th>
                          <th>File</th>
                          <th>Status</th>
                          <th>Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {queueRows.length === 0 && (
                          <tr>
                            <td colSpan={4} className="empty-row">
                              No sender jobs yet. Select a file and send it.
                            </td>
                          </tr>
                        )}
                        {queueRows.map((row, idx) => (
                          <tr key={`${row.timestamp}-${row.file}-${idx}`}>
                            <td className="mono">tx_{idx + 1}_{String(row.file).slice(0, 8)}</td>
                            <td>{row.file}</td>
                            <td>
                              <span className={`tiny-pill ${row.ok ? "ok" : "bad"}`}>
                                {row.ok ? "Completed" : "Failed"}
                              </span>
                            </td>
                            <td>{row.timestamp.split(" ")[1] || row.timestamp}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="log-side">
                  <div className="terminal-head">
                    <div className="lights">
                      <span />
                      <span />
                      <span />
                    </div>
                    <small>Sender Console</small>
                  </div>

                  <div className="terminal-body">
                    {logLines.map((line, idx) => (
                      <p key={`log-${idx}`} className={line.level}>
                        [{line.time}] {line.text}
                      </p>
                    ))}
                    <p className="cursor">_</p>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        {currentPage === "receiver" && (
          <section className="jobs-panel">
            <div className="jobs-grid">
              <div className="queue-side">
                <div className="queue-tabs">
                  <button>Pending (0)</button>
                  <button className="active">Running ({runningJobs})</button>
                  <button>Completed ({history.length})</button>
                </div>

                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Job ID</th>
                        <th>Node</th>
                        <th>Status</th>
                        <th>Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {queueRows.length === 0 && (
                        <tr>
                          <td colSpan={4} className="empty-row">
                            No receiver jobs yet. Probe receiver to populate status.
                          </td>
                        </tr>
                      )}
                      {queueRows.map((row, idx) => (
                        <tr key={`${row.timestamp}-${row.file}-${idx}`}>
                          <td className="mono">job_{idx + 1}_{String(row.file).slice(0, 8)}</td>
                          <td>Node_{(idx % 3) + 1}</td>
                          <td>
                            <span className={`tiny-pill ${row.ok ? "ok" : "bad"}`}>
                              {row.ok ? "Completed" : "Failed"}
                            </span>
                          </td>
                          <td>{row.timestamp.split(" ")[1] || row.timestamp}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="log-side">
                <div className="terminal-head">
                  <div className="lights">
                    <span />
                    <span />
                    <span />
                  </div>
                  <small>Receiver Console</small>
                </div>

                <div className="terminal-body">
                  {logLines.map((line, idx) => (
                    <p key={`log-${idx}`} className={line.level}>
                      [{line.time}] {line.text}
                    </p>
                  ))}
                  <p className="cursor">_</p>
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      <button
        className="fab"
        onClick={() => {
          setCurrentPage("sender");
          fileInputRef.current?.click();
        }}
        title="Choose file"
      >
        add
      </button>
    </div>
  );
}
