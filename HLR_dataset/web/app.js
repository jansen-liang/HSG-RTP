const state = { scenes: [], currentScene: null, currentFloorId: null };
const sceneSelect = document.getElementById("sceneSelect");
const sceneTitle = document.getElementById("sceneTitle");
const floorTitle = document.getElementById("floorTitle");
const sceneStats = document.getElementById("sceneStats");
const floorList = document.getElementById("floorList");
const nodeInfo = document.getElementById("nodeInfo");
const graphSvg = document.getElementById("graphSvg");

function info(node) {
  if (!node) {
    nodeInfo.textContent = "点击中间画布里的房间或物体查看详情。";
    return;
  }
  nodeInfo.innerHTML = `<div><strong>${node.label}</strong></div><div class="node-hint">类型：${node.kind}${node.room_id ? ` | 房间：${node.room_id}` : ""}</div><pre>${JSON.stringify(node.meta ?? {}, null, 2)}</pre>`;
}

function edgeClass(kind) {
  return kind === "neighbor" ? "edge-neighbor" : kind === "contains" ? "edge-contains" : kind === "ontop" ? "edge-ontop" : "edge-next_to";
}

function draw() {
  const view = state.currentScene.floor_views[state.currentFloorId];
  if (!view) return;
  floorList.innerHTML = "";
  for (const floor of state.currentScene.floors) {
    const btn = document.createElement("button");
    btn.className = `floor-button ${floor.id === state.currentFloorId ? "active" : ""}`;
    btn.innerHTML = `<div><div>${floor.name}</div><div class="floor-meta">${floor.room_count} rooms</div></div><div>${floor.id === state.currentScene.agent.current_floor ? "Robot" : ""}</div>`;
    btn.addEventListener("click", () => { state.currentFloorId = floor.id; draw(); });
    floorList.appendChild(btn);
  }
  sceneTitle.textContent = state.currentScene.scene.name;
  floorTitle.textContent = `${view.floor_name} · 仅显示本层拓扑`;
  sceneStats.innerHTML = `<div>${view.node_count} nodes</div><div>${view.edge_count} edges</div><div>Agent: ${state.currentScene.agent.current_room || "unknown"}</div>`;
  graphSvg.innerHTML = "";
  const map = new Map(view.nodes.map((node) => [node.id, node]));
  for (const edge of view.edges) {
    const s = map.get(edge.source), t = map.get(edge.target);
    if (!s || !t) continue;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", s.x); line.setAttribute("y1", s.y);
    line.setAttribute("x2", t.x); line.setAttribute("y2", t.y);
    line.setAttribute("class", edgeClass(edge.kind));
    graphSvg.appendChild(line);
  }
  for (const node of view.nodes) {
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "clickable");
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", node.x); c.setAttribute("cy", node.y);
    c.setAttribute("r", node.kind === "room" ? 34 : 16);
    c.setAttribute("class", node.kind === "room" ? `room-node ${node.is_agent_room ? "agent-room" : ""}` : "object-node");
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", node.x);
    text.setAttribute("y", node.kind === "room" ? node.y : node.y + 26);
    text.setAttribute("class", `node-label ${node.kind === "object" ? "object-label" : ""}`);
    text.textContent = node.label;
    g.appendChild(c); g.appendChild(text);
    g.addEventListener("click", () => info(node));
    graphSvg.appendChild(g);
  }
  info(null);
}

async function loadScene(sceneId) {
  const res = await fetch(`/api/scene/${encodeURIComponent(sceneId)}`);
  if (!res.ok) throw new Error(`Failed to load scene ${sceneId}`);
  state.currentScene = await res.json();
  state.currentFloorId = state.currentScene.current_floor || state.currentScene.floors?.[0]?.id || null;
  draw();
}

async function init() {
  const res = await fetch("/api/scenes");
  const payload = await res.json();
  state.scenes = payload.scenes || [];
  for (const scene of state.scenes) {
    const opt = document.createElement("option");
    opt.value = scene.id;
    opt.textContent = `${scene.name} · ${scene.floor_count}F`;
    sceneSelect.appendChild(opt);
  }
  sceneSelect.addEventListener("change", () => loadScene(sceneSelect.value).catch(console.error));
  if (!state.scenes.length) {
    sceneTitle.textContent = "没有找到可用场景";
    floorTitle.textContent = "请检查 HLR_dataset 下的 scene JSON 资产";
    return;
  }
  sceneSelect.value = state.scenes[0].id;
  await loadScene(state.scenes[0].id);
}

init().catch((error) => {
  console.error(error);
  sceneTitle.textContent = "Graphworld 启动失败";
  floorTitle.textContent = error.message;
});
