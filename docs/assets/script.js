const icons = {
  file: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/></svg>',
  github: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5a12 12 0 0 0-3.8 23.38c.6.1.82-.26.82-.58v-2.02c-3.34.72-4.04-1.42-4.04-1.42-.55-1.38-1.34-1.75-1.34-1.75-1.08-.74.08-.72.08-.72 1.2.08 1.84 1.24 1.84 1.24 1.08 1.82 2.82 1.3 3.5 1 .1-.78.42-1.3.76-1.6-2.66-.3-5.46-1.34-5.46-5.94 0-1.32.46-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.18 0 0 1.02-.32 3.3 1.24a11.42 11.42 0 0 1 6 0c2.28-1.56 3.3-1.24 3.3-1.24.66 1.66.24 2.88.12 3.18.78.84 1.24 1.9 1.24 3.22 0 4.62-2.8 5.62-5.48 5.92.44.38.82 1.12.82 2.26v3.36c0 .32.22.7.82.58A12 12 0 0 0 12 .5Z"/></svg>',
  database: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
  layers: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m12.83 2.18 8.5 4.25a.75.75 0 0 1 0 1.34l-8.5 4.25a1.85 1.85 0 0 1-1.66 0l-8.5-4.25a.75.75 0 0 1 0-1.34l8.5-4.25a1.85 1.85 0 0 1 1.66 0Z"/><path d="m22 12-9.17 4.59a1.85 1.85 0 0 1-1.66 0L2 12"/><path d="m22 17-9.17 4.59a1.85 1.85 0 0 1-1.66 0L2 17"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>',
  check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>'
};

const citations = {
  BibTeX: `@misc{elghazoualisymtrs,
      title={SyMTRS: Benchmark Multi-Task Synthetic Dataset for Depth, Domain Adaptation and Super-Resolution in Aerial Imagery}, 
      author={Safouane El Ghazouali and Nicola Venturi and Michael Rueegsegger and Umberto Michelucci},
      year={2026},
      eprint={2604.21801},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.21801}, 
}`,
  PlainText: "El Ghazouali, S., Venturi, N., Rueegsegger, M., & Michelucci, U. (2026). SyMTRS: Benchmark Multi-Task Synthetic Dataset for Depth, Domain Adaptation and Super-Resolution in Aerial Imagery. arXiv preprint arXiv:2604.21801.",
  RIS: `TY  - JOUR
AU  - El Ghazouali, Safouane
AU  - Venturi, Nicola
AU  - Rueegsegger, Michael
AU  - Michelucci, Umberto
PY  - 2026
TI  - SyMTRS: Benchmark Multi-Task Synthetic Dataset for Depth, Domain Adaptation and Super-Resolution in Aerial Imagery
JO  - arXiv preprint arXiv:2604.21801
UR  - https://arxiv.org/abs/2604.21801
ID  - elghazouali2026symtrs
ER  - `
};

let activeCitation = "BibTeX";

function mountIcons() {
  document.querySelectorAll("[data-icon]").forEach((node) => {
    const icon = node.dataset.icon;
    if (icons[icon]) {
      node.innerHTML = icons[icon];
    }
  });
}

function setupCitationTabs() {
  const output = document.getElementById("citationText");
  const copyButton = document.getElementById("copyCitation");
  const copyLabel = document.getElementById("copyLabel");
  const tabs = document.querySelectorAll(".tab");

  const render = () => {
    output.textContent = citations[activeCitation];
    copyLabel.textContent = `Copy ${activeCitation}`;
    copyButton.querySelector("[data-icon]").innerHTML = icons.copy;
    tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === activeCitation));
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      activeCitation = tab.dataset.tab;
      render();
    });
  });

  copyButton.addEventListener("click", async () => {
    await copyText(citations[activeCitation]);
    copyButton.querySelector("[data-icon]").innerHTML = icons.check;
    copyLabel.textContent = "Copied!";
    window.setTimeout(render, 2000);
  });

  render();
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand("copy");
  textarea.remove();
}

function parseVector(value, fallback) {
  if (!value) return fallback;
  const vector = value.split(",").map(Number);
  return vector.length === 3 && vector.every(Number.isFinite) ? vector : fallback;
}

function showViewerError(container, message) {
  const loading = container.querySelector(".viewer-loading");
  if (loading) {
    loading.className = "viewer-error";
    loading.textContent = message;
  }
}

async function loadPLYLibraries() {
  const [threeModule, controlsModule, loaderModule] = await Promise.all([
    import("three"),
    import("three/addons/controls/TrackballControls.js"),
    import("three/addons/loaders/PLYLoader.js")
  ]);

  return {
    THREE: threeModule,
    TrackballControls: controlsModule.TrackballControls,
    PLYLoader: loaderModule.PLYLoader
  };
}

function setupPLYViewer(container, libraries) {
  const { THREE, TrackballControls, PLYLoader } = libraries;
  const title = container.dataset.title || "";
  const urls = [container.dataset.url, container.dataset.fallbackUrl].filter(Boolean);
  const pos = parseVector(container.dataset.pos, [8, 8, 8]);
  const rot = parseVector(container.dataset.rot, null);

  const loading = document.createElement("div");
  loading.className = "viewer-loading";
  loading.innerHTML = '<div class="spinner"></div>';
  container.appendChild(loading);

  const label = document.createElement("div");
  label.className = "viewer-label";
  label.textContent = title;
  container.appendChild(label);

  const width = container.clientWidth;
  const height = container.clientHeight;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf5f5f5);

  const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
  camera.position.set(...pos);
  if (rot) {
    camera.rotation.set(...rot);
  }

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.insertBefore(renderer.domElement, loading);

  const controls = new TrackballControls(camera, renderer.domElement);
  controls.rotateSpeed = 4.0;
  controls.zoomSpeed = 1.2;
  controls.panSpeed = 0.8;
  controls.noZoom = false;
  controls.noPan = false;
  controls.staticMoving = true;
  controls.dynamicDampingFactor = 0.3;

  if (rot) {
    const vector = new THREE.Vector3(0, 0, -1);
    vector.applyQuaternion(camera.quaternion);
    controls.target.copy(camera.position).add(vector.multiplyScalar(10));
  }

  scene.add(new THREE.AmbientLight(0xffffff, 0.8));

  const addGeometry = (geometry) => {
      geometry.computeVertexNormals();
      geometry.center();

      const material = new THREE.PointsMaterial({ size: 0.005, vertexColors: true });
      const points = new THREE.Points(geometry, material);
      points.rotation.x = -Math.PI / 2;
      points.rotation.y = Math.PI;

      scene.add(points);
      loading.remove();
  };

  loadPLYWithFallback(new PLYLoader(), urls, addGeometry, (error) => {
    console.error("PLY load failed:", error);
    loading.className = "viewer-error";
    loading.textContent = "Failed to load PLY";
  });

  const resize = () => {
    const nextWidth = container.clientWidth;
    const nextHeight = container.clientHeight;
    camera.aspect = nextWidth / nextHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(nextWidth, nextHeight);
    controls.handleResize();
  };

  window.addEventListener("resize", resize);

  const animate = () => {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  };

  animate();
}

function loadPLYWithFallback(loader, urls, onLoad, onError) {
  let index = 0;
  let lastError = null;

  const tryNext = () => {
    const url = urls[index];
    if (!url) {
      onError(lastError);
      return;
    }

    loader.load(
      url,
      onLoad,
      undefined,
      (error) => {
        lastError = error;
        index += 1;
        tryNext();
      }
    );
  };

  tryNext();
}

mountIcons();
setupCitationTabs();

const viewers = Array.from(document.querySelectorAll(".ply-viewer"));
viewers.forEach((container) => {
  const loading = document.createElement("div");
  loading.className = "viewer-loading";
  loading.innerHTML = '<div class="spinner"></div>';
  container.appendChild(loading);
});

loadPLYLibraries()
  .then((libraries) => {
    viewers.forEach((container) => {
      container.querySelector(".viewer-loading")?.remove();
      setupPLYViewer(container, libraries);
    });
  })
  .catch((error) => {
    console.error("3D viewer libraries failed to load:", error);
    viewers.forEach((container) => showViewerError(container, "Failed to load 3D viewer"));
  });
