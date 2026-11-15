const icons = {
  github:
    '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5a12 12 0 0 0-3.8 23.38c.6.1.82-.26.82-.58v-2.02c-3.34.72-4.04-1.42-4.04-1.42-.55-1.38-1.34-1.75-1.34-1.75-1.08-.74.08-.72.08-.72 1.2.08 1.84 1.24 1.84 1.24 1.08 1.82 2.82 1.3 3.5 1 .1-.78.42-1.3.76-1.6-2.66-.3-5.46-1.34-5.46-5.94 0-1.32.46-2.38 1.24-3.22-.12-.3-.54-1.52.12-3.18 0 0 1.02-.32 3.3 1.24a11.42 11.42 0 0 1 6 0c2.28-1.56 3.3-1.24 3.3-1.24.66 1.66.24 2.88.12 3.18.78.84 1.24 1.9 1.24 3.22 0 4.62-2.8 5.62-5.48 5.92.44.38.82 1.12.82 2.26v3.36c0 .32.22.7.82.58A12 12 0 0 0 12 .5Z"/></svg>',
  database:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3"/></svg>',
  copy: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>',
  check:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>'
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
  PlainText:
    "El Ghazouali, S., Venturi, N., Rueegsegger, M., & Michelucci, U. (2026). SyMTRS: Benchmark Multi-Task Synthetic Dataset for Depth, Domain Adaptation and Super-Resolution in Aerial Imagery. arXiv preprint arXiv:2604.21801.",
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

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ------------------------------------------------------------------ icons */

function mountIcons() {
  document.querySelectorAll("[data-icon]").forEach((node) => {
    const icon = icons[node.dataset.icon];
    if (icon) node.innerHTML = icon;
  });
}

/* ------------------------------------------------------------------ nav */

function setupNav() {
  const nav = document.getElementById("nav");
  const progress = document.getElementById("scrollProgress");
  const menuBtn = document.getElementById("menuBtn");
  const links = document.getElementById("navLinks");

  const onScroll = () => {
    const top = window.scrollY || document.documentElement.scrollTop;
    nav.classList.toggle("is-scrolled", top > 10);
    const height = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = `${height > 0 ? Math.min(100, (top / height) * 100) : 0}%`;
  };

  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  if (menuBtn && links) {
    menuBtn.addEventListener("click", () => {
      const open = links.classList.toggle("is-open");
      menuBtn.setAttribute("aria-expanded", String(open));
    });
    links.addEventListener("click", (event) => {
      if (event.target.closest("a")) {
        links.classList.remove("is-open");
        menuBtn.setAttribute("aria-expanded", "false");
      }
    });
    document.addEventListener("click", (event) => {
      if (!links.contains(event.target) && !menuBtn.contains(event.target)) {
        links.classList.remove("is-open");
        menuBtn.setAttribute("aria-expanded", "false");
      }
    });
  }
}

/* ------------------------------------------------------------------ reveal + counters */

function setupReveal() {
  const targets = document.querySelectorAll(
    ".hero-copy, .hero-visual, .section-head, .glass, .task-shell, .cloud-card, .marquee"
  );
  if (!targets.length) return;

  targets.forEach((node) => node.classList.add("reveal"));

  if (reduceMotion || !("IntersectionObserver" in window)) {
    targets.forEach((node) => node.classList.add("is-in"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-in");
          observer.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
  );
  targets.forEach((node) => observer.observe(node));
}

function setupCounters() {
  const counters = document.querySelectorAll("[data-count]");
  if (!counters.length) return;

  const animate = (node) => {
    const target = Number(node.dataset.count);
    if (!Number.isFinite(target) || reduceMotion) return;
    const duration = 1400;
    const start = performance.now();
    const step = (now) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      node.textContent = Math.round(target * eased).toLocaleString("en-US");
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  };

  if (!("IntersectionObserver" in window)) {
    counters.forEach(animate);
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animate(entry.target);
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.6 }
  );
  counters.forEach((node) => observer.observe(node));
}

/* ------------------------------------------------------------------ before/after sliders */

function setupComparisons() {
  document.querySelectorAll(".ba").forEach((figure) => {
    const frame = figure.querySelector(".ba-frame");
    const range = figure.querySelector(".ba-range");
    if (!frame || !range) return;

    let pos = Number(range.value) || 50;
    let direction = 1;
    let auto = !reduceMotion;
    let last = performance.now();
    let idleTimer = 0;

    const place = (value) => {
      pos = value;
      frame.style.setProperty("--pos", `${value.toFixed(2)}%`);
      range.value = String(Math.round(value));
    };

    place(pos);

    const pause = () => {
      auto = false;
      clearTimeout(idleTimer);
    };

    const resumeLater = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => {
        if (!reduceMotion) {
          auto = true;
          last = performance.now();
        }
      }, 3500);
    };

    range.addEventListener("pointerdown", pause);
    range.addEventListener("input", () => {
      pause();
      pos = Number(range.value);
      frame.style.setProperty("--pos", `${pos}%`);
      resumeLater();
    });

    const tick = (now) => {
      const delta = Math.min(64, now - last);
      last = now;
      // Hidden tab panels have no offsetParent; their sliders cost nothing.
      if (auto && frame.offsetParent) {
        pos += direction * delta * 0.011;
        if (pos >= 78) {
          pos = 78;
          direction = -1;
        } else if (pos <= 22) {
          pos = 22;
          direction = 1;
        }
        place(pos);
      }
      requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
  });
}

/* ------------------------------------------------------------------ task tabs */

function setupTabs() {
  const tabs = Array.from(document.querySelectorAll(".task-tab"));
  const panels = Array.from(document.querySelectorAll(".task-panel"));
  if (!tabs.length) return;

  const show = (id) => {
    tabs.forEach((tab) => {
      const active = tab.dataset.target === id;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    panels.forEach((panel) => {
      const active = panel.id === id;
      panel.classList.toggle("is-active", active);
      panel.hidden = !active;
    });
  };

  tabs.forEach((tab, index) => {
    tab.tabIndex = tab.classList.contains("is-active") ? 0 : -1;
    tab.addEventListener("click", () => show(tab.dataset.target));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      event.preventDefault();
      const next = (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      tabs[next].focus();
      show(tabs[next].dataset.target);
    });
  });
}

/* ------------------------------------------------------------------ 3D point clouds */

async function mountCloud(container) {
  const status = container.querySelector(".cloud-status");
  const fail = (message) => {
    container.classList.add("is-failed");
    if (status) status.lastElementChild.textContent = message;
  };

  let THREE;
  let TrackballControls;
  let PLYLoader;
  try {
    THREE = await import("three");
    ({ TrackballControls } = await import("three/addons/controls/TrackballControls.js"));
    ({ PLYLoader } = await import("three/addons/loaders/PLYLoader.js"));
  } catch (error) {
    console.error("3D libraries failed to load:", error);
    fail("3D viewer unavailable");
    return;
  }

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: "high-performance"
    });
  } catch (error) {
    console.error("WebGL unavailable:", error);
    fail("WebGL unavailable");
    return;
  }

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1e6);
  camera.position.set(0, 60, 220);

  const controls = new TrackballControls(camera, renderer.domElement);
  controls.rotateSpeed = 2.4;
  controls.zoomSpeed = 1.1;
  controls.panSpeed = 0.6;
  controls.noPan = true;
  controls.staticMoving = true;
  controls.dynamicDampingFactor = 0.18;

  let points = null;
  let radius = 100;
  let visible = true;
  let lastInteraction = performance.now();

  const touched = () => {
    lastInteraction = performance.now();
  };
  renderer.domElement.addEventListener("pointerdown", touched);
  renderer.domElement.addEventListener("wheel", touched, { passive: true });

  const resize = () => {
    const width = container.clientWidth || 1;
    const height = container.clientHeight || 1;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    controls.handleResize();
  };

  if ("ResizeObserver" in window) {
    new ResizeObserver(resize).observe(container);
  } else {
    window.addEventListener("resize", resize);
  }
  resize();

  const loader = new PLYLoader();
  loader.load(
    container.dataset.url,
    (geometry) => {
      geometry.computeBoundingSphere();
      radius = geometry.boundingSphere ? geometry.boundingSphere.radius : 100;
      geometry.center();

      const material = new THREE.PointsMaterial({
        size: radius * 0.0042,
        vertexColors: true,
        sizeAttenuation: true
      });
      points = new THREE.Points(geometry, material);
      scene.add(points);

      camera.near = radius * 0.01;
      camera.far = radius * 20;
      // A three-quarter view: viewed head-on, the depth relief foreshortens into the
      // original photo and the viewer stops looking like 3D at all. The camera distance is
      // solved per point — a point fits when its planar offset is within the frustum at its
      // own depth — and the same solve is run at the auto-oscillation extremes so the sweep
      // cannot crop the cloud at the card's edges.
      const dir = new THREE.Vector3(0.62, 0.3, 0.72).normalize();
      const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), dir).normalize();
      const up = new THREE.Vector3().crossVectors(dir, right).normalize();
      const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
      const positions = geometry.attributes.position;
      const stride = Math.max(1, Math.floor(positions.count / 24000));
      const requiredDistance = (yaw) => {
        const c = Math.cos(yaw);
        const s = Math.sin(yaw);
        let needed = 0;
        for (let i = 0; i < positions.count; i += stride) {
          const x = positions.getX(i) * c + positions.getZ(i) * s;
          const y = positions.getY(i);
          const z = -positions.getX(i) * s + positions.getZ(i) * c;
          const depth = x * dir.x + y * dir.y + z * dir.z;
          const px = Math.abs(x * right.x + y * right.y + z * right.z);
          const py = Math.abs(x * up.x + y * up.y + z * up.z);
          needed = Math.max(needed, py / tanHalfFov + depth, px / (tanHalfFov * camera.aspect) + depth);
        }
        return needed;
      };
      const distance = Math.max(
        requiredDistance(0),
        requiredDistance(0.18),
        requiredDistance(-0.18)
      );
      camera.position.copy(dir.multiplyScalar(distance * 1.04));
      camera.updateProjectionMatrix();
      camera.lookAt(0, 0, 0);
      controls.target.set(0, 0, 0);
      controls.update();

      container.classList.add("is-ready");
      if (status) status.remove();
    },
    undefined,
    (error) => {
      console.error("PLY load failed:", error);
      fail("Could not load the point cloud");
    }
  );

  if ("IntersectionObserver" in window) {
    new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          visible = entry.isIntersecting;
        });
      },
      { rootMargin: "120px" }
    ).observe(container);
  }

  resize();

  const animate = (now) => {
    if (visible) {
      if (points && !reduceMotion && now - lastInteraction > 2400) {
        // A small oscillation, not a full turn: the framing was fitted at zero yaw, and a
        // full rotation would eventually swing the cloud out of the card.
        points.rotation.y = Math.sin(now * 0.00035) * 0.18;
      }
      controls.update();
      renderer.render(scene, camera);
    }
    requestAnimationFrame(animate);
  };
  requestAnimationFrame(animate);
}

function setupClouds() {
  const viewers = document.querySelectorAll(".cloud-viewer");
  if (!viewers.length) return;

  if (!("IntersectionObserver" in window)) {
    viewers.forEach(mountCloud);
    return;
  }

  // The clouds are a few megabytes each; building one only when it approaches the
  // viewport keeps the hero from paying for the two below it.
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          observer.unobserve(entry.target);
          mountCloud(entry.target);
        }
      });
    },
    { rootMargin: "400px 0px" }
  );
  viewers.forEach((viewer) => observer.observe(viewer));
}

/* ------------------------------------------------------------------ citation */

let activeCitation = "BibTeX";

function setupCitation() {
  const output = document.getElementById("citationText");
  const copyButton = document.getElementById("copyCitation");
  const copyLabel = document.getElementById("copyLabel");
  const tabs = Array.from(document.querySelectorAll(".tab"));
  if (!output || !copyButton) return;

  const render = () => {
    output.textContent = citations[activeCitation];
    copyLabel.textContent = `Copy ${activeCitation === "PlainText" ? "text" : activeCitation}`;
    copyButton.querySelector("[data-icon]").innerHTML = icons.copy;
    tabs.forEach((tab) => tab.classList.toggle("is-active", tab.dataset.tab === activeCitation));
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

/* ------------------------------------------------------------------ boot */

mountIcons();
setupNav();
setupReveal();
setupCounters();
setupComparisons();
setupTabs();
setupClouds();
setupCitation();
