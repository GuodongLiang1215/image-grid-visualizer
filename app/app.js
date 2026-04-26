(function () {
  const paths = {
    classifications: "../data/classification_results.json",
    tsne: "../data/tsne_results.json",
    words: "../data/words.txt",
    imageBase: "../data/images/"
  };

  const state = {
    data: [],
    visible: [],
    selectedIds: new Set(),
    activeDatum: null,
    filters: {
      predicted: "all",
      source: "all",
      search: "",
      confidence: 0,
      match: "all"
    },
    colorBy: "predicted",
    mainView: "grid",
    interactionMode: "pan",
    grid: {
      groupBy: "embedding",
      sortBy: "confidence-desc",
      tileSize: 112
    },
    transform: d3.zoomIdentity
  };

  const color = d3.scaleOrdinal(d3.schemeTableau10);
  const fmtPercent = d3.format(".1%");
  const fmtDate = d3.timeFormat("%Y-%m-%d");

  const el = {
    status: d3.select("#load-status"),
    classFilter: d3.select("#class-filter"),
    sourceFilter: d3.select("#source-filter"),
    colorMode: d3.select("#color-mode"),
    matchFilter: d3.select("#match-filter"),
    gridGroup: d3.select("#grid-group"),
    gridSort: d3.select("#grid-sort"),
    tileSize: d3.select("#tile-size"),
    tileSizeLabel: d3.select("#tile-size-label"),
    search: d3.select("#search-input"),
    confidence: d3.select("#confidence-filter"),
    confidenceLabel: d3.select("#confidence-label"),
    reset: d3.select("#reset-view"),
    clear: d3.select("#clear-selection"),
    total: d3.select("#stat-total"),
    visible: d3.select("#stat-visible"),
    selected: d3.select("#stat-selected"),
    avgConfidence: d3.select("#stat-confidence"),
    agreement: d3.select("#stat-agreement"),
    scatterCaption: d3.select("#scatter-caption"),
    barTitle: d3.select("#bar-title"),
    barCaption: d3.select("#bar-caption"),
    mainTitle: d3.select("#main-title"),
    viewGrid: d3.select("#view-grid"),
    viewMap: d3.select("#view-map"),
    chartTools: d3.select(".chart-tools"),
    modePan: d3.select("#mode-pan"),
    modeSelect: d3.select("#mode-select"),
    legend: d3.select("#legend"),
    imageGrid: d3.select("#image-grid"),
    scatter: d3.select("#scatter"),
    barChart: d3.select("#bar-chart"),
    timeline: d3.select("#timeline"),
    tooltip: d3.select("#tooltip"),
    detailsEmpty: d3.select("#details-empty"),
    detailsContent: d3.select("#details-content"),
    detailImage: d3.select("#detail-image"),
    detailName: d3.select("#detail-name"),
    detailClass: d3.select("#detail-class"),
    detailConfidence: d3.select("#detail-confidence"),
    detailSource: d3.select("#detail-source"),
    detailAgreement: d3.select("#detail-agreement"),
    detailTop5: d3.select("#detail-top5"),
    detailDate: d3.select("#detail-date")
  };

  let scatter = {};
  let bar = {};
  let timeline = {};

  init();

  async function init() {
    try {
      const [classificationData, tsneData, wordsText] = await Promise.all([
        d3.json(paths.classifications),
        d3.json(paths.tsne),
        d3.text(paths.words)
      ]);

      const words = parseWords(wordsText);
      state.data = prepareData(classificationData, tsneData, words);
      state.visible = state.data;

      buildControls();
    setupCharts();
      setupGrid();
      setupResizeHandler();
      bindEvents();
      setMainView("grid");
      applyFilters();

      el.status.text(`${state.data.length} images loaded`);
    } catch (error) {
      console.error(error);
      el.status.classed("error", true).text("Data failed to load");
      el.scatter.html(
        '<div class="empty-state" style="padding:16px">Start the local server with <strong>node server.js</strong>, then open <strong>http://localhost:4173/app/</strong>.</div>'
      );
    }
  }

  function parseWords(text) {
    const lookup = new Map();
    text.split(/\r?\n/).forEach((line) => {
      const [id, label] = line.split(/\t/);
      if (id && label) {
        const terms = label.split(",").map((term) => term.trim()).filter(Boolean);
        lookup.set(id, {
          label,
          primary: terms[0] || id,
          terms
        });
      }
    });
    return lookup;
  }

  function prepareData(classifications, tsne, words) {
    if (!Array.isArray(classifications) || !Array.isArray(tsne)) {
      throw new Error("Expected JSON arrays for classification and t-SNE data.");
    }
    if (classifications.length !== tsne.length) {
      throw new Error(`Data length mismatch: ${classifications.length} classifications vs ${tsne.length} t-SNE rows.`);
    }

    return classifications.map((d, index) => {
      const sourceId = String(d.image || "").match(/^(n\d+)_/)?.[1] || "unknown";
      const source = words.get(sourceId) || { label: sourceId, primary: sourceId, terms: [sourceId] };
      const date = new Date(d.date);
      const predicted = d.class_name || "unknown";
      const labelMatch = source.terms.some((term) => normalizeLabel(term) === normalizeLabel(predicted));
      const top5 = Array.isArray(d.top5)
        ? d.top5.map((item) => ({
          className: item.class_name || item.className || "unknown",
          probability: Number(item.probability) || 0
        }))
        : [];
      return {
        id: index,
        image: d.image,
        predicted,
        probability: Number(d.probability) || 0,
        top5,
        sourceId,
        sourceName: source.primary,
        sourceLabel: source.label,
        sourceTerms: source.terms,
        labelMatch,
        labelStatus: labelMatch ? "Text label match" : "Text label differs",
        // Accept either a bare filename ("foo.JPEG") or a path that already
        // includes a folder ("images/foo.JPEG"); strip everything up to and
        // including the last slash so we always end up with just the basename.
        path: `${paths.imageBase}${String(d.path || d.image || "").replace(/\\/g, "/").split("/").pop()}`,
        date,
        dateLabel: Number.isNaN(date.getTime()) ? "Unknown" : `${fmtDate(date)} (simulated)`,
        x: Number(tsne[index][0]),
        y: Number(tsne[index][1])
      };
    });
  }

  function buildControls() {
    const predicted = uniqueSorted(state.data.map((d) => d.predicted));
    const sources = uniqueSorted(state.data.map((d) => `${d.sourceId} - ${d.sourceName}`));

    el.classFilter
      .selectAll("option.class-option")
      .data(predicted)
      .join("option")
      .attr("class", "class-option")
      .attr("value", (d) => d)
      .text((d) => d);

    el.sourceFilter
      .selectAll("option.source-option")
      .data(sources)
      .join("option")
      .attr("class", "source-option")
      .attr("value", (d) => d.split(" - ")[0])
      .text((d) => d);
  }

  function setupCharts() {
    setupScatter();
    setupBarChart();
    setupTimeline();
  }

  function setupGrid() {
    el.imageGrid.style("--tile-size", `${state.grid.tileSize}px`);
  }

  function setupResizeHandler() {
    window.addEventListener("resize", debounce(() => {
      setupCharts();
      renderAll();
    }, 120));
  }

  function setupScatter() {
    el.scatter.html("");
    const bounds = el.scatter.node().getBoundingClientRect();
    const margin = { top: 18, right: 22, bottom: 36, left: 42 };
    const width = Math.max(420, bounds.width || 720);
    const height = Math.max(420, bounds.height || 560);

    const svg = el.scatter.append("svg").attr("viewBox", `0 0 ${width} ${height}`);
    const plot = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const clipId = "scatter-plot-clip";

    svg.append("defs")
      .append("clipPath")
      .attr("id", clipId)
      .attr("clipPathUnits", "userSpaceOnUse")
      .append("rect")
      .attr("x", 0)
      .attr("y", 0)
      .attr("width", innerWidth)
      .attr("height", innerHeight);

    const x = d3.scaleLinear()
      .domain(d3.extent(state.data, (d) => d.x)).nice()
      .range([0, innerWidth]);
    const y = d3.scaleLinear()
      .domain(d3.extent(state.data, (d) => d.y)).nice()
      .range([innerHeight, 0]);

    plot.append("g")
      .attr("class", "axis axis-x")
      .attr("transform", `translate(0,${innerHeight})`);
    plot.append("g").attr("class", "axis axis-y");
    plot.append("g")
      .attr("class", "points")
      .attr("clip-path", `url(#${clipId})`);
    plot.append("g").attr("class", "brush");

    const zoom = d3.zoom()
      .scaleExtent([0.75, 12])
      .translateExtent([[-80, -80], [innerWidth + 80, innerHeight + 80]])
      .filter((event) => event.type === "wheel" || state.interactionMode === "pan")
      .on("zoom", (event) => {
        state.transform = event.transform;
        renderScatter();
      });

    svg.call(zoom).on("dblclick.zoom", null);

    const brush = d3.brush()
      .extent([[0, 0], [innerWidth, innerHeight]])
      .on("end", (event) => {
        if (!event.selection) return;
        const [[x0, y0], [x1, y1]] = event.selection;
        const zx = state.transform.rescaleX(x);
        const zy = state.transform.rescaleY(y);
        state.selectedIds = new Set(
          state.visible
            .filter((d) => {
              const px = zx(d.x);
              const py = zy(d.y);
              return px >= x0 && px <= x1 && py >= y0 && py <= y1;
            })
            .map((d) => d.id)
        );
        plot.select(".brush").call(brush.move, null);
        renderAll();
      });

    plot.select(".brush").call(brush);

    scatter = { svg, plot, x, y, width, height, innerWidth, innerHeight, zoom, brush };
    updateInteractionMode();
  }

  function updateInteractionMode() {
    if (!scatter.plot) return;
    const isSelect = state.interactionMode === "select";
    el.scatter
      .classed("pan-mode", !isSelect)
      .classed("select-mode", isSelect);
    scatter.plot.select(".brush")
      .style("pointer-events", isSelect ? "all" : "none")
      .style("display", isSelect ? null : "none");
  }

  function setupBarChart() {
    el.barChart.html("");
    const bounds = el.barChart.node().getBoundingClientRect();
    const width = Math.max(360, bounds.width || 520);
    const height = Math.max(230, bounds.height || 260);
    const margin = { top: 8, right: 42, bottom: 20, left: 132 };
    const svg = el.barChart.append("svg").attr("viewBox", `0 0 ${width} ${height}`);
    const plot = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    bar = {
      svg,
      plot,
      width,
      height,
      innerWidth: width - margin.left - margin.right,
      innerHeight: height - margin.top - margin.bottom
    };
  }

  function setupTimeline() {
    el.timeline.html("");
    const bounds = el.timeline.node().getBoundingClientRect();
    const width = Math.max(360, bounds.width || 520);
    const height = Math.max(230, bounds.height || 260);
    const margin = { top: 22, right: 20, bottom: 34, left: 20 };
    const svg = el.timeline.append("svg").attr("viewBox", `0 0 ${width} ${height}`);
    const plot = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
    timeline = {
      svg,
      plot,
      width,
      height,
      innerWidth: width - margin.left - margin.right,
      innerHeight: height - margin.top - margin.bottom
    };
  }

  function bindEvents() {
    el.classFilter.on("change", (event) => {
      state.filters.predicted = event.target.value;
      state.selectedIds.clear();
      applyFilters();
    });

    el.sourceFilter.on("change", (event) => {
      state.filters.source = event.target.value;
      state.selectedIds.clear();
      applyFilters();
    });

    el.colorMode.on("change", (event) => {
      state.colorBy = event.target.value;
      renderAll();
    });

    el.matchFilter.on("change", (event) => {
      state.filters.match = event.target.value;
      state.selectedIds.clear();
      applyFilters();
    });

    el.gridGroup.on("change", (event) => {
      state.grid.groupBy = event.target.value;
      renderAll();
    });

    el.gridSort.on("change", (event) => {
      state.grid.sortBy = event.target.value;
      renderAll();
    });

    el.tileSize.on("input", (event) => {
      state.grid.tileSize = Number(event.target.value);
      el.tileSizeLabel.text(`${state.grid.tileSize}px`);
      el.imageGrid.style("--tile-size", `${state.grid.tileSize}px`);
      if (state.grid.groupBy === "embedding") renderGrid();
    });

    el.search.on("input", debounce((event) => {
      state.filters.search = event.target.value.trim().toLowerCase();
      state.selectedIds.clear();
      applyFilters();
    }, 120));

    el.confidence.on("input", (event) => {
      state.filters.confidence = Number(event.target.value) / 100;
      el.confidenceLabel.text(`${event.target.value}%`);
      state.selectedIds.clear();
      applyFilters();
    });

    el.reset.on("click", () => {
      state.transform = d3.zoomIdentity;
      scatter.svg.transition().duration(250).call(scatter.zoom.transform, d3.zoomIdentity);
    });

    el.clear.on("click", () => {
      state.selectedIds.clear();
      state.activeDatum = null;
      renderAll();
    });

    el.modePan.on("click", () => setInteractionMode("pan"));
    el.modeSelect.on("click", () => setInteractionMode("select"));
    el.viewGrid.on("click", () => setMainView("grid"));
    el.viewMap.on("click", () => setMainView("map"));
  }

  function setMainView(view) {
    state.mainView = view;
    el.viewGrid.classed("active", view === "grid");
    el.viewMap.classed("active", view === "map");
    el.imageGrid.classed("hidden", view !== "grid");
    el.scatter.classed("hidden", view !== "map");
    el.chartTools.classed("hidden", view !== "map");
    el.mainTitle.text(view === "grid" ? "Image Grid" : "t-SNE Projection");
    renderLegend();
  }

  function setInteractionMode(mode) {
    state.interactionMode = mode;
    el.modePan.classed("active", mode === "pan");
    el.modeSelect.classed("active", mode === "select");
    updateInteractionMode();
  }

  function applyFilters() {
    state.visible = state.data.filter((d) => {
      const matchesClass = state.filters.predicted === "all" || d.predicted === state.filters.predicted;
      const matchesSource = state.filters.source === "all" || d.sourceId === state.filters.source;
      const matchesConfidence = d.probability >= state.filters.confidence;
      const matchesStatus = state.filters.match === "all"
        || (state.filters.match === "match" && d.labelMatch)
        || (state.filters.match === "mismatch" && !d.labelMatch);
      const haystack = `${d.image} ${d.predicted} ${d.sourceId} ${d.sourceLabel}`.toLowerCase();
      const matchesSearch = !state.filters.search || haystack.includes(state.filters.search);
      return matchesClass && matchesSource && matchesConfidence && matchesStatus && matchesSearch;
    });
    renderAll();
  }

  function renderAll() {
    renderSummary();
    renderLegend();
    renderGrid();
    renderScatter();
    renderBars();
    renderTimeline();
    renderDetails();
  }

  function renderSummary() {
    const selected = selectedData();
    const confidenceSource = selected.length ? selected : state.visible;
    const avg = d3.mean(confidenceSource, (d) => d.probability) || 0;
    const agreement = d3.mean(confidenceSource, (d) => d.labelMatch ? 1 : 0) || 0;
    el.total.text(state.data.length);
    el.visible.text(state.visible.length);
    el.selected.text(selected.length);
    el.avgConfidence.text(fmtPercent(avg));
    el.agreement.text(fmtPercent(agreement));
  }

  function renderLegend() {
    const top = labelCounts(state.visible).slice(0, 8);
    color.domain(top.map((d) => d.key));
    updateChartCopy();
    el.legend
      .selectAll(".legend-item")
      .data(top, (d) => d.key)
      .join("div")
      .attr("class", "legend-item")
      .html((d) => `<span class="swatch" style="background:${color(d.key)}"></span><span>${escapeHtml(d.key)}</span>`);
  }

  function updateChartCopy() {
    if (state.mainView === "grid") {
      if (state.grid.groupBy === "embedding") {
        el.scatterCaption.text("Images are assigned to the nearest open grid cell from their t-SNE position.");
      } else {
        const group = groupLabel(state.grid.groupBy).toLowerCase();
        el.scatterCaption.text(`Images grouped by ${group}, sorted with ${sortLabel(state.grid.sortBy).toLowerCase()}.`);
      }
    } else if (state.colorBy === "source") {
      el.scatterCaption.text("Colored by dataset label. Pan with drag, zoom with wheel.");
    } else if (state.colorBy === "agreement") {
      el.scatterCaption.text("Colored by prediction/source-label agreement. Pan with drag, zoom with wheel.");
    } else {
      el.scatterCaption.text("Colored by model prediction. Pan with drag, zoom with wheel.");
    }

    if (state.colorBy === "source") {
      el.barTitle.text("Dataset Label Distribution");
      el.barCaption.text("Top source labels in the current filtered view.");
    } else if (state.colorBy === "agreement") {
      el.barTitle.text("Prediction Check");
      el.barCaption.text("Prediction/source-label agreement in the current filtered view.");
    } else {
      el.barTitle.text("Model Prediction Distribution");
      el.barCaption.text("Top predicted labels in the current filtered view.");
    }
  }

  function renderScatter() {
    const zx = state.transform.rescaleX(scatter.x);
    const zy = state.transform.rescaleY(scatter.y);

    scatter.plot.select(".axis-x").call(d3.axisBottom(zx).ticks(7));
    scatter.plot.select(".axis-y").call(d3.axisLeft(zy).ticks(7));

    const points = scatter.plot.select(".points")
      .selectAll("circle")
      .data(state.visible, (d) => d.id);

    points.join(
      (enter) => enter.append("circle")
        .attr("class", "point")
        .attr("r", 4.5)
        .on("mouseenter", showTooltip)
        .on("mousemove", moveTooltip)
        .on("mouseleave", hideTooltip)
        .on("click", (event, d) => {
          event.stopPropagation();
          state.activeDatum = d;
          state.selectedIds = new Set([d.id]);
          renderAll();
        }),
      (update) => update,
      (exit) => exit.remove()
    )
      .attr("cx", (d) => zx(d.x))
      .attr("cy", (d) => zy(d.y))
      .attr("fill", (d) => color(labelForMode(d)))
      .classed("selected", (d) => state.selectedIds.has(d.id))
      .classed("dimmed", (d) => state.selectedIds.size > 0 && !state.selectedIds.has(d.id));
  }

  function renderGrid() {
    el.imageGrid.classed("embedding-mode", state.grid.groupBy === "embedding");
    if (state.grid.groupBy === "embedding") {
      renderEmbeddingGrid();
      return;
    }

    el.imageGrid.selectAll(".embedding-grid-shell").remove();
    const groups = groupedGridData();

    el.imageGrid
      .selectAll(".grid-empty")
      .data(groups.length ? [] : [null])
      .join("div")
      .attr("class", "grid-empty empty-state")
      .text("No images match the current filters.");

    const sections = el.imageGrid
      .selectAll(".grid-section")
      .data(groups, (d) => d.key);

    const sectionsEnter = sections.enter()
      .append("section")
      .attr("class", "grid-section");

    const header = sectionsEnter.append("div").attr("class", "grid-section-header");
    header.append("h3").attr("class", "grid-section-title");
    header.append("span").attr("class", "grid-section-count");
    sectionsEnter.append("div").attr("class", "tile-grid");

    const merged = sectionsEnter.merge(sections);

    merged.select(".grid-section-title").text((d) => d.key);
    merged.select(".grid-section-count").text((d) => `${d.values.length} images`);

    merged.each(function (group) {
      const tiles = d3.select(this)
        .select(".tile-grid")
        .selectAll(".image-tile")
        .data(group.values, (d) => d.id);

      const tilesEnter = tiles.enter()
        .append("button")
        .attr("type", "button")
        .attr("class", "image-tile")
        .on("mouseenter", showTooltip)
        .on("mousemove", moveTooltip)
        .on("mouseleave", hideTooltip)
        .on("click", (event, d) => {
          event.stopPropagation();
          state.activeDatum = d;
          state.selectedIds = new Set([d.id]);
          renderAll();
        });

      tilesEnter.append("div")
        .attr("class", "tile-image-wrap")
        .append("img")
        .attr("loading", "lazy")
        .attr("alt", (d) => d.image);

      const meta = tilesEnter.append("div").attr("class", "tile-meta");
      meta.append("div").attr("class", "tile-title");
      meta.append("div").attr("class", "tile-subtitle");
      meta.append("span").attr("class", "tile-badge");

      const mergedTiles = tilesEnter.merge(tiles);

      mergedTiles
        .style("border-top-color", (d) => color(labelForMode(d)))
        .classed("selected", (d) => state.selectedIds.has(d.id))
        .classed("match", (d) => d.labelMatch)
        .classed("mismatch", (d) => !d.labelMatch);

      mergedTiles.select("img")
        .attr("src", (d) => d.path)
        .attr("alt", (d) => d.image);

      mergedTiles.select(".tile-title")
        .text((d) => d.predicted);

      mergedTiles.select(".tile-subtitle")
        .text((d) => d.sourceName);

      mergedTiles.select(".tile-badge")
        .classed("match", (d) => d.labelMatch)
        .classed("mismatch", (d) => !d.labelMatch)
        .text((d) => `${fmtPercent(d.probability)} - ${d.labelMatch ? "match" : "review"}`);

      tiles.exit().remove();
    });

    sections.exit().remove();
  }

  function renderEmbeddingGrid() {
    el.imageGrid.selectAll(".grid-section").remove();
    const layout = embeddingGridLayout(state.visible);

    el.imageGrid
      .selectAll(".grid-empty")
      .data(layout.items.length ? [] : [null])
      .join("div")
      .attr("class", "grid-empty empty-state")
      .text("No images match the current filters.");

    const shell = el.imageGrid
      .selectAll(".embedding-grid-shell")
      .data(layout.items.length ? [layout] : []);

    const shellEnter = shell.enter()
      .append("section")
      .attr("class", "embedding-grid-shell");

    const header = shellEnter.append("div").attr("class", "grid-section-header");
    header.append("h3").attr("class", "grid-section-title");
    header.append("span").attr("class", "grid-section-count");
    shellEnter.append("div").attr("class", "embedding-grid");

    const mergedShell = shellEnter.merge(shell);
    mergedShell.select(".grid-section-title").text("Embedding grid");
    mergedShell.select(".grid-section-count")
      .text(`${layout.items.length} images · ${layout.columns} x ${layout.rows}`);

    const grid = mergedShell.select(".embedding-grid")
      .style("--embedding-columns", layout.columns)
      .style("--embedding-rows", layout.rows)
      .style("--embedding-cell-size", `${state.grid.tileSize}px`);

    const tiles = grid
      .selectAll(".embedding-tile")
      .data(layout.items, (d) => d.datum.id);

    const tilesEnter = tiles.enter()
      .append("button")
      .attr("type", "button")
      .attr("class", "embedding-tile")
      .on("mouseenter", (event, d) => showTooltip(event, d.datum))
      .on("mousemove", moveTooltip)
      .on("mouseleave", hideTooltip)
      .on("click", (event, d) => {
        event.stopPropagation();
        state.activeDatum = d.datum;
        state.selectedIds = new Set([d.datum.id]);
        renderAll();
      });

    tilesEnter.append("img")
      .attr("loading", "lazy")
      .attr("alt", (d) => d.datum.image);
    tilesEnter.append("span").attr("class", "embedding-tile-badge");

    const mergedTiles = tilesEnter.merge(tiles);

    mergedTiles
      .style("grid-column", (d) => d.column + 1)
      .style("grid-row", (d) => d.row + 1)
      .style("border-color", (d) => color(labelForMode(d.datum)))
      .classed("selected", (d) => state.selectedIds.has(d.datum.id))
      .classed("match", (d) => d.datum.labelMatch)
      .classed("mismatch", (d) => !d.datum.labelMatch);

    mergedTiles.select("img")
      .attr("src", (d) => d.datum.path)
      .attr("alt", (d) => d.datum.image);

    mergedTiles.select(".embedding-tile-badge")
      .text((d) => fmtPercent(d.datum.probability));

    tiles.exit().remove();
    shell.exit().remove();
  }

  function renderBars() {
    const counts = labelCounts(state.visible).slice(0, 10);
    const x = d3.scaleLinear()
      .domain([0, d3.max(counts, (d) => d.value) || 1])
      .range([0, bar.innerWidth]);
    const y = d3.scaleBand()
      .domain(counts.map((d) => d.key))
      .range([0, bar.innerHeight])
      .padding(0.22);

    const row = bar.plot.selectAll("g.bar-row")
      .data(counts, (d) => d.key)
      .join(
        (enter) => {
          const g = enter.append("g").attr("class", "bar-row");
          g.append("rect").attr("class", "bar").attr("height", y.bandwidth());
          g.append("text").attr("class", "bar-label").attr("x", -8).attr("text-anchor", "end");
          g.append("text").attr("class", "bar-value").attr("x", 8);
          return g;
        },
        (update) => update,
        (exit) => exit.remove()
      )
      .attr("transform", (d) => `translate(0,${y(d.key)})`);

    row.select("rect")
      .attr("width", (d) => x(d.value))
      .attr("fill", (d) => color(d.key))
      .on("click", (event, d) => {
        if (state.colorBy === "predicted") {
          state.filters.predicted = d.key;
          el.classFilter.property("value", d.key);
        } else if (state.colorBy === "source") {
          const source = state.visible.find((item) => item.sourceName === d.key);
          if (source) {
            state.filters.source = source.sourceId;
            el.sourceFilter.property("value", source.sourceId);
          }
        }
        state.selectedIds.clear();
        applyFilters();
      });

    row.select(".bar-label")
      .attr("y", y.bandwidth() / 2 + 4)
      .text((d) => truncate(d.key, 20));

    row.select(".bar-value")
      .attr("x", (d) => x(d.value) + 8)
      .attr("y", y.bandwidth() / 2 + 4)
      .text((d) => d.value);
  }

  function renderTimeline() {
    const selected = selectedData();
    const data = (selected.length ? selected : state.visible)
      .filter((d) => !Number.isNaN(d.date.getTime()))
      .sort((a, b) => a.date - b.date)
      .slice(0, 80);

    timeline.plot.selectAll("*").remove();

    if (!data.length) {
      timeline.plot.append("text")
        .attr("x", 4)
        .attr("y", 24)
        .attr("fill", "#667085")
        .text("No dated images in the current view.");
      return;
    }

    const x = d3.scaleTime()
      .domain(d3.extent(data, (d) => d.date))
      .range([0, timeline.innerWidth]);
    const y = d3.scalePoint()
      .domain(data.map((d) => d.id))
      .range([16, timeline.innerHeight - 24])
      .padding(0.5);

    timeline.plot.append("g")
      .attr("class", "axis")
      .attr("transform", `translate(0,${timeline.innerHeight - 20})`)
      .call(d3.axisBottom(x).ticks(5));

    timeline.plot.selectAll("image")
      .data(data, (d) => d.id)
      .join("image")
      .attr("class", "timeline-thumb")
      .attr("href", (d) => d.path)
      .attr("x", (d) => x(d.date) - 11)
      .attr("y", (d) => y(d.id) - 11)
      .attr("width", 22)
      .attr("height", 22)
      .attr("preserveAspectRatio", "xMidYMid meet")
      .on("mouseenter", showTooltip)
      .on("mousemove", moveTooltip)
      .on("mouseleave", hideTooltip)
      .on("click", (event, d) => {
        event.stopPropagation();
        state.activeDatum = d;
        state.selectedIds = new Set([d.id]);
        renderAll();
      });
  }

  function renderDetails() {
    const selected = selectedData();
    const datum = state.activeDatum && state.visible.some((d) => d.id === state.activeDatum.id)
      ? state.activeDatum
      : selected[0];

    if (!datum) {
      el.detailsEmpty.classed("hidden", false);
      el.detailsContent.classed("hidden", true);
      return;
    }

    el.detailsEmpty.classed("hidden", true);
    el.detailsContent.classed("hidden", false);
    el.detailImage.attr("src", datum.path).attr("alt", datum.image);
    el.detailName.text(datum.image);
    el.detailClass.text(datum.predicted);
    el.detailConfidence.text(fmtPercent(datum.probability));
    el.detailSource.text(`${datum.sourceId} - ${datum.sourceLabel}`);
    el.detailAgreement.text(datum.labelStatus);
    el.detailTop5.text(top5Label(datum));
    el.detailDate.text(datum.dateLabel);
  }

  function showTooltip(event, d) {
    el.tooltip
      .classed("hidden", false)
      .html(`
        <img src="${d.path}" alt="">
        <strong>${escapeHtml(d.image)}</strong><br>
        ImageNet prediction: ${escapeHtml(d.predicted)}<br>
        Confidence: ${fmtPercent(d.probability)}<br>
        Tiny ImageNet label: ${escapeHtml(d.sourceName)}<br>
        Label comparison: ${escapeHtml(d.labelStatus)}<br>
        Date: ${escapeHtml(d.dateLabel)}
      `);
    moveTooltip(event);
  }

  function moveTooltip(event) {
    const pad = 16;
    const node = el.tooltip.node();
    const box = node.getBoundingClientRect();
    const x = Math.min(window.innerWidth - box.width - pad, event.clientX + pad);
    const y = Math.min(window.innerHeight - box.height - pad, event.clientY + pad);
    el.tooltip.style("left", `${Math.max(pad, x)}px`).style("top", `${Math.max(pad, y)}px`);
  }

  function hideTooltip() {
    el.tooltip.classed("hidden", true);
  }

  function selectedData() {
    if (!state.selectedIds.size) return [];
    return state.visible.filter((d) => state.selectedIds.has(d.id));
  }

  function embeddingGridLayout(data) {
    if (!data.length) return { columns: 1, rows: 1, items: [] };

    const bounds = el.imageGrid.node().getBoundingClientRect();
    const availableWidth = Math.max(320, bounds.width - 32);
    const cell = Math.max(64, state.grid.tileSize);
    const columns = Math.max(1, Math.floor(availableWidth / cell));
    const rows = Math.ceil(data.length / columns);
    const xExtent = d3.extent(data, (d) => d.x);
    const yExtent = d3.extent(data, (d) => d.y);
    const xSpan = xExtent[1] - xExtent[0] || 1;
    const ySpan = yExtent[1] - yExtent[0] || 1;
    const occupied = new Set();
    const cells = d3.range(columns * rows).map((index) => ({
      column: index % columns,
      row: Math.floor(index / columns)
    }));

    const sorted = data.slice().sort((a, b) => {
      const ay = normalizedGridRow(a, yExtent, ySpan, rows);
      const by = normalizedGridRow(b, yExtent, ySpan, rows);
      const ax = normalizedGridColumn(a, xExtent, xSpan, columns);
      const bx = normalizedGridColumn(b, xExtent, xSpan, columns);
      return d3.ascending(ay, by) || d3.ascending(ax, bx) || compareForGrid(a, b);
    });

    const items = sorted.map((datum) => {
      const targetColumn = normalizedGridColumn(datum, xExtent, xSpan, columns);
      const targetRow = normalizedGridRow(datum, yExtent, ySpan, rows);
      const best = cells
        .filter((cellInfo) => !occupied.has(cellKey(cellInfo.column, cellInfo.row)))
        .sort((a, b) => {
          const da = gridDistance(a, targetColumn, targetRow);
          const db = gridDistance(b, targetColumn, targetRow);
          return d3.ascending(da, db) || d3.ascending(a.row, b.row) || d3.ascending(a.column, b.column);
        })[0];

      occupied.add(cellKey(best.column, best.row));
      return { datum, column: best.column, row: best.row };
    });

    return { columns, rows, items };
  }

  function normalizedGridColumn(d, extent, span, columns) {
    return Math.max(0, Math.min(columns - 1, Math.round(((d.x - extent[0]) / span) * (columns - 1))));
  }

  function normalizedGridRow(d, extent, span, rows) {
    return Math.max(0, Math.min(rows - 1, Math.round(((extent[1] - d.y) / span) * (rows - 1))));
  }

  function gridDistance(cell, targetColumn, targetRow) {
    const dx = cell.column - targetColumn;
    const dy = cell.row - targetRow;
    return dx * dx + dy * dy;
  }

  function cellKey(column, row) {
    return `${column}:${row}`;
  }

  function groupedGridData() {
    const groups = d3.groups(state.visible, gridGroupValue)
      .map(([key, values]) => ({
        key,
        values: values.slice().sort(compareForGrid)
      }));

    return groups.sort((a, b) => {
      if (state.grid.groupBy === "none") return 0;
      return d3.ascending(a.key, b.key);
    });
  }

  function gridGroupValue(d) {
    if (state.grid.groupBy === "predicted") return d.predicted;
    if (state.grid.groupBy === "agreement") return d.labelStatus;
    if (state.grid.groupBy === "none") return "All images";
    return d.sourceName;
  }

  function compareForGrid(a, b) {
    if (state.grid.sortBy === "confidence-asc") return d3.ascending(a.probability, b.probability);
    if (state.grid.sortBy === "date-asc") return d3.ascending(a.date, b.date);
    if (state.grid.sortBy === "date-desc") return d3.descending(a.date, b.date);
    if (state.grid.sortBy === "name-asc") return d3.ascending(a.image, b.image);
    return d3.descending(a.probability, b.probability);
  }

  function labelCounts(data) {
    return Array.from(d3.rollup(data, (v) => v.length, labelForMode), ([key, value]) => ({ key, value }))
      .sort((a, b) => d3.descending(a.value, b.value) || d3.ascending(a.key, b.key));
  }

  function labelForMode(d) {
    if (state.colorBy === "source") return d.sourceName;
    if (state.colorBy === "agreement") return d.labelStatus;
    return d.predicted;
  }

  function normalizeLabel(value) {
    return String(value)
      .toLowerCase()
      .replace(/_/g, " ")
      .replace(/-/g, " ")
      .replace(/[^a-z0-9 ]/g, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function groupLabel(value) {
    if (value === "embedding") return "embedding position";
    if (value === "predicted") return "model prediction";
    if (value === "agreement") return "prediction check";
    if (value === "none") return "a single grid";
    return "dataset label";
  }

  function sortLabel(value) {
    if (value === "confidence-asc") return "confidence low to high";
    if (value === "date-asc") return "date old to new";
    if (value === "date-desc") return "date new to old";
    if (value === "name-asc") return "filename A to Z";
    return "confidence high to low";
  }

  function uniqueSorted(values) {
    return Array.from(new Set(values)).sort(d3.ascending);
  }

  function debounce(fn, delay) {
    let timer;
    return function (...args) {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => fn.apply(this, args), delay);
    };
  }

  function truncate(value, max) {
    const text = String(value);
    return text.length > max ? `${text.slice(0, max - 1)}...` : text;
  }

  function top5Label(datum) {
    if (!datum.top5.length) return "Not available in this data file";
    return datum.top5
      .slice(0, 5)
      .map((item) => `${item.className} ${fmtPercent(item.probability)}`)
      .join(", ");
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
})();
