(() => {
  "use strict";

  const loader = document.currentScript;
  if (!loader) {
    return;
  }

  const plots = Array.from(document.querySelectorAll(".skfolio-lazy-plot"));
  if (!plots.length) {
    return;
  }

  let plotlyPromise;

  function loadPlotly() {
    if (window.Plotly) {
      return Promise.resolve(window.Plotly);
    }
    if (plotlyPromise) {
      return plotlyPromise;
    }

    plotlyPromise = new Promise((resolve, reject) => {
      window.PlotlyConfig = { MathJaxConfig: "local" };
      const script = document.createElement("script");
      script.src = loader.dataset.plotlySrc;
      script.async = true;
      if (loader.dataset.plotlyIntegrity) {
        script.integrity = loader.dataset.plotlyIntegrity;
      }
      if (loader.dataset.plotlyCrossorigin) {
        script.crossOrigin = loader.dataset.plotlyCrossorigin;
      }
      script.addEventListener(
        "load",
        () => {
          if (!window.Plotly) {
            reject(new Error("The Plotly runtime did not initialize."));
            return;
          }
          resolve(window.Plotly);
        },
        { once: true },
      );
      script.addEventListener(
        "error",
        () => reject(new Error(`Unable to load the Plotly runtime: ${script.src}`)),
        { once: true },
      );
      document.head.appendChild(script);
    });
    return plotlyPromise;
  }

  function showLoadError(plot, error) {
    console.error("Unable to render an interactive Plotly chart.", error);
    plot.removeAttribute("aria-busy");
    plot.setAttribute("role", "status");
    plot.setAttribute("aria-live", "polite");
    plot.dataset.skfolioPlotlyState = "error";
    plot.classList.add("skfolio-plotly-error");
    plot.textContent = "The interactive chart could not be loaded.";
  }

  function loadInitializer(source) {
    return new Promise((resolve, reject) => {
      if (!source) {
        reject(new Error("Missing Plotly initializer URL."));
        return;
      }
      const executable = document.createElement("script");
      executable.src = source;
      executable.async = true;
      executable.addEventListener("load", resolve, { once: true });
      executable.addEventListener(
        "error",
        () => reject(new Error(`Unable to load a Plotly chart: ${source}`)),
        { once: true },
      );
      document.head.appendChild(executable);
    });
  }

  function renderPlot(plot) {
    if (plot.dataset.skfolioPlotlyState) {
      return;
    }
    const source = plot.dataset.skfolioPlotlySrc;
    if (!source) {
      showLoadError(plot, new Error("Missing Plotly initializer URL."));
      return;
    }

    plot.dataset.skfolioPlotlyState = "loading";
    plot.setAttribute("aria-busy", "true");
    loadPlotly()
      .then(() => loadInitializer(source))
      .then(() => {
        plot.dataset.skfolioPlotlyState = "rendered";
        plot.removeAttribute("aria-busy");
      })
      .catch((error) => showLoadError(plot, error));
  }

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            observer.unobserve(entry.target);
            renderPlot(entry.target);
          }
        });
      },
      { rootMargin: "600px 0px" },
    );
    plots.forEach((plot) => observer.observe(plot));
  } else {
    plots.forEach(renderPlot);
  }
})();
