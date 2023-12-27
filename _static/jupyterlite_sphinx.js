window.jupyterliteShowIframe = (tryItButtonId, iframeSrc) => {
  const tryItButton = document.getElementById(tryItButtonId);
  const iframe = document.createElement('iframe');

  iframe.src = iframeSrc;
  iframe.width = iframe.height = '100%';
  iframe.classList.add('jupyterlite_sphinx_iframe');

  tryItButton.parentNode.appendChild(iframe);
  tryItButton.innerText = 'Loading ...';
  tryItButton.classList.remove('jupyterlite_sphinx_try_it_button_unclicked');
  tryItButton.classList.add('jupyterlite_sphinx_try_it_button_clicked');
}

window.jupyterliteConcatSearchParams = (iframeSrc, params) => {
  const baseURL = window.location.origin;
  const iframeUrl = new URL(iframeSrc, baseURL);

  let pageParams = new URLSearchParams(window.location.search);

  if (params === true) {
    params = Array.from(pageParams.keys());
  } else if (params === false) {
    params = [];
  } else if (!Array.isArray(params)) {
    console.error('The search parameters are not an array');
  }

  params.forEach(param => {
    value = pageParams.get(param);
    if (value !== null) {
      iframeUrl.searchParams.append(param, value);
    }
  });

  if (iframeUrl.searchParams.size) {
    return `${iframeSrc.split('?')[0]}?${iframeUrl.searchParams.toString()}`;
  } else {
    return iframeSrc;
  }
}


window.tryExamplesShowIframe = (
    examplesContainerId, iframeContainerId, iframeParentContainerId, iframeSrc,
    iframeMinHeight
) => {
    const examplesContainer = document.getElementById(examplesContainerId);
    const iframeParentContainer = document.getElementById(iframeParentContainerId);
    const iframeContainer = document.getElementById(iframeContainerId);

    let iframe = iframeContainer.querySelector('iframe.jupyterlite_sphinx_raw_iframe');

    if (!iframe) {
	      const examples = examplesContainer.querySelector('.try_examples_content');
	      iframe = document.createElement('iframe');
	      iframe.src = iframeSrc;
	      iframe.style.width = '100%';
              minHeight = parseInt(iframeMinHeight);
	      height = Math.max(minHeight, examples.offsetHeight);
	      iframe.style.height = `${height}px`;
	      iframe.classList.add('jupyterlite_sphinx_raw_iframe');
	      examplesContainer.classList.add("hidden");
	      iframeContainer.appendChild(iframe);
    } else {
	      examplesContainer.classList.add("hidden");
    }
    iframeParentContainer.classList.remove("hidden");
}


window.tryExamplesHideIframe = (examplesContainerId, iframeParentContainerId) => {
    const examplesContainer = document.getElementById(examplesContainerId);
    const iframeParentContainer = document.getElementById(iframeParentContainerId);

    iframeParentContainer.classList.add("hidden");
    examplesContainer.classList.remove("hidden");
}
