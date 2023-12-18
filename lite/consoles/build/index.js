// Copyright (c) Jupyter Development Team.
// Distributed under the terms of the Modified BSD License.

import { NotebookApp } from '@jupyter-notebook/application';

import { JupyterLiteServer } from '@jupyterlite/server';

// The webpack public path needs to be set before loading the CSS assets.
import { PageConfig } from '@jupyterlab/coreutils';

import './style.js';

const serverExtensions = [
  import('@jupyterlite/server-extension')
];

// custom list of disabled plugins
const disabled = [
  "@jupyterlab/application-extension:dirty",
  "@jupyterlab/application-extension:info",
  "@jupyterlab/application-extension:layout",
  "@jupyterlab/application-extension:logo",
  "@jupyterlab/application-extension:main",
  "@jupyterlab/application-extension:mode-switch",
  "@jupyterlab/application-extension:notfound",
  "@jupyterlab/application-extension:paths",
  "@jupyterlab/application-extension:property-inspector",
  "@jupyterlab/application-extension:shell",
  "@jupyterlab/application-extension:status",
  "@jupyterlab/application-extension:tree-resolver",
  "@jupyterlab/apputils-extension:announcements",
  "@jupyterlab/apputils-extension:kernel-status",
  "@jupyterlab/apputils-extension:notification",
  "@jupyterlab/apputils-extension:palette-restorer",
  "@jupyterlab/apputils-extension:print",
  "@jupyterlab/apputils-extension:resolver",
  "@jupyterlab/apputils-extension:running-sessions-status",
  "@jupyterlab/apputils-extension:splash",
  "@jupyterlab/apputils-extension:workspaces",
  "@jupyterlab/console-extension:kernel-status",
  "@jupyterlab/docmanager-extension:download",
  "@jupyterlab/docmanager-extension:opener",
  "@jupyterlab/docmanager-extension:path-status",
  "@jupyterlab/docmanager-extension:saving-status",
  "@jupyterlab/documentsearch-extension:labShellWidgetListener",
  "@jupyterlab/filebrowser-extension:browser",
  "@jupyterlab/filebrowser-extension:download",
  "@jupyterlab/filebrowser-extension:file-upload-status",
  "@jupyterlab/filebrowser-extension:open-with",
  "@jupyterlab/filebrowser-extension:share-file",
  "@jupyterlab/filebrowser-extension:widget",
  "@jupyterlab/fileeditor-extension:editor-syntax-status",
  "@jupyterlab/fileeditor-extension:language-server",
  "@jupyterlab/fileeditor-extension:search",
  "@jupyterlab/notebook-extension:execution-indicator",
  "@jupyterlab/notebook-extension:kernel-status",
  "@jupyter-notebook/application-extension:logo",
  "@jupyter-notebook/application-extension:opener",
  "@jupyter-notebook/application-extension:path-opener",
];

async function createModule(scope, module) {
  try {
    const factory = await window._JUPYTERLAB[scope].get(module);
    return factory();
  } catch (e) {
    console.warn(`Failed to create module: package: ${scope}; module: ${module}`);
    throw e;
  }
}

/**
 * The main entry point for the application.
 */
export async function main() {
  const pluginsToRegister = [];
  const federatedExtensionPromises = [];
  const federatedMimeExtensionPromises = [];
  const federatedStylePromises = [];
  const litePluginsToRegister = [];
  const liteExtensionPromises = [];

  // This is all the data needed to load and activate plugins. This should be
  // gathered by the server and put onto the initial page template.
  const extensions = JSON.parse(
    PageConfig.getOption('federated_extensions')
  );

  // The set of federated extension names.
  const federatedExtensionNames = new Set();

  extensions.forEach(data => {
    if (data.liteExtension) {
      liteExtensionPromises.push(createModule(data.name, data.extension));
      return;
    }
    if (data.extension) {
      federatedExtensionNames.add(data.name);
      federatedExtensionPromises.push(createModule(data.name, data.extension));
    }
    if (data.mimeExtension) {
      federatedExtensionNames.add(data.name);
      federatedMimeExtensionPromises.push(createModule(data.name, data.mimeExtension));
    }
    if (data.style) {
      federatedStylePromises.push(createModule(data.name, data.style));
    }
  });

  /**
   * Iterate over active plugins in an extension.
   */
  function* activePlugins(extension) {
    // Handle commonjs or es2015 modules
    let exports;
    if (extension.hasOwnProperty('__esModule')) {
      exports = extension.default;
    } else {
      // CommonJS exports.
      exports = extension;
    }

    let plugins = Array.isArray(exports) ? exports : [exports];
    for (let plugin of plugins) {
      if (
        PageConfig.Extension.isDisabled(plugin.id) ||
        disabled.includes(plugin.id) ||
        disabled.includes(plugin.id.split(':')[0])
      ) {
        continue;
      }
      yield plugin;
    }
  }

  // Handle the mime extensions.
  const mimeExtensions = [];
  if (!federatedExtensionNames.has('@jupyterlab/javascript-extension')) {
    try {
      let ext = require('@jupyterlab/javascript-extension');
      for (let plugin of activePlugins(ext)) {
        mimeExtensions.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/json-extension')) {
    try {
      let ext = require('@jupyterlab/json-extension');
      for (let plugin of activePlugins(ext)) {
        mimeExtensions.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/vega5-extension')) {
    try {
      let ext = require('@jupyterlab/vega5-extension');
      for (let plugin of activePlugins(ext)) {
        mimeExtensions.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlite/iframe-extension')) {
    try {
      let ext = require('@jupyterlite/iframe-extension');
      for (let plugin of activePlugins(ext)) {
        mimeExtensions.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }

  // Add the federated mime extensions.
  const federatedMimeExtensions = await Promise.allSettled(federatedMimeExtensionPromises);
  federatedMimeExtensions.forEach(p => {
    if (p.status === "fulfilled") {
      for (let plugin of activePlugins(p.value)) {
        mimeExtensions.push(plugin);
      }
    } else {
      console.error(p.reason);
    }
  });

  // Handle the standard extensions.
  if (!federatedExtensionNames.has('@jupyterlab/application-extension')) {
    try {
      let ext = require('@jupyterlab/application-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/apputils-extension')) {
    try {
      let ext = require('@jupyterlab/apputils-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/cell-toolbar-extension')) {
    try {
      let ext = require('@jupyterlab/cell-toolbar-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/codemirror-extension')) {
    try {
      let ext = require('@jupyterlab/codemirror-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/completer-extension')) {
    try {
      let ext = require('@jupyterlab/completer-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/console-extension')) {
    try {
      let ext = require('@jupyterlab/console-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/docmanager-extension')) {
    try {
      let ext = require('@jupyterlab/docmanager-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/documentsearch-extension')) {
    try {
      let ext = require('@jupyterlab/documentsearch-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/filebrowser-extension')) {
    try {
      let ext = require('@jupyterlab/filebrowser-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/fileeditor-extension')) {
    try {
      let ext = require('@jupyterlab/fileeditor-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/lsp-extension')) {
    try {
      let ext = require('@jupyterlab/lsp-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/mainmenu-extension')) {
    try {
      let ext = require('@jupyterlab/mainmenu-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/mathjax-extension')) {
    try {
      let ext = require('@jupyterlab/mathjax-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/metadataform-extension')) {
    try {
      let ext = require('@jupyterlab/metadataform-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/notebook-extension')) {
    try {
      let ext = require('@jupyterlab/notebook-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/rendermime-extension')) {
    try {
      let ext = require('@jupyterlab/rendermime-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/shortcuts-extension')) {
    try {
      let ext = require('@jupyterlab/shortcuts-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/theme-dark-extension')) {
    try {
      let ext = require('@jupyterlab/theme-dark-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/theme-light-extension')) {
    try {
      let ext = require('@jupyterlab/theme-light-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/toc-extension')) {
    try {
      let ext = require('@jupyterlab/toc-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/tooltip-extension')) {
    try {
      let ext = require('@jupyterlab/tooltip-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/translation-extension')) {
    try {
      let ext = require('@jupyterlab/translation-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlab/ui-components-extension')) {
    try {
      let ext = require('@jupyterlab/ui-components-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyter-notebook/application-extension')) {
    try {
      let ext = require('@jupyter-notebook/application-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyter-notebook/docmanager-extension')) {
    try {
      let ext = require('@jupyter-notebook/docmanager-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyter-notebook/help-extension')) {
    try {
      let ext = require('@jupyter-notebook/help-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlite/application-extension')) {
    try {
      let ext = require('@jupyterlite/application-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }
  if (!federatedExtensionNames.has('@jupyterlite/notebook-application-extension')) {
    try {
      let ext = require('@jupyterlite/notebook-application-extension');
      for (let plugin of activePlugins(ext)) {
        pluginsToRegister.push(plugin);
      }
    } catch (e) {
      console.error(e);
    }
  }

  // Add the federated extensions.
  const federatedExtensions = await Promise.allSettled(federatedExtensionPromises);
  federatedExtensions.forEach(p => {
    if (p.status === "fulfilled") {
      for (let plugin of activePlugins(p.value)) {
        pluginsToRegister.push(plugin);
      }
    } else {
      console.error(p.reason);
    }
  });

  // Add the base serverlite extensions
  const baseServerExtensions = await Promise.all(serverExtensions);
  baseServerExtensions.forEach(p => {
    for (let plugin of activePlugins(p)) {
      litePluginsToRegister.push(plugin);
    }
  })

  // Add the serverlite federated extensions.
  const federatedLiteExtensions = await Promise.allSettled(liteExtensionPromises);
  federatedLiteExtensions.forEach(p => {
    if (p.status === "fulfilled") {
      for (let plugin of activePlugins(p.value)) {
        litePluginsToRegister.push(plugin);
      }
    } else {
      console.error(p.reason);
    }
  });

  // Load all federated component styles and log errors for any that do not
  (await Promise.allSettled(federatedStylePromises)).filter(({status}) => status === "rejected").forEach(({reason}) => {
     console.error(reason);
    });

  // create the in-browser JupyterLite Server
  const jupyterLiteServer = new JupyterLiteServer({});
  jupyterLiteServer.registerPluginModules(litePluginsToRegister);
  // start the server
  await jupyterLiteServer.start();

  // retrieve the custom service manager from the server app
  const { serviceManager } = jupyterLiteServer;

  // create a full-blown JupyterLab frontend
  const app = new NotebookApp({
    mimeExtensions,
    serviceManager
  });
  app.name = PageConfig.getOption('appName') || 'JupyterLite';

  app.registerPluginModules(pluginsToRegister);

  // Expose global app instance when in dev mode or when toggled explicitly.
  const exposeAppInBrowser =
    (PageConfig.getOption('exposeAppInBrowser') || '').toLowerCase() === 'true';

  if (exposeAppInBrowser) {
    window.jupyterapp = app;
  }

  /* eslint-disable no-console */
  await app.start();
  await app.restored;
}