#!/usr/bin/env node

/**
 * Audit rendered blog posts through the Chrome DevTools Protocol.
 *
 * Start Chrome with remote debugging enabled, then pass one or more URLs:
 *
 *   node scripts/blog_visual_audit.mjs http://127.0.0.1:8765/blog/2026/example/
 *
 * The script reports viewport overflow, clipped/upscaled images, and wide
 * tables, equations, or code blocks at desktop and phone widths.
 */

const endpoints = {
  browser: process.env.CHROME_DEBUG_URL || "http://127.0.0.1:9223",
};

const urls = process.argv.slice(2);
if (urls.length === 0) {
  console.error("Pass at least one rendered blog URL.");
  process.exit(2);
}

class CdpClient {
  constructor(socketUrl) {
    this.nextId = 1;
    this.pending = new Map();
    this.socket = new WebSocket(socketUrl);
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (!message.id) return;
      const waiter = this.pending.get(message.id);
      if (!waiter) return;
      this.pending.delete(message.id);
      if (message.error) waiter.reject(new Error(message.error.message));
      else waiter.resolve(message.result || {});
    });
  }

  send(method, params = {}) {
    const id = this.nextId++;
    const result = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
    this.socket.send(JSON.stringify({ id, method, params }));
    return result;
  }

  close() {
    this.socket.close();
  }
}

async function delay(milliseconds) {
  await new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function evaluate(client, expression) {
  const result = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text || "Page evaluation failed");
  }
  return result.result.value;
}

async function loadPage(client, url) {
  await client.send("Page.navigate", { url });
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const ready = await evaluate(client, "document.readyState");
    if (ready === "complete") break;
    await delay(100);
  }
  await evaluate(
    client,
    `(async () => {
      if (document.fonts?.ready) await document.fonts.ready;
      const distance = Math.max(600, Math.floor(innerHeight * 0.8));
      for (let y = 0; y < document.documentElement.scrollHeight; y += distance) {
        scrollTo(0, y);
        await new Promise((resolve) => setTimeout(resolve, 12));
      }
      scrollTo(0, 0);
      await new Promise((resolve) => setTimeout(resolve, 80));
      return true;
    })()`,
  );
}

const auditExpression = `(() => {
  const root = document.documentElement;
  const post = document.querySelector('.post-content');
  if (!post) return { error: 'missing .post-content' };
  const postRect = post.getBoundingClientRect();
  const tolerance = 1.5;
  const normalize = (value) => Math.round(value * 10) / 10;
  const captionFor = (image) => {
    const figure = image.closest('figure');
    if (figure?.querySelector('figcaption')) {
      return figure.querySelector('figcaption').textContent.trim();
    }
    const paragraph = image.closest('p');
    const inlineCaption = paragraph?.querySelector(':scope > em');
    if (inlineCaption) return inlineCaption.textContent.trim();
    const next = paragraph?.nextElementSibling;
    if (next && (next.querySelector('em') || next.classList.contains('caption'))) {
      return next.textContent.trim();
    }
    return '';
  };
  const metrics = (element) => {
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return {
      width: normalize(rect.width),
      height: normalize(rect.height),
      scrollWidth: element.scrollWidth,
      clientWidth: element.clientWidth,
      left: normalize(rect.left),
      right: normalize(rect.right),
      overflowX: style.overflowX,
      clipped: rect.left < postRect.left - tolerance || rect.right > postRect.right + tolerance,
    };
  };
  return {
    title: document.title,
    viewport: { width: innerWidth, height: innerHeight },
    documentOverflow: root.scrollWidth - root.clientWidth,
    postOverflow: post.scrollWidth - post.clientWidth,
    images: [...post.querySelectorAll('img')].map((image, index) => {
      const box = metrics(image);
      return {
        index: index + 1,
        src: image.currentSrc || image.src,
        alt: image.alt,
        naturalWidth: image.naturalWidth,
        naturalHeight: image.naturalHeight,
        ...box,
        upscaled: image.naturalWidth > 0 && image.naturalWidth + tolerance < box.width,
        viewportTall: box.height > innerHeight * 0.92,
        caption: captionFor(image),
      };
    }),
    tables: [...post.querySelectorAll('table')].map((table, index) => ({
      index: index + 1,
      ...metrics(table),
    })),
    equations: [...post.querySelectorAll('mjx-container[display="true"]')].map((equation, index) => ({
      index: index + 1,
      ...metrics(equation),
    })),
    codeBlocks: [...post.querySelectorAll('pre')].map((block, index) => ({
      index: index + 1,
      ...metrics(block),
    })),
    overflowingElements: [...post.querySelectorAll('*')]
      .map((element) => {
        const box = metrics(element);
        return {
          tag: element.tagName.toLowerCase(),
          className: typeof element.className === 'string' ? element.className : '',
          text: (element.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 120),
          ...box,
          excess: element.scrollWidth - element.clientWidth,
        };
      })
      .filter((element) => element.excess > 2 && element.overflowX !== 'auto')
      .sort((left, right) => right.excess - left.excess)
      .slice(0, 12),
  };
})()`;

function summarize(url, mode, report) {
  if (report.error) return { url, mode, error: report.error };
  const problemImages = report.images.filter(
    (image) => image.clipped || image.upscaled || image.viewportTall || !image.alt,
  );
  const problemTables = report.tables.filter(
    (table) => table.clipped || (table.scrollWidth > table.clientWidth && table.overflowX !== "auto"),
  );
  const problemEquations = report.equations.filter(
    (equation) => equation.clipped || equation.scrollWidth > equation.clientWidth + 2,
  );
  const problemCode = report.codeBlocks.filter(
    (block) => block.clipped || (block.scrollWidth > block.clientWidth && block.overflowX !== "auto"),
  );
  return {
    url,
    mode,
    title: report.title,
    viewport: report.viewport,
    documentOverflow: report.documentOverflow,
    postOverflow: report.postOverflow,
    imageCount: report.images.length,
    tableCount: report.tables.length,
    equationCount: report.equations.length,
    problemImages,
    problemTables,
    problemEquations,
    problemCode,
    overflowingElements: report.overflowingElements,
  };
}

const tabs = await fetch(`${endpoints.browser}/json/list`).then((response) => response.json());
const tab = tabs.find((candidate) => candidate.type === "page");
if (!tab) throw new Error("No debuggable Chrome page was found.");

const client = new CdpClient(tab.webSocketDebuggerUrl);
await client.open();
await client.send("Page.enable");
await client.send("Runtime.enable");

const modes = [
  { name: "desktop", width: 1440, height: 1000, scale: 1, mobile: false },
  { name: "phone", width: 390, height: 844, scale: 2, mobile: true },
];

for (const mode of modes) {
  await client.send("Emulation.setDeviceMetricsOverride", {
    width: mode.width,
    height: mode.height,
    deviceScaleFactor: mode.scale,
    mobile: mode.mobile,
  });
  for (const url of urls) {
    await loadPage(client, url);
    const report = await evaluate(client, auditExpression);
    console.log(JSON.stringify(summarize(url, mode.name, report)));
  }
}

await client.send("Emulation.clearDeviceMetricsOverride");
client.close();
