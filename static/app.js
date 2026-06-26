const API_BASE = '/api';
const DEFAULT_PDF_BATCH_SIZE = 1;
const MAX_PDF_BATCH_SIZE = 400;
const PDF_BATCH_SIZE_STORAGE_KEY = 'pandocr.pdfBatchSize';
const MODEL_STORAGE_KEY = 'pandocr.selectedModelId';
const API_TOKEN_STORAGE_KEY = 'pandocr.apiToken';
const LANGUAGE_STORAGE_KEY = 'pandocr.language';
const DEFAULT_MODEL_ID = 'paddleocr-vl-1.6';
const DEFAULT_PDF_ZOOM = 1;
const PDF_DEFAULT_PAGE_WIDTH = 595;
const PDF_FIT_WIDTH_GUTTER = 12;
const MAX_DEFAULT_PDF_ZOOM = 1.3;
const DEFAULT_MAX_UPLOAD_BYTES = 512 * 1024 * 1024;
const I18N_CONFIG = window.PANDOCR_I18N || {
    defaultLanguage: 'zh-CN',
    supportedLanguages: ['zh-CN'],
    titles: {
        'zh-CN': 'PaddleOCR Local - 本地 OCR 解析工作台'
    },
    dictionaries: {}
};

let availableModels = [{
    id: DEFAULT_MODEL_ID,
    name: 'PaddleOCR-VL-1.6-0.9B',
    label: 'PaddleOCR-VL 1.6',
    endpoint: '/api/paddleocr-vl-1.6'
}];
let selectedModelId = localStorage.getItem(MODEL_STORAGE_KEY) || DEFAULT_MODEL_ID;
let tasks = [];
let activeTaskId = null;
let activeFilter = 'all';
let taskSelectionMode = false;
const selectedTaskIds = new Set();

// Persist the last selected task id so the workbench can come back to
// the same place after a refresh, without auto-rendering the source.
const ACTIVE_TASK_STORAGE_KEY = 'pandocr.lastActiveTaskId';
const BATCH_QUEUE_STORAGE_KEY = 'pandocr.batchQueue';
let lastActiveTaskId = localStorage.getItem(ACTIVE_TASK_STORAGE_KEY) || null;
let activeResultView = 'markdown';
let isProcessing = false;
let currentPdf = null;
let currentPage = 1;
let currentZoom = DEFAULT_PDF_ZOOM;
let pdfDefaultPageWidth = PDF_DEFAULT_PAGE_WIDTH;
let sourceRenderToken = 0;
let renderedResultTaskId = null;
let lastRenderedHtml = '';
let renderedMarkdownKey = '';
let renderedOfficialLayoutContext = '';
let renderedPPOCRVisualContext = '';
let renderedJsonKey = '';
let cachedJsonLines = [];
let cachedJsonMaxLineLength = 0;
let jsonRenderToken = 0;
let ppocrScrollSyncFrame = 0;
let sourceScrollSyncFrame = 0;
let splitScrollSyncLocked = false;
let modelRuntime = null;
let modelRuntimePollTimer = null;
let modelRuntimeLoadInFlight = false;
let modelSwitchInFlight = false;
let maxUploadBytes = DEFAULT_MAX_UPLOAD_BYTES;
let maxTotalUploadBytes = 4096 * 1024 * 1024;
let chunkedUploadThreshold = 100 * 1024 * 1024;
let defaultChunkSize = 10 * 1024 * 1024;
let maxBatchBytes = 200 * 1024 * 1024;
let backgroundProcessingEnabled = true;
let activeEventSource = null;
let activePollInterval = null;
let folders = [];
let activeFolderId = null; // null = show all
let draggedTaskId = null;
let currentLanguage = normalizeLanguage(localStorage.getItem(LANGUAGE_STORAGE_KEY) || I18N_CONFIG.defaultLanguage);
const sourcePdfCache = new Map();
const sourceBytesCache = new Map();
const i18nTextSources = new WeakMap();
const i18nAttributeSources = new WeakMap();
const I18N_ATTRIBUTES = ['title', 'placeholder', 'aria-label'];
const JSON_LINE_HEIGHT = 21;
const JSON_PADDING_TOP = 34;
const JSON_PADDING_RIGHT = 40;
const JSON_PADDING_BOTTOM = 34;
const JSON_PADDING_LEFT = 40;
const JSON_OVERSCAN_LINES = 10;

const els = {
    sidebar: document.getElementById('sidebar'),
    sidebarToggle: document.getElementById('sidebar-toggle'),
    sidebarScrim: document.getElementById('sidebar-scrim'),
    fileInput: document.getElementById('file-input'),
    folderInput: document.getElementById('folder-input'),
    browseBtn: document.getElementById('browse-btn'),
    newTaskBtn: document.getElementById('new-task-btn'),
    batchFolderBtn: document.getElementById('batch-folder-btn'),
    batchQueueBar: document.getElementById('batch-queue-bar'),
    batchQueueTitle: document.getElementById('batch-queue-title'),
    batchQueueCounts: document.getElementById('batch-queue-counts'),
    batchQueueProgressFill: document.getElementById('batch-queue-progress-fill'),
    batchQueueCurrent: document.getElementById('batch-queue-current'),
    batchPauseBtn: document.getElementById('batch-pause-btn'),
    batchSkipBtn: document.getElementById('batch-skip-btn'),
    batchRetryBtn: document.getElementById('batch-retry-btn'),
    batchStopBtn: document.getElementById('batch-stop-btn'),
    batchStatPages: document.getElementById('batch-stat-pages'),
    batchStatElapsed: document.getElementById('batch-stat-elapsed'),
    batchStatEta: document.getElementById('batch-stat-eta'),
    batchStatPerPage: document.getElementById('batch-stat-perpage'),
    taskSelectModeBtn: document.getElementById('task-select-mode-btn'),
    taskSelectBar: document.getElementById('task-select-bar'),
    taskSelectAllCb: document.getElementById('task-select-all-cb'),
    taskSelectCounter: document.getElementById('task-select-counter'),
    taskSelectCancel: document.getElementById('task-select-cancel'),
    taskSelectParse: document.getElementById('task-select-parse'),
    taskSelectMove: document.getElementById('task-select-move'),
    taskSelectDelete: document.getElementById('task-select-delete'),
    dropZone: document.getElementById('drop-zone'),
    taskList: document.getElementById('task-list'),
    taskSearch: document.getElementById('task-search'),
    clearHistoryBtn: document.getElementById('clear-history-btn'),
    languageToggle: document.getElementById('language-toggle'),
    statusDot: document.getElementById('model-status-dot'),
    statusText: document.getElementById('model-status-text'),
    modelSelect: document.getElementById('model-select'),
    activeModelName: document.getElementById('active-model-name'),
    resultPane: document.querySelector('.result-pane'),
    sourceTitle: document.getElementById('source-title'),
    sourceMeta: document.getElementById('source-meta'),
    sourceViewer: document.getElementById('source-viewer'),
    pdfControls: document.getElementById('pdf-controls'),
    pageIndicator: document.getElementById('page-indicator'),
    prevPageBtn: document.getElementById('prev-page-btn'),
    nextPageBtn: document.getElementById('next-page-btn'),
    zoomInBtn: document.getElementById('zoom-in-btn'),
    zoomOutBtn: document.getElementById('zoom-out-btn'),
    resetZoomBtn: document.getElementById('reset-zoom-btn'),
    resultTitle: document.getElementById('result-title'),
    startBtn: document.getElementById('start-btn'),
    copyBtn: document.getElementById('copy-btn'),
    downloadBtn: document.getElementById('download-btn'),
    clearResultBtn: document.getElementById('clear-result-btn'),
    translateBtn: document.getElementById('translate-btn'),
    markdownView: document.getElementById('markdown-view'),
    jsonView: document.getElementById('json-view'),
    chartRecognitionSwitch: document.getElementById('chart-recognition-switch'),
    docUnwarpingSwitch: document.getElementById('doc-unwarping-switch'),
    docOrientationSwitch: document.getElementById('doc-orientation-switch'),
    sealRecognitionSwitch: document.getElementById('seal-recognition-switch'),
    formulaNumberSwitch: document.getElementById('formula-number-switch'),
    ignoreHeaderSwitch: document.getElementById('ignore-header-switch'),
    ignoreFooterSwitch: document.getElementById('ignore-footer-switch'),
    ignoreNumberSwitch: document.getElementById('ignore-number-switch'),
    pdfBatchSizeInput: document.getElementById('pdf-batch-size-input'),
    newFolderBtn: document.getElementById('new-folder-btn'),
    folderSelect: document.getElementById('folder-select'),
    taskTemplate: document.getElementById('task-item-template')
};

document.addEventListener('DOMContentLoaded', async () => {
    pdfjsLib.GlobalWorkerOptions.workerSrc = '/static/vendor/pdfjs/pdf.worker.min.js';
    initLanguage();
    initPdfBatchSizeSetting();
    setupEventListeners();
    renderModelSelect();
    await checkBackendConnection();
    await loadFolders();
    await loadTasks();
    renderTaskList();
    // Don't auto-select a task on launch — picking one would force a
    // PDF render (or huge image load) before the user asked for it.
    // Big PDFs easily freeze the page for several seconds.
    try { restoreBatchQueueFromStorage(); } catch (e) { console.warn('restoreBatchQueueFromStorage failed', e); clearPersistedBatchQueue(); }
    try {
        if (tasks.length > 0 && lastActiveTaskId && tasks.some((task) => task.id === lastActiveTaskId)) {
            await selectTaskLight(lastActiveTaskId);
        } else {
            showEmptyWorkbench();
        }
    } catch (e) {
        console.error('Initial workbench render failed', e);
        // Don't let one bug take down the whole UI — show empty state.
        try { showEmptyWorkbench(); } catch (_) {}
    }
    applyLanguage(document.body);
});

// True when the viewport is narrow enough that the sidebar overlays
// content as a drawer instead of taking up its own column.
function isNarrowViewport() {
    return window.matchMedia('(max-width: 900px)').matches;
}

function closeMobileSidebar() {
    document.body.classList.remove('sidebar-open');
}

// Which mobile pane is visible: "source" or "result". Persisted on the
// body so plain CSS can hide the inactive pane.
function setMobilePane(pane) {
    const next = pane === 'result' ? 'result' : 'source';
    document.body.dataset.mobilePane = next;
    document.querySelectorAll('.mobile-pane-tab').forEach((tab) => {
        const isActive = tab.dataset.pane === next;
        tab.classList.toggle('active', isActive);
        tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
}

// Initialize the default visible pane on first paint so CSS rules apply
// even before the user taps a tab.
setMobilePane('source');

function setupEventListeners() {
    [els.browseBtn, els.newTaskBtn].forEach((button) => {
        button?.addEventListener('click', () => els.fileInput.click());
    });
    els.fileInput.addEventListener('change', async (event) => {
        await handleFiles(event.target.files);
        els.fileInput.value = '';
    });

    // Batch-parse-folder: open the directory picker, then route the
    // selected files through handleFiles() under a folder named after
    // the chosen directory so they're auto-grouped + auto-parsed.
    els.batchFolderBtn?.addEventListener('click', () => {
        if (!els.folderInput) return;
        if (!('webkitdirectory' in els.folderInput)) {
            alert(t('当前浏览器不支持选择文件夹，请改用最新版 Chrome / Edge / Safari，或者直接拖拽文件夹到上传区。'));
            return;
        }
        els.folderInput.click();
    });

    els.folderInput?.addEventListener('change', async (event) => {
        const files = event.target.files;
        await handleFolderBatch(files);
        els.folderInput.value = '';
    });

    // Batch queue controls (paused/skipped/retry/stop).
    els.batchPauseBtn?.addEventListener('click', () => {
        if (batchQueue.paused) {
            resumeBatchQueue();
        } else {
            pauseBatchQueue();
        }
    });
    els.batchSkipBtn?.addEventListener('click', () => {
        skipCurrentBatchJob();
    });
    els.batchStopBtn?.addEventListener('click', () => {
        stopBatchQueue();
    });
    els.batchRetryBtn?.addEventListener('click', () => {
        retryFailedBatchJobs();
    });

    // Multi-select mode (batch parse / move / delete existing tasks).
    els.taskSelectModeBtn?.addEventListener('click', () => toggleTaskSelectionMode());
    els.taskSelectCancel?.addEventListener('click', () => toggleTaskSelectionMode(false));
    els.taskSelectAllCb?.addEventListener('change', () => {
        if (els.taskSelectAllCb.checked) selectAllVisibleTasks();
        else clearTaskSelection();
        renderTaskList();
        updateSelectionBar();
    });
    els.taskSelectParse?.addEventListener('click', () => batchParseSelected());
    els.taskSelectMove?.addEventListener('click', () => batchMoveSelected());
    els.taskSelectDelete?.addEventListener('click', () => batchDeleteSelected());

    ['dragenter', 'dragover'].forEach((name) => {
        document.addEventListener(name, (event) => {
            event.preventDefault();
            els.dropZone?.classList.add('drag-over');
        });
    });

    ['dragleave', 'drop'].forEach((name) => {
        document.addEventListener(name, (event) => {
            event.preventDefault();
            els.dropZone?.classList.remove('drag-over');
        });
    });

    document.addEventListener('drop', async (event) => {
        // If the drag carries any directory entries, recurse through them
        // and route everything through the batch flow so it lands in a
        // folder named after the dropped directory. Plain file drops keep
        // the existing single/multi-file behavior.
        const items = event.dataTransfer?.items;
        if (items && Array.from(items).some((it) => it.webkitGetAsEntry?.()?.isDirectory)) {
            const { files, topDir } = await collectDroppedEntries(items);
            if (!files.length) return;
            await handleFolderBatch(filesToFakeListWithRelative(files, topDir));
            return;
        }
        await handleFiles(event.dataTransfer.files);
    });

    els.sidebarToggle.addEventListener('click', () => {
        // On narrow viewports the sidebar is an overlay drawer toggled via
        // `body.sidebar-open`; on wider screens it's a column toggled via
        // `body.sidebar-collapsed`. Use the same icon for both.
        if (isNarrowViewport()) {
            document.body.classList.toggle('sidebar-open');
        } else {
            document.body.classList.toggle('sidebar-collapsed');
        }
    });

    els.sidebarScrim?.addEventListener('click', closeMobileSidebar);

    // Mobile-only pane switcher (source ↔ result). On narrow screens the
    // two panes stack and the inactive one is hidden via CSS — this just
    // flips the `data-mobile-pane` attribute that CSS keys off.
    document.querySelectorAll('.mobile-pane-tab').forEach((tab) => {
        tab.addEventListener('click', () => {
            const pane = tab.dataset.pane === 'result' ? 'result' : 'source';
            setMobilePane(pane);
        });
    });

    // If the user resizes from phone to desktop while the drawer is open,
    // drop the drawer state so the sidebar returns to its column layout.
    window.addEventListener('resize', () => {
        if (!isNarrowViewport()) {
            document.body.classList.remove('sidebar-open');
        }
    });

    els.taskSearch.addEventListener('input', renderTaskList);
    els.clearHistoryBtn.addEventListener('click', clearHistory);
    els.startBtn.addEventListener('click', () => {
        // If server-side processing is active, cancel instead of starting new
        const task = getActiveTask();
        const isBackendProcessing = task?.status === 'processing' && task?.sourceUrl && !task?.sourceDataUrl;
        // On phones the result is hidden behind the source pane — flip to it
        // as soon as the user starts parsing so they can watch progress.
        if (isNarrowViewport()) setMobilePane('result');
        if (isProcessing || isBackendProcessing) {
            cancelServerProcessing(task.id);
        } else if (shouldResumeTask(task)) {
            // Ask user: continue from where left off, or restart from scratch?
            const doneCount = (task.batches || []).filter((b) => b.status === 'completed').length;
            const totalCount = (task.batches || []).length;
            const hasContent = Boolean(task.markdown);
            if (!hasContent && doneCount > 0) {
                // Results were lost (e.g., server restart without hydration) — offer restart
                if (confirm(t('之前 {done}/{total} 批次的结果已丢失（可能因服务重启）。建议从头重新解析。是否从头开始？', { done: doneCount, total: totalCount }))) {
                    resetTaskForFullRerun(task);
                    processActiveTask();
                }
            } else {
                processActiveTask();
            }
        } else {
            processActiveTask();
        }
    });
    els.copyBtn.addEventListener('click', copyActiveResult);
    els.downloadBtn.addEventListener('click', downloadActiveTask);
    els.clearResultBtn.addEventListener('click', clearActiveResult);
    els.translateBtn.addEventListener('click', showTranslateDialog);
    els.prevPageBtn.addEventListener('click', () => changePdfPage(-1));
    els.nextPageBtn.addEventListener('click', () => changePdfPage(1));
    els.zoomInBtn.addEventListener('click', () => changeZoom(0.15));
    els.zoomOutBtn.addEventListener('click', () => changeZoom(-0.15));
    els.resetZoomBtn?.addEventListener('click', resetZoom);
    els.sourceViewer.addEventListener('scroll', handleSourceViewerScroll);
    els.markdownView.addEventListener('scroll', handlePPOCRMarkdownScroll);
    els.jsonView.addEventListener('scroll', renderVisibleJsonLines);
    els.modelSelect?.addEventListener('change', handleModelSelectionChange);
    els.languageToggle?.addEventListener('click', toggleLanguage);
    els.pdfBatchSizeInput?.addEventListener('input', handlePdfBatchSizeInput);
    ['change', 'blur'].forEach((eventName) => {
        els.pdfBatchSizeInput?.addEventListener(eventName, syncPdfBatchSizeSetting);
    });
    // Model-specific batch size inputs
    document.getElementById('pdf-batch-size-input-mineru')?.addEventListener('input', handlePdfBatchSizeInput);
    document.getElementById('pdf-batch-size-input-glm-ocr')?.addEventListener('input', handlePdfBatchSizeInput);
    els.newFolderBtn?.addEventListener('click', createFolderDialog);
    els.folderSelect?.addEventListener('change', handleFolderSelectChange);
    els.folderSelect?.addEventListener('contextmenu', handleFolderContextMenu);

    document.querySelectorAll('.task-tab').forEach((button) => {
        button.addEventListener('click', () => {
            document.querySelectorAll('.task-tab').forEach((tab) => tab.classList.remove('active'));
            button.classList.add('active');
            activeFilter = button.dataset.filter;
            renderTaskList();
        });
    });

    document.querySelectorAll('.view-tab').forEach((button) => {
        button.addEventListener('click', () => {
            setActiveResultView(button.dataset.view);
        });
    });
}

function initLanguage() {
    currentLanguage = normalizeLanguage(currentLanguage);
    localStorage.setItem(LANGUAGE_STORAGE_KEY, currentLanguage);
    applyLanguage(document.body);
}

function normalizeLanguage(language) {
    const supported = Array.isArray(I18N_CONFIG.supportedLanguages)
        ? I18N_CONFIG.supportedLanguages
        : ['zh-CN'];
    if (supported.includes(language)) return language;
    return I18N_CONFIG.defaultLanguage || supported[0] || 'zh-CN';
}

function toggleLanguage() {
    setLanguage(currentLanguage === 'en' ? 'zh-CN' : 'en');
}

function setLanguage(language) {
    const nextLanguage = normalizeLanguage(language);
    if (nextLanguage === currentLanguage) {
        applyLanguage(document.body);
        return;
    }
    currentLanguage = nextLanguage;
    localStorage.setItem(LANGUAGE_STORAGE_KEY, currentLanguage);
    refreshLanguageSensitiveUi();
}

function refreshLanguageSensitiveUi() {
    const task = getActiveTask();
    renderTaskList();
    updateActiveModelDisplay(task);
    updateResultViewLabels(task);

    if (task) {
        els.sourceMeta.textContent = taskSourceMeta(task);
        els.resultTitle.textContent = resultPaneTitle(task);
        renderResultPane(task);
    } else if (!isProcessing) {
        els.sourceTitle.textContent = t('等待上传文件');
        els.sourceMeta.textContent = t('PDF、图片、Office 文档');
        if (els.sourceViewer.querySelector('#drop-zone')) {
            els.sourceViewer.innerHTML = emptyDropZoneHtml();
            els.dropZone = document.getElementById('drop-zone');
            els.browseBtn = document.getElementById('browse-btn');
            els.browseBtn?.addEventListener('click', () => els.fileInput.click());
        }
        renderResultPane(null);
    }

    updateActionState(task);
    applyLanguage(document.body);
}

function applyLanguage(root = document.body) {
    if (!root) return;
    document.documentElement.lang = currentLanguage;
    document.title = I18N_CONFIG.titles?.[currentLanguage] || I18N_CONFIG.titles?.[I18N_CONFIG.defaultLanguage] || document.title;
    updateLanguageToggle();
    translateElementTree(root);
}

function updateLanguageToggle() {
    if (!els.languageToggle) return;
    els.languageToggle.dataset.lang = currentLanguage === 'en' ? 'en' : 'zh-CN';
    const labelSource = currentLanguage === 'en' ? '切换到中文' : '切换到英文';
    const label = t(labelSource);
    els.languageToggle.setAttribute('title', label);
    els.languageToggle.setAttribute('aria-label', label);
    const sources = getI18nAttributeSources(els.languageToggle);
    sources.set('title', labelSource);
    sources.set('aria-label', labelSource);
}

function translateElementTree(root) {
    const startingElement = root.nodeType === Node.ELEMENT_NODE ? root : root.parentElement;
    if (startingElement) translateElementAttributes(startingElement);

    const walker = document.createTreeWalker(
        root,
        NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT,
        {
            acceptNode(node) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    return shouldSkipI18nElement(node)
                        ? NodeFilter.FILTER_REJECT
                        : NodeFilter.FILTER_ACCEPT;
                }
                const parent = node.parentElement;
                if (!parent || shouldSkipI18nElement(parent)) return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
            }
        }
    );

    let node = walker.nextNode();
    while (node) {
        if (node.nodeType === Node.ELEMENT_NODE) {
            translateElementAttributes(node);
        } else {
            translateTextNode(node);
        }
        node = walker.nextNode();
    }
}

function shouldSkipI18nElement(element) {
    if (!element || element.closest('script, style, pre, code, #json-view')) return true;
    const markdownView = element.closest('#markdown-view');
    if (!markdownView) return false;
    if (element.id === 'markdown-view') return false;
    return Boolean(!element.closest('.empty-result') && !element.querySelector?.('.empty-result'));
}

function translateElementAttributes(element) {
    I18N_ATTRIBUTES.forEach((attribute) => {
        if (!element.hasAttribute(attribute)) return;
        const value = element.getAttribute(attribute);
        const sources = getI18nAttributeSources(element);
        if (!sources.has(attribute) && hasCjk(value)) {
            sources.set(attribute, value);
        }
        const source = sources.get(attribute);
        if (source) {
            element.setAttribute(attribute, t(source));
        }
    });
}

function getI18nAttributeSources(element) {
    let sources = i18nAttributeSources.get(element);
    if (!sources) {
        sources = new Map();
        i18nAttributeSources.set(element, sources);
    }
    return sources;
}

function translateTextNode(node) {
    const value = node.nodeValue || '';
    const trimmed = value.trim();
    if (!trimmed) return;
    if (!i18nTextSources.has(node) && hasCjk(trimmed)) {
        i18nTextSources.set(node, trimmed);
    }
    const source = i18nTextSources.get(node);
    if (!source) return;
    const leading = value.match(/^\s*/)?.[0] || '';
    const trailing = value.match(/\s*$/)?.[0] || '';
    node.nodeValue = `${leading}${t(source)}${trailing}`;
}

function hasCjk(value) {
    return /[\u3400-\u9fff]/.test(String(value || ''));
}

function languageLocale() {
    return currentLanguage === 'en' ? 'en-US' : 'zh-CN';
}

function t(source, params = {}) {
    const text = String(source ?? '');
    const translated = currentLanguage === normalizeLanguage(I18N_CONFIG.defaultLanguage)
        ? text
        : (I18N_CONFIG.dictionaries?.[currentLanguage]?.[text] || translateDynamicText(text) || text);
    return interpolateI18n(translated, params);
}

function interpolateI18n(text, params = {}) {
    return String(text).replace(/\{([a-zA-Z0-9_]+)\}/g, (match, key) => (
        Object.prototype.hasOwnProperty.call(params, key) ? String(params[key]) : match
    ));
}

function translateDynamicText(text) {
    if (currentLanguage === normalizeLanguage(I18N_CONFIG.defaultLanguage)) return text;
    const dynamicPatterns = [
        [/^(.+) 状态检查中$/, (name) => `${name} ${t('状态检查中')}`],
        [/^(.+) 就绪$/, (name) => `${name} ${t('就绪')}`],
        [/^(.+) 启动中$/, (name) => `${name} ${t('启动中')}`],
        [/^(.+) 启动失败$/, (name) => `${name} ${t('启动失败')}`],
        [/^(.+) 容器未创建$/, (name) => `${name} ${t('容器未创建')}`],
        [/^(.+) 待启动$/, (name) => `${name} ${t('待启动')}`],
        [/^(.+) 未就绪$/, (name) => `${name} ${t('未就绪')}`],
        [/^(.+) 还没有就绪，请稍后再试。$/, (name) => t('{name} 还没有就绪，请稍后再试。', { name })],
        [/^正在读取 (\d+) 个文件\.\.\.$/, (count) => t('正在读取 {count} 个文件...', { count })],
        [/^(\d+)\/(\d+) 解析中$/, (done, total) => t('{done}/{total} 解析中', { done, total })],
        [/^(\d+)\/(\d+) 可继续$/, (done, total) => t('{done}/{total} 可继续', { done, total })],
        [/^解析失败：(.+)$/, (detail) => t('解析失败：{detail}', { detail })],
        [/^Office 已转 PDF · (.+)$/, (name) => t('Office 已转 PDF · {name}', { name })],
        [/^第 (\d+) 页$/, (start) => t('第 {start} 页', { start })],
        [/^第 (\d+)-(\d+) 页$/, (start, end) => t('第 {start}-{end} 页', { start, end })],
        [/^(.+) 超过上传上限 (.+)，请压缩或拆分后再试。$/, (name, limit) => t('{name} 超过上传上限 {limit}，请压缩或拆分后再试。', { name, limit })],
        [/^不支持的文件格式：(.+)$/, (name) => t('不支持的文件格式：{name}', { name })],
        [/^确定要删除“(.+)”吗？当前操作不可回撤。$/, (name) => t('确定要删除“{name}”吗？当前操作不可回撤。', { name })],
        [/^保存本地任务失败：(.+)$/, (detail) => t('保存本地任务失败：{detail}', { detail })],
        [/^读取本地任务失败：(.+)$/, (detail) => t('读取本地任务失败：{detail}', { detail })],
        [/^清空本地任务失败：(.+)$/, (detail) => t('清空本地任务失败：{detail}', { detail })],
        [/^删除本地任务失败：(.+)$/, (detail) => t('删除本地任务失败：{detail}', { detail })],
        [/^保存源文件失败：(.+)$/, (detail) => t('保存源文件失败：{detail}', { detail })],
        [/^Office 转 PDF 失败：(.+)$/, (detail) => t('Office 转 PDF 失败：{detail}', { detail })],
        [/^读取 PDF 分页失败：(.+)$/, (detail) => t('读取 PDF 分页失败：{detail}', { detail })],
        [/^读取源文件失败：(.+)$/, (detail) => t('读取源文件失败：{detail}', { detail })]
    ];

    for (const [pattern, translate] of dynamicPatterns) {
        const match = text.match(pattern);
        if (match) return translate(...match.slice(1));
    }
    return '';
}

function isLocalApiUrl(url) {
    const text = String(url || '');
    if (text.startsWith(API_BASE) || text.startsWith('/api/')) return true;
    try {
        const parsed = new URL(text, window.location.href);
        return parsed.origin === window.location.origin && parsed.pathname.startsWith('/api/');
    } catch (error) {
        return false;
    }
}

function authHeaders(headers = {}, url = '') {
    const merged = new Headers(headers);
    if (isLocalApiUrl(url)) {
        const token = localStorage.getItem(API_TOKEN_STORAGE_KEY);
        if (token) merged.set('Authorization', `Bearer ${token}`);
    }
    return merged;
}

async function apiFetch(url, options = {}) {
    const requestOptions = {
        ...options,
        headers: authHeaders(options.headers, url)
    };
    let response = await fetch(url, requestOptions);
    if (response.status !== 401 || !isLocalApiUrl(url)) return response;

    const token = window.prompt(t('请输入 PaddleOCR Local API Token'));
    if (!token) return response;
    localStorage.setItem(API_TOKEN_STORAGE_KEY, token.trim());
    response = await fetch(url, {
        ...options,
        headers: authHeaders(options.headers, url)
    });
    return response;
}

async function responseErrorText(response) {
    const text = await response.text();
    try {
        const data = JSON.parse(text);
        return data.detail || text;
    } catch (error) {
        return text;
    }
}

async function checkBackendConnection() {
    try {
        const response = await apiFetch(`${API_BASE}/models`);
        if (!response.ok) throw new Error('API Error');
        const data = await response.json();
        availableModels = normalizeModelList(data);
        if (Number.isFinite(Number(data.maxUploadBytes)) && Number(data.maxUploadBytes) > 0) {
            maxUploadBytes = Number(data.maxUploadBytes);
        }
        if (Number.isFinite(Number(data.maxTotalUploadBytes)) && Number(data.maxTotalUploadBytes) > 0) {
            maxTotalUploadBytes = Number(data.maxTotalUploadBytes);
        }
        if (Number.isFinite(Number(data.chunkedUploadThreshold)) && Number(data.chunkedUploadThreshold) > 0) {
            chunkedUploadThreshold = Number(data.chunkedUploadThreshold);
        }
        if (Number.isFinite(Number(data.defaultChunkSize)) && Number(data.defaultChunkSize) > 0) {
            defaultChunkSize = Number(data.defaultChunkSize);
        }
        if (Number.isFinite(Number(data.maxBatchBytes)) && Number(data.maxBatchBytes) > 0) {
            maxBatchBytes = Number(data.maxBatchBytes);
        }
        if (!availableModels.some((model) => model.id === selectedModelId)) {
            selectedModelId = data.default || availableModels[0]?.id || DEFAULT_MODEL_ID;
        }
        localStorage.setItem(MODEL_STORAGE_KEY, selectedModelId);
        renderModelSelect();
        await loadModelRuntime({ silent: true });
        startModelRuntimePolling();
        updateActiveModelDisplay(getActiveTask());
        checkTranslateAvailable();
    } catch (error) {
        els.statusDot.className = 'dot error';
        els.statusText.textContent = t('模型未连接');
        setTimeout(checkBackendConnection, 5000);
    }
}

function startModelRuntimePolling() {
    if (modelRuntimePollTimer) return;
    modelRuntimePollTimer = window.setInterval(() => {
        loadModelRuntime({ silent: true }).catch((error) => {
            console.warn('Model runtime polling failed', error);
        });
    }, 2500);
}

async function loadModelRuntime({ silent = false } = {}) {
    if (modelRuntimeLoadInFlight) return modelRuntime;
    modelRuntimeLoadInFlight = true;
    try {
        const response = await apiFetch(`${API_BASE}/model-runtime`, { cache: 'no-store' });
        if (!response.ok) throw new Error(await response.text());
        modelRuntime = await response.json();
        // Clear modelSwitchInFlight once the backend operation is no longer 'switching'
        if (modelSwitchInFlight && modelRuntime?.operation?.state !== 'switching') {
            modelSwitchInFlight = false;
        }
        syncSelectedModelWithRuntime();
        updateActiveModelDisplay(getActiveTask());
        updateActionState(getActiveTask());
        return modelRuntime;
    } catch (error) {
        if (!silent) console.warn('Model runtime status failed', error);
        updateActiveModelDisplay(getActiveTask());
        return modelRuntime;
    } finally {
        modelRuntimeLoadInFlight = false;
    }
}

function normalizeModelList(data) {
    const models = Array.isArray(data?.data) ? data.data : [];
    if (!models.length) return availableModels;

    return models.map((model) => {
        if (typeof model === 'string') {
            return {
                id: model,
                name: model,
                label: model,
                endpoint: '/api/paddleocr-vl-1.6'
            };
        }
        return {
            id: model.id || model.name || DEFAULT_MODEL_ID,
            name: model.name || model.id || DEFAULT_MODEL_ID,
            label: model.label || model.name || model.id || DEFAULT_MODEL_ID,
            endpoint: model.endpoint || '/api/paddleocr-vl-1.6',
            kind: model.kind || 'document_parsing'
        };
    });
}

function renderModelSelect() {
    if (!els.modelSelect) return;
    els.modelSelect.innerHTML = '';
    availableModels.forEach((model) => {
        const option = document.createElement('option');
        option.value = model.id;
        option.textContent = modelDisplayName(model);
        option.selected = model.id === selectedModelId;
        els.modelSelect.appendChild(option);
    });
    updateSettingsPanelForModel(selectedModelId);
}

function syncSelectedModelWithRuntime() {
    if (!modelRuntime || !availableModels.length || modelSwitchInFlight) return false;
    if (modelRuntime.controlAvailable === false) return false;
    const knownModelIds = new Set(availableModels.map((model) => model.id));
    const operation = modelRuntime.operation;
    let runtimeModelId = null;

    if (operation?.state === 'switching' && knownModelIds.has(operation.targetModelId)) {
        runtimeModelId = operation.targetModelId;
    } else if (operation?.state === 'error' && knownModelIds.has(operation.targetModelId)) {
        // Switch failed — keep the UI on the failed target so the user sees the error
        runtimeModelId = operation.targetModelId;
    } else if (knownModelIds.has(modelRuntime.activeModelId)) {
        const activeStatus = getModelRuntimeStatus(modelRuntime.activeModelId);
        if (activeStatus?.running || activeStatus?.ready) {
            runtimeModelId = modelRuntime.activeModelId;
        }
    }

    if (!runtimeModelId || runtimeModelId === selectedModelId) return false;
    selectedModelId = runtimeModelId;
    localStorage.setItem(MODEL_STORAGE_KEY, selectedModelId);
    renderModelSelect();
    return true;
}

async function handleModelSelectionChange() {
    const nextModelId = els.modelSelect.value || DEFAULT_MODEL_ID;
    if (isProcessing || modelSwitchInFlight) {
        els.modelSelect.value = selectedModelId;
        alert(t('当前正在解析或切换模型，请完成后再切换。'));
        return;
    }
    const previousModelId = selectedModelId;
    selectedModelId = nextModelId;
    localStorage.setItem(MODEL_STORAGE_KEY, selectedModelId);
    updateSettingsPanelForModel(nextModelId);
    updateActiveModelDisplay(getActiveTask());
    updateActionState(getActiveTask());
    const switched = await switchModelRuntime(nextModelId, { wait: false });
    if (!switched) {
        selectedModelId = previousModelId;
        localStorage.setItem(MODEL_STORAGE_KEY, selectedModelId);
        renderModelSelect();
        updateSettingsPanelForModel(selectedModelId);
        updateActiveModelDisplay(getActiveTask());
        updateActionState(getActiveTask());
    }
}

function updateSettingsPanelForModel(modelId) {
    const paddleocrSettings = document.getElementById('paddleocr-settings');
    const mineruSettings = document.getElementById('mineru-settings');
    const glmOcrSettings = document.getElementById('glm-ocr-settings');
    if (!paddleocrSettings || !mineruSettings) return;
    const isMineru = modelId === 'mineru';
    const isGlmOcr = modelId === 'glm-ocr';
    paddleocrSettings.classList.toggle('hidden', isMineru || isGlmOcr);
    mineruSettings.classList.toggle('hidden', !isMineru);
    if (glmOcrSettings) glmOcrSettings.classList.toggle('hidden', !isGlmOcr);
}

function getSelectedModel() {
    return availableModels.find((model) => model.id === selectedModelId)
        || availableModels[0]
        || {
            id: DEFAULT_MODEL_ID,
            name: 'PaddleOCR-VL-1.6-0.9B',
            label: 'PaddleOCR-VL 1.6',
            endpoint: '/api/paddleocr-vl-1.6'
        };
}

function getTaskModel(task) {
    if (task?.modelId) {
        const known = availableModels.find((model) => model.id === task.modelId);
        if (known) return known;
        return {
            id: task.modelId,
            name: task.modelName || task.modelId,
            label: task.modelName || task.modelId,
            endpoint: task.modelEndpoint || '/api/paddleocr-vl-1.6'
        };
    }
    return getSelectedModel();
}

function modelDisplayName(model) {
    return model?.label || model?.name || model?.id || DEFAULT_MODEL_ID;
}

function modelShortName(model) {
    return modelDisplayName(model).replace('PaddleOCR-', '').replace('PaddleOCR ', '');
}

function modelApiUrl(model) {
    const endpoint = model?.endpoint || '/api/paddleocr-vl-1.6';
    if (/^https?:\/\//i.test(endpoint)) return endpoint;
    if (endpoint.startsWith('/api/')) return endpoint;
    return `${API_BASE}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
}

function getModelRuntimeStatus(modelId) {
    return modelRuntime?.models?.[modelId] || null;
}

function isModelRuntimeReady(modelId) {
    if (!modelRuntime) return true;
    return Boolean(getModelRuntimeStatus(modelId)?.ready);
}

function isModelRuntimeSwitching(modelId = null) {
    const operation = modelRuntime?.operation;
    if (modelSwitchInFlight) return !modelId || selectedModelId === modelId;
    return operation?.state === 'switching' && (!modelId || operation.targetModelId === modelId);
}

function canSwitchModelRuntime(modelId) {
    if (!modelRuntime) return true;
    const status = getModelRuntimeStatus(modelId);
    // glm-ocr is externally managed (Ollama); always allow "switching" to it
    if (modelId === 'glm-ocr') return true;
    return Boolean(modelRuntime.controlAvailable && status?.state !== 'missing');
}

function modelRuntimeDotClass(modelId) {
    const status = getModelRuntimeStatus(modelId);
    const operation = modelRuntime?.operation;
    if (!modelRuntime || isModelRuntimeSwitching(modelId) || status?.state === 'starting' || status?.state === 'partial') {
        return 'dot connecting';
    }
    if (status?.ready) return 'dot connected';
    if (operation?.state === 'error' && operation.targetModelId === modelId) return 'dot error';
    if (status?.state === 'missing') return 'dot error';
    if (status?.state === 'model_missing') return 'dot error';
    if (status?.state === 'offline') return 'dot error';
    return 'dot connecting';
}

function modelRuntimeStatusText(model) {
    const modelName = modelDisplayName(model);
    const status = getModelRuntimeStatus(model.id);
    const operation = modelRuntime?.operation;
    if (!modelRuntime) return t('{modelName} 状态检查中', { modelName });
    if (status?.ready) return t('{modelName} 就绪', { modelName });
    if (isModelRuntimeSwitching(model.id) || status?.state === 'starting' || status?.state === 'partial') {
        return t('{modelName} 启动中', { modelName });
    }
    if (operation?.state === 'error' && operation.targetModelId === model.id) {
        return t('{modelName} 启动失败', { modelName });
    }
    if (status?.state === 'missing') return t('{modelName} 容器未创建', { modelName });
    if (status?.state === 'model_missing') return t('{modelName} 模型未加载', { modelName });
    if (status?.state === 'offline') return t('{modelName} 服务离线', { modelName });
    if (status?.state === 'stopped') return t('{modelName} 待启动', { modelName });
    if (modelRuntime.controlAvailable === false) return t('{modelName} 未就绪', { modelName });
    return t('{modelName} 未就绪', { modelName });
}

function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function switchModelRuntime(modelId, { wait = false } = {}) {
    const model = availableModels.find((item) => item.id === modelId) || getSelectedModel();
    if (isModelRuntimeReady(modelId)) {
        updateActiveModelDisplay(getActiveTask());
        updateActionState(getActiveTask());
        return true;
    }
    if (!canSwitchModelRuntime(modelId)) {
        updateActiveModelDisplay(getActiveTask());
        updateActionState(getActiveTask());
        return false;
    }

    modelSwitchInFlight = true;
    updateActiveModelDisplay(getActiveTask());
    updateActionState(getActiveTask());
    try {
        const response = await apiFetch(`${API_BASE}/model-runtime/switch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ modelId })
        });
        if (!response.ok) throw new Error(await responseErrorText(response));
        modelRuntime = await response.json();
        syncSelectedModelWithRuntime();
        updateActiveModelDisplay(getActiveTask());
        updateActionState(getActiveTask());
        if (wait) {
            const result = await waitForModelRuntimeReady(modelId);
            modelSwitchInFlight = false;
            return result;
        }
        // Don't clear modelSwitchInFlight immediately — the backend switch is still
        // in progress. The polling loop (loadModelRuntime) will clear it once the
        // operation transitions away from 'switching'.
        return true;
    } catch (error) {
        console.error(error);
        modelSwitchInFlight = false;
        els.statusDot.className = 'dot error';
        els.statusText.textContent = t('{modelName} 启动失败', { modelName: modelDisplayName(model) });
        return false;
    } finally {
        updateActiveModelDisplay(getActiveTask());
        updateActionState(getActiveTask());
    }
}

async function waitForModelRuntimeReady(modelId, timeoutMs = 20 * 60 * 1000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        await loadModelRuntime({ silent: true });
        if (isModelRuntimeReady(modelId)) return true;
        const operation = modelRuntime?.operation;
        if (operation?.targetModelId === modelId && operation.state === 'error') {
            throw new Error(operation.message || t('模型启动失败'));
        }
        await sleep(2500);
    }
    throw new Error(t('模型启动超时'));
}

async function ensureModelRuntimeReadyForTask(task, model) {
    if (isModelRuntimeReady(model.id)) return true;
    const switched = await switchModelRuntime(model.id, { wait: true });
    if (switched && isModelRuntimeReady(model.id)) return true;
    alert(t('{name} 还没有就绪，请稍后再试。', { name: modelDisplayName(model) }));
    updateActionState(task);
    return false;
}

function updateActiveModelDisplay(task = null) {
    const selectedModel = getSelectedModel();
    const activeModel = task?.modelId ? getTaskModel(task) : selectedModel;
    els.statusDot.className = modelRuntimeDotClass(selectedModel.id);
    els.statusText.textContent = modelRuntimeStatusText(selectedModel);
    els.activeModelName.textContent = modelShortName(activeModel);
}

function applySelectedModelToTask(task) {
    const model = getSelectedModel();
    task.modelId = model.id;
    task.modelName = modelDisplayName(model);
    task.modelEndpoint = model.endpoint;
    return model;
}

async function saveTask(task, { includeResults = true } = {}) {
    await saveTaskToServer(task, { includeResults });
}

async function saveTaskToServer(task, { includeResults = true } = {}) {
    const response = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(task.id)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(taskForPersistence(task, { includeResults }))
    });
    if (!response.ok) {
        throw new Error(t('保存本地任务失败：{detail}', { detail: await responseErrorText(response) }));
    }
}

async function loadTasks() {
    const localTasks = await loadServerTasks();
    tasks = dedupeTasks(localTasks.map(reconcileTaskStatus));
}

function reconcileTaskStatus(task) {
    if (task?.status !== 'processing') return task;

    const batches = Array.isArray(task.batches) ? task.batches : [];
    const allBatchesCompleted = batches.length > 0 && batches.every((batch) => batch.status === 'completed');
    const hasAllOcrResults = Array.isArray(task.ocrResults) && task.ocrResults.length >= batches.length;
    if (task.status === 'processing' && allBatchesCompleted && hasAllOcrResults) {
        return { ...task, status: 'completed', updatedAt: task.updatedAt || Date.now() };
    }
    if (!isTaskDetailLoaded(task) && Number(task.completedPages || 0) >= Number(task.pageCount || Infinity)) {
        return { ...task, status: 'completed', updatedAt: task.updatedAt || Date.now() };
    }
    const reconciled = {
        ...task,
        status: 'pending',
        error: task.error || t('上次解析中断，可继续解析。'),
        updatedAt: task.updatedAt || Date.now()
    };
    if (isTaskDetailLoaded(task)) {
        reconciled.batches = batches.map((batch) => (
            batch.status === 'processing'
                ? { ...batch, status: 'pending' }
                : batch
        ));
    }
    return reconciled;
}

function dedupeTasks(taskItems) {
    const byFingerprint = new Map();
    taskItems.forEach((task) => {
        const fingerprint = [
            task.name,
            task.originalName || '',
            task.sourceKind || '',
            task.size || 0,
            task.pageCount || 0,
            task.modelId || ''
        ].join('|');
        const existing = byFingerprint.get(fingerprint);
        if (!existing || (task.updatedAt || 0) > (existing.updatedAt || 0)) {
            byFingerprint.set(fingerprint, task);
        }
    });
    return Array.from(byFingerprint.values()).sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
}

async function loadServerTasks() {
    try {
        const response = await apiFetch(`${API_BASE}/tasks`);
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        return data.tasks || [];
    } catch (error) {
        console.warn('读取本地任务目录失败', error);
        return [];
    }
}

async function loadTaskFromServer(taskId) {
    // Use lite mode by default to avoid loading hundreds of MB of
    // base64 images and OCR results into the browser at once.
    const response = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}?lite=true`);
    if (!response.ok) {
        throw new Error(t('读取本地任务失败：{detail}', { detail: await responseErrorText(response) }));
    }
    return reconcileTaskStatus(await response.json());
}

async function loadTaskResult(taskId, options = {}) {
    /** Lazily load heavy result data (images, ocrResults) with pagination.
     *  options.fields — comma-separated: 'images,ocrResults,markdown,translation'
     *  options.imageOffset / imageLimit — pagination for images
     *  options.ocrOffset / ocrLimit — pagination for ocrResults
     */
    const params = new URLSearchParams();
    if (options.fields) params.set('fields', options.fields);
    if (options.imageOffset) params.set('image_offset', options.imageOffset);
    if (options.imageLimit) params.set('image_limit', options.imageLimit);
    if (options.ocrOffset) params.set('ocr_offset', options.ocrOffset);
    if (options.ocrLimit) params.set('ocr_limit', options.ocrLimit);
    const qs = params.toString();
    const url = `${API_BASE}/tasks/${encodeURIComponent(taskId)}/result${qs ? '?' + qs : ''}`;
    const response = await apiFetch(url);
    if (!response.ok) {
        throw new Error(t('读取任务结果失败：{detail}', { detail: await responseErrorText(response) }));
    }
    return response.json();
}

function isTaskDetailLoaded(task) {
    return Boolean((task?.sourceDataUrl || task?.sourceUrl) && Array.isArray(task?.batches));
}

function replaceTask(task) {
    const index = tasks.findIndex((item) => item.id === task.id);
    if (index === -1) {
        tasks.unshift(task);
        return task;
    }
    tasks[index] = { ...tasks[index], ...task, detailLoaded: true };
    return tasks[index];
}

async function ensureTaskLoaded(taskId) {
    let task = tasks.find((item) => item.id === taskId);
    if (!task) return null;
    if (isTaskDetailLoaded(task)) return task;

    els.sourceTitle.textContent = task.name || t('正在加载任务');
    els.sourceMeta.textContent = t('正在加载本地任务详情...');
    els.resultTitle.textContent = t('正在加载');
    els.markdownView.innerHTML = `<div class="empty-result">${escapeHtml(t('正在加载任务详情...'))}</div>`;
    els.jsonView.textContent = '';
    updateActionState(null);

    task = await loadTaskFromServer(taskId);
    return replaceTask(task);
}

async function deleteAllTasks() {
    const response = await apiFetch(`${API_BASE}/tasks`, { method: 'DELETE' });
    if (!response.ok) {
        throw new Error(t('清空本地任务失败：{detail}', { detail: await responseErrorText(response) }));
    }
}

async function deleteTaskById(taskId) {
    const response = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}`, { method: 'DELETE' });
    if (!response.ok) {
        throw new Error(t('删除本地任务失败：{detail}', { detail: await responseErrorText(response) }));
    }
}

async function handleFiles(files) {
    if (!files || files.length === 0) return;

    const previousActiveTaskId = activeTaskId;
    const fileList = Array.from(files);
    showIncomingFileState(fileList);
    const results = await Promise.allSettled(fileList.map((file) => createTaskFromFile(file)));
    const newTasks = results
        .filter((result) => result.status === 'fulfilled')
        .map((result) => result.value);
    const failed = results.filter((result) => result.status === 'rejected');

    if (failed.length > 0) {
        console.warn('Some files could not be added', failed.map((result) => result.reason));
        const message = failed
            .map((result) => result.reason?.message || String(result.reason || t('文件读取失败')))
            .join('\n');
        els.markdownView.innerHTML = `<div class="empty-result">${escapeHtml(message)}</div>`;
    }
    if (newTasks.length === 0) {
        if (previousActiveTaskId && tasks.some((task) => task.id === previousActiveTaskId)) {
            await selectTask(previousActiveTaskId);
        } else {
            resetWorkbench();
        }
        return;
    }

    tasks = [...newTasks, ...tasks];

    // Auto-move new tasks into the currently selected folder
    if (activeFolderId) {
        const folderName = folders.find(f => f.id === activeFolderId)?.name || '';
        for (const task of newTasks) {
            task.folderId = activeFolderId;
            task.folderName = folderName;
        }
    }

    renderTaskList();
    await selectTask(newTasks[0].id);
    const saveResults = await Promise.allSettled(newTasks.map((task) => saveTask(task)));
    const saveFailures = saveResults.filter((result) => result.status === 'rejected');
    if (saveFailures.length > 0) {
        console.warn('Some tasks could not be saved before processing', saveFailures.map((result) => result.reason));
    }

    // Hand the freshly-created tasks to the batch queue driver. The
    // driver enforces strict serialization (one OCR job in flight at a
    // time) — even for a single task it's a thin wrapper around
    // processTask(). For >1 task it's the only thing that actually
    // works, because processTask() returns early for server-side jobs.
    enqueueTasks(newTasks.map((task) => task.id));
}

// Extensions accepted by the batch folder picker. Mirrors the
// `accept` attribute on the single-file <input>.
const BATCH_SUPPORTED_EXTS = new Set([
    'pdf', 'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff', 'tif', 'gif',
    'ppt', 'pptx', 'doc', 'docx'
]);

// Walk a DataTransferItemList that came from a drag-drop and produce
// { files, topDir }. Each File gets a synthetic `webkitRelativePath`
// so the downstream batch handler can name the auto-folder.
async function collectDroppedEntries(items) {
    const collected = [];
    let topDir = '';
    const tasks = [];
    for (const item of Array.from(items)) {
        const entry = item.webkitGetAsEntry?.();
        if (!entry) continue;
        if (entry.isDirectory) topDir ||= entry.name;
        tasks.push(walkEntry(entry, '', collected));
    }
    await Promise.all(tasks);
    return { files: collected, topDir };
}

function walkEntry(entry, prefix, out) {
    return new Promise((resolve) => {
        if (entry.isFile) {
            entry.file((file) => {
                // Preserve the relative path for later display / folder naming.
                file._batchRelativePath = prefix ? `${prefix}/${file.name}` : file.name;
                out.push(file);
                resolve();
            }, () => resolve());
            return;
        }
        if (entry.isDirectory) {
            const reader = entry.createReader();
            const nested = prefix ? `${prefix}/${entry.name}` : entry.name;
            const readBatch = () => {
                reader.readEntries(async (entries) => {
                    if (!entries.length) return resolve();
                    await Promise.all(entries.map((child) => walkEntry(child, nested, out)));
                    readBatch();   // readEntries returns at most ~100 per call
                }, () => resolve());
            };
            readBatch();
            return;
        }
        resolve();
    });
}

// handleFolderBatch reads `webkitRelativePath` from each File, but for
// drag-dropped files we stored the path on `_batchRelativePath` instead
// (the real property is read-only). Wrap them so both code paths agree.
function filesToFakeListWithRelative(files, topDir) {
    return files.map((f) => {
        // If a real webkitRelativePath already exists (rare from drag),
        // leave it alone. Otherwise expose our synthetic copy under the
        // expected name via a Proxy-free property assignment.
        if (!f.webkitRelativePath && f._batchRelativePath) {
            try {
                Object.defineProperty(f, 'webkitRelativePath', {
                    value: f._batchRelativePath,
                    configurable: true,
                });
            } catch (_) { /* some browsers refuse — fine, we still have the file */ }
        }
        return f;
    });
}

// Entry point for batch picks (file picker + folder drag-drop). Strips
// unsupported extensions, drops the files into an auto-named sidebar
// folder, then hands them to the queue driver which is what actually
// keeps the OCR pipeline serial (handleFiles() alone doesn't, because
// server-side jobs return as soon as SSE is wired up — only the queue
// driver knows when one job is *really* finished).
async function handleFolderBatch(fileListLike, { folderNameHint = '' } = {}) {
    if (!fileListLike || fileListLike.length === 0) return;

    const all = Array.from(fileListLike);
    const supported = all.filter((file) => {
        if (file.name?.startsWith('.')) return false;            // skip dotfiles
        return BATCH_SUPPORTED_EXTS.has(getExtension(file.name));
    });

    if (supported.length === 0) {
        alert(t('没有可解析的文件（支持 PDF、图片、PPT/DOC）。'));
        return;
    }

    // Folder name: prefer caller hint (drag-drop top dir), else first
    // file's webkitRelativePath, else a timestamped fallback.
    const firstRel = supported[0].webkitRelativePath || '';
    const dirFromRel = firstRel.split('/')[0];
    const topDir = (folderNameHint || dirFromRel || `批量解析-${formatBatchTimestamp()}`).trim();

    const skipped = all.length - supported.length;
    const proceed = confirm(t('将批量解析 {count} 个文件{skip}，归到文件夹「{name}」。确认继续？', {
        count: supported.length,
        skip: skipped > 0 ? t('（跳过 {n} 个不支持的）', { n: skipped }) : '',
        name: topDir
    }));
    if (!proceed) return;

    // Make sure the folder exists, then activate it so newly created
    // tasks land inside it automatically.
    const folder = await ensureFolderByName(topDir);
    if (folder) {
        activeFolderId = folder.id;
        if (els.folderSelect) els.folderSelect.value = folder.id;
    }

    await handleFiles(supported);
}

function formatBatchTimestamp() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
}

// ── Batch queue driver ────────────────────────────────────────────
//
// The queue keeps OCR strictly serial across an arbitrary number of
// uploaded tasks. handleFiles() used to await processTask() in a
// for-loop, but server-side processTask returns the moment SSE is
// connected — the second iteration would then see isProcessing===true
// and bail out, leaving every task after the first unparsed. The
// driver below awaits a completion promise that only resolves when the
// SSE / polling layer reports the task as completed / error / paused,
// so the next task only starts when the previous one is actually done.
//
// State:
//   queue        — task ids waiting to be parsed (FIFO)
//   running      — task id currently being parsed (or null)
//   results      — {taskId: 'completed'|'error'|'skipped'} for the
//                  current batch run, used to drive the retry button
//                  and the progress display
//   totalThisRun — how many tasks were in this run when it started
//                  (used to compute the progress bar)
//   paused       — when true, the driver stops at the next boundary
//   stopping     — when true, the driver drains immediately
//   completion   — Promise that resolves when the active task finishes
//                  (resolved by finishServerProcessing / pollTaskProgress
//                  / client-side processTask normal+error paths)
const batchQueue = {
    queue: [],
    running: null,
    results: new Map(),
    totalThisRun: 0,
    paused: false,
    stopping: false,
    completion: null,
    completionResolve: null,
    driverRunning: false,
    // Aggregate stats across the run, recomputed in renderBatchQueueBar
    // (so we can cheaply update them every progress tick):
    // - startedAt: when the current run began (ms epoch)
    // - totalPages: pages across every task enqueued in this run
    // - completedPages: pages already finished (counted once per task as
    //   soon as the task reaches a terminal state; falls back to
    //   task.pageCount for client-side jobs)
    startedAt: 0,
    totalPages: 0,
    completedPages: 0,
};

function enqueueTasks(taskIds, { autoStart = true } = {}) {
    if (!taskIds || !taskIds.length) return;
    // If the run is fully idle (no in-flight, no queued), start a fresh
    // stats window. Otherwise we keep the existing startedAt and just
    // extend the totals — this is what "add another file mid-batch"
    // does and we don't want to reset the elapsed-time clock.
    const wasIdle = !batchQueue.running && batchQueue.queue.length === 0 && batchQueue.results.size === 0;
    if (wasIdle) {
        batchQueue.startedAt = Date.now();
        batchQueue.completedPages = 0;
        batchQueue.totalPages = 0;
    }
    for (const id of taskIds) {
        if (id && !batchQueue.queue.includes(id) && batchQueue.running !== id) {
            batchQueue.queue.push(id);
            // pageCount is set during createTaskFromFile() / the lite
            // server response. Fall back to 1 so single-page tasks still
            // contribute to ETA.
            const task = tasks.find((t) => t.id === id);
            batchQueue.totalPages += Number(task?.pageCount || 1);
        }
    }
    batchQueue.totalThisRun = countQueueTotal();
    renderBatchQueueBar();
    persistBatchQueue();
    if (autoStart) startBatchDriver();
}

function countQueueTotal() {
    // Total = already done in this run + currently running + still queued.
    return batchQueue.results.size + (batchQueue.running ? 1 : 0) + batchQueue.queue.length;
}

function countDoneStatus(status) {
    let n = 0;
    for (const value of batchQueue.results.values()) if (value === status) n++;
    return n;
}

function resetBatchRunStats() {
    batchQueue.results.clear();
    batchQueue.totalThisRun = 0;
    batchQueue.startedAt = 0;
    batchQueue.completedPages = 0;
    batchQueue.totalPages = 0;
}

async function startBatchDriver() {
    if (batchQueue.driverRunning) return;
    batchQueue.driverRunning = true;
    try {
        while (!batchQueue.stopping && (batchQueue.queue.length > 0 || batchQueue.running)) {
            if (batchQueue.paused) {
                renderBatchQueueBar();
                await waitWhile(() => batchQueue.paused && !batchQueue.stopping);
                if (batchQueue.stopping) break;
            }
            const nextId = batchQueue.queue.shift();
            if (!nextId) break;
            const task = tasks.find((t) => t.id === nextId);
            if (!task) continue;
            batchQueue.running = nextId;
            renderBatchQueueBar();

            // Surface the task that's running so users see progress live.
            if (activeTaskId !== nextId) {
                selectTask(nextId).catch(() => {});
            }

            try {
                await runOneAndAwait(task);
                if (!batchQueue.results.has(nextId)) {
                    const final = (tasks.find((t) => t.id === nextId)?.status) || 'error';
                    batchQueue.results.set(nextId, final === 'completed' ? 'completed' : 'error');
                }
            } catch (err) {
                console.error('Batch task failed', err);
                batchQueue.results.set(nextId, 'error');
            } finally {
                // Add the just-finished task's pages to the completed
                // tally — even on error we count the pages that did get
                // parsed (best-effort: pageCount or completedPages from
                // the live task object). The driver keeps a running
                // total so the ETA math stays correct across the run.
                const doneTask = tasks.find((t) => t.id === nextId);
                if (doneTask) {
                    const pagesDone = Number(
                        doneTask.completedPages
                        || (Array.isArray(doneTask.batches) ? doneTask.batches.filter((b) => b.status === 'completed').length : 0)
                        || doneTask.pageCount
                        || 1
                    );
                    batchQueue.completedPages += pagesDone;
                }
                batchQueue.running = null;
                persistBatchQueue();
                batchQueue.running = null;
                batchQueue.completion = null;
                batchQueue.completionResolve = null;
                renderBatchQueueBar();
            }
        }
    } finally {
        batchQueue.driverRunning = false;
        if (batchQueue.queue.length === 0 && !batchQueue.running) {
            renderBatchQueueBar({ finished: true });
        }
    }
}

// Wraps processTask so the driver waits until the task hits a terminal
// state (completed / error / paused / skipped), regardless of whether
// it was processed client-side or via the server SSE path.
async function runOneAndAwait(task) {
    batchQueue.completion = new Promise((resolve) => {
        batchQueue.completionResolve = resolve;
    });
    // processTask resolves immediately for server-side jobs (after SSE
    // is connected). That's why we don't return its promise — we wait
    // on our own completion promise instead, which the SSE / polling /
    // client-side paths all resolve via signalBatchJobDone().
    processTask(task, { confirmCompleted: false }).catch((err) => {
        console.error('processTask threw inside batch queue', err);
        signalBatchJobDone(task.id, 'error');
    });
    await batchQueue.completion;
}

// Called from finishServerProcessing / pollTaskProgress / client-side
// terminal paths to release the queue driver.
function signalBatchJobDone(taskId, status) {
    if (batchQueue.running !== taskId) return;          // not the one we're awaiting
    if (status && !batchQueue.results.has(taskId)) batchQueue.results.set(taskId, status);
    if (batchQueue.completionResolve) {
        const resolve = batchQueue.completionResolve;
        batchQueue.completionResolve = null;
        resolve();
    }
}

function waitWhile(predicate) {
    return new Promise((resolve) => {
        const tick = () => {
            if (!predicate()) return resolve();
            setTimeout(tick, 120);
        };
        tick();
    });
}

function pauseBatchQueue() {
    if (!isBatchActive()) return;
    batchQueue.paused = true;
    renderBatchQueueBar();
    persistBatchQueue();
}

function resumeBatchQueue() {
    batchQueue.paused = false;
    renderBatchQueueBar();
    persistBatchQueue();
    startBatchDriver();
}

async function skipCurrentBatchJob() {
    const runningId = batchQueue.running;
    if (!runningId) return;
    // Mark as skipped so the retry button doesn't try to re-run it.
    batchQueue.results.set(runningId, 'skipped');
    // Use the existing cancel path for server-side jobs; for client-side
    // there's no in-flight cancellation, so we just let the driver
    // detect the early resolve and move on.
    try {
        await cancelServerProcessing(runningId);
    } catch (_) { /* ignore — driver will move on regardless */ }
    signalBatchJobDone(runningId, 'skipped');
    persistBatchQueue();
}

async function stopBatchQueue() {
    if (!confirm(t('停止批量解析？剩余 {n} 个任务将不会被解析。', { n: batchQueue.queue.length }))) return;
    batchQueue.stopping = true;
    batchQueue.queue = [];
    const runningId = batchQueue.running;
    if (runningId) {
        batchQueue.results.set(runningId, 'skipped');
        try { await cancelServerProcessing(runningId); } catch (_) {}
        signalBatchJobDone(runningId, 'skipped');
    }
    // Let the driver loop unwind, then clear stopping for next run.
    setTimeout(() => {
        batchQueue.stopping = false;
        batchQueue.paused = false;
        renderBatchQueueBar();
        clearPersistedBatchQueue();
    }, 200);
}

function retryFailedBatchJobs() {
    const failed = [];
    for (const [id, status] of batchQueue.results.entries()) {
        if (status === 'error' || status === 'skipped') failed.push(id);
    }
    if (failed.length === 0) return;
    // Drop the failed entries from the results map so the progress
    // counter reflects the new attempt.
    for (const id of failed) batchQueue.results.delete(id);
    enqueueTasks(failed);
    persistBatchQueue();
}

function isBatchActive() {
    return Boolean(batchQueue.running) || batchQueue.queue.length > 0;
}

// ── Batch queue persistence ───────────────────────────────────────
//
// Queue + pause state is saved to localStorage on every meaningful
// change so a browser refresh (or a service-worker-style restart)
// comes back to the same place. We don't try to "resume" an active
// run — server-side jobs are gone anyway. Instead we restore the
// pending queue in paused mode and let the user press Continue.

function persistBatchQueue() {
    if (!batchQueue.queue.length && !batchQueue.running) {
        try { localStorage.removeItem(BATCH_QUEUE_STORAGE_KEY); } catch (_) {}
        return;
    }
    const payload = {
        v: 1,
        queue: batchQueue.queue,
        running: batchQueue.running,
        results: Array.from(batchQueue.results.entries()),
        totalThisRun: batchQueue.totalThisRun,
        startedAt: batchQueue.startedAt,
        totalPages: batchQueue.totalPages,
        completedPages: batchQueue.completedPages,
        // Always persist as paused on save; the user can hit 继续
        // after restart. Auto-resume would be confusing: the user
        // didn't ask for the run to continue.
        paused: true,
    };
    try { localStorage.setItem(BATCH_QUEUE_STORAGE_KEY, JSON.stringify(payload)); } catch (_) {}
}

function clearPersistedBatchQueue() {
    try { localStorage.removeItem(BATCH_QUEUE_STORAGE_KEY); } catch (_) {}
}

function restoreBatchQueueFromStorage() {
    let raw;
    try { raw = localStorage.getItem(BATCH_QUEUE_STORAGE_KEY); } catch (_) { return; }
    if (!raw) return;
    let payload;
    try { payload = JSON.parse(raw); } catch (_) { return; }
    if (!payload || payload.v !== 1) { clearPersistedBatchQueue(); return; }
    // Drop entries that no longer correspond to a real task (user may
    // have deleted them in the meantime).
    const knownIds = new Set(tasks.map((t) => t.id));
    const survivingQueue = Array.isArray(payload.queue) ? payload.queue.filter((id) => knownIds.has(id)) : [];
    const survivingRunning = payload.running && knownIds.has(payload.running) ? payload.running : null;
    if (survivingQueue.length === 0 && !survivingRunning) {
        clearPersistedBatchQueue();
        return;
    }
    batchQueue.queue = survivingQueue;
    batchQueue.running = null;            // never auto-resume an in-flight task
    batchQueue.results = new Map(Array.isArray(payload.results) ? payload.results : []);
    batchQueue.totalThisRun = Number(payload.totalThisRun) || (survivingQueue.length + (survivingRunning ? 1 : 0));
    batchQueue.startedAt = 0;             // don't carry over the elapsed clock
    batchQueue.totalPages = Number(payload.totalPages) || 0;
    batchQueue.completedPages = Number(payload.completedPages) || 0;
    batchQueue.paused = true;             // always come back paused
    batchQueue.stopping = false;
    renderBatchQueueBar();
    startBatchStatsTicker();
}

// 1-Hz ticker to keep the elapsed/ETA numbers honest even when the
// active task's progress events slow down. Started lazily by render
// and stopped when the bar hides itself.
let batchStatsTicker = null;
function startBatchStatsTicker() {
    if (batchStatsTicker) return;
    batchStatsTicker = setInterval(() => {
        if (!batchQueue.running && batchQueue.queue.length === 0) {
            stopBatchStatsTicker();
        }
        renderBatchQueueBar();
    }, 1000);
}
function stopBatchStatsTicker() {
    if (batchStatsTicker) {
        clearInterval(batchStatsTicker);
        batchStatsTicker = null;
    }
}

function renderBatchQueueBar({ finished = false } = {}) {
    const bar = els.batchQueueBar;
    if (!bar) return;

    const total = countQueueTotal();
    const completed = countDoneStatus('completed');
    const errored = countDoneStatus('error');
    const skipped = countDoneStatus('skipped');
    const remaining = batchQueue.queue.length + (batchQueue.running ? 1 : 0);
    const denominator = Math.max(total, batchQueue.totalThisRun, 1);
    const percent = Math.round(((total - remaining) / denominator) * 100);

    // Show / hide the whole bar based on whether a batch is active.
    const shouldShow = total > 1 || isBatchActive() || (finished && batchQueue.totalThisRun > 1);
    bar.classList.toggle('hidden', !shouldShow);
    if (!shouldShow) {
        // Reset run stats when the user has dismissed/finished a batch
        resetBatchRunStats();
        stopBatchStatsTicker();
        return;
    }
    if (isBatchActive()) startBatchStatsTicker();

    bar.dataset.state = batchQueue.paused ? 'paused' : (isBatchActive() ? 'running' : 'idle');
    bar.classList.toggle('is-paused', batchQueue.paused);

    if (els.batchQueueCounts) {
        els.batchQueueCounts.textContent = `${total - remaining}/${denominator}`;
    }
    if (els.batchQueueProgressFill) {
        els.batchQueueProgressFill.style.width = `${percent}%`;
    }
    if (els.batchQueueTitle) {
        if (batchQueue.paused) {
            els.batchQueueTitle.textContent = t('批量解析已暂停');
        } else if (!isBatchActive()) {
            els.batchQueueTitle.textContent = t('批量解析结束');
        } else {
            els.batchQueueTitle.textContent = t('批量解析中');
        }
    }
    if (els.batchQueueCurrent) {
        const runningTask = batchQueue.running ? tasks.find((t) => t.id === batchQueue.running) : null;
        const parts = [];
        if (runningTask) parts.push(t('当前：{name}', { name: runningTask.name || runningTask.id }));
        if (errored) parts.push(t('失败 {n}', { n: errored }));
        if (skipped) parts.push(t('跳过 {n}', { n: skipped }));
        if (!parts.length && !isBatchActive()) parts.push(t('已完成 {n} 个', { n: completed }));
        els.batchQueueCurrent.textContent = parts.join(' · ');
    }

    if (els.batchPauseBtn) {
        els.batchPauseBtn.disabled = !isBatchActive() && !batchQueue.paused;
        els.batchPauseBtn.textContent = batchQueue.paused ? t('继续') : t('暂停');
    }
    if (els.batchSkipBtn) {
        els.batchSkipBtn.disabled = !batchQueue.running;
    }
    if (els.batchStopBtn) {
        els.batchStopBtn.disabled = !isBatchActive();
    }
    if (els.batchRetryBtn) {
        const canRetry = (errored + skipped) > 0;
        els.batchRetryBtn.hidden = !canRetry;
        els.batchRetryBtn.disabled = isBatchActive() && !batchQueue.paused;
    }

    // ── Stats line: total pages / elapsed / ETA / per-page ─────────
    const stats = computeBatchStats();
    if (els.batchStatPages) {
        els.batchStatPages.textContent = t('{done}/{total} 页', { done: stats.completedPages, total: stats.totalPages });
    }
    if (els.batchStatElapsed) {
        els.batchStatElapsed.textContent = t('用时 {t}', { t: stats.elapsedLabel });
    }
    if (els.batchStatEta) {
        els.batchStatEta.textContent = t('剩余 {t}', { t: stats.etaLabel });
    }
    if (els.batchStatPerPage) {
        els.batchStatPerPage.textContent = t('{t}/页', { t: stats.perPageLabel });
    }
}

// Build the numbers behind the stats line in one place. The driver
// doesn't call this on every progress tick itself; renderBatchQueueBar
// re-runs it cheaply.
function computeBatchStats() {
    const startedAt = batchQueue.startedAt || 0;
    const elapsedMs = startedAt ? Date.now() - startedAt : 0;
    const totalPages = Math.max(batchQueue.totalPages, 0);
    const completedPages = Math.max(batchQueue.completedPages, 0);
    const remainingPages = Math.max(0, totalPages - completedPages);
    const perPageMs = completedPages > 0 ? elapsedMs / completedPages : 0;
    const etaMs = perPageMs > 0 ? Math.round(perPageMs * remainingPages) : 0;
    return {
        totalPages,
        completedPages,
        elapsedLabel: formatHms(elapsedMs),
        etaLabel: remainingPages > 0 && perPageMs > 0 ? formatHms(etaMs) : '--',
        perPageLabel: completedPages > 0 && perPageMs > 0 ? formatHms(perPageMs) : '--',
    };
}

function formatHms(ms) {
    const totalSec = Math.max(0, Math.round(ms / 1000));
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    const pad = (n) => String(n).padStart(2, '0');
    return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

async function ensureFolderByName(name) {
    const trimmed = (name || '').trim();
    if (!trimmed) return null;

    const existing = folders.find((f) => f.name === trimmed);
    if (existing) return existing;

    try {
        const resp = await fetch('/api/folders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: trimmed }),
        });
        if (!resp.ok) {
            console.warn('Failed to create folder for batch parse', await resp.text());
            return null;
        }
        const created = await resp.json();
        await loadFolders();
        return folders.find((f) => f.id === created.id) || created;
    } catch (err) {
        console.error('ensureFolderByName error', err);
        return null;
    }
}

function showIncomingFileState(fileList) {
    const filesToAdd = Array.from(fileList || []);
    const primaryFile = filesToAdd[0];
    const fileCount = filesToAdd.length;

    activeTaskId = null;
    isViewingTranslation = false;
    sourceRenderToken += 1;
    currentPdf = null;
    currentPage = 1;
    renderTaskList();
    resetResultRenderCache();
    resetResultScrollPositions();
    activeResultView = 'markdown';
    document.querySelectorAll('.view-tab').forEach((tab) => {
        tab.classList.toggle('active', tab.dataset.view === 'markdown');
    });
    showResultView('markdown');
    updateActionState(null);

    els.sourceTitle.textContent = primaryFile?.name || t('正在读取新文件');
    els.sourceMeta.textContent = fileCount > 1
        ? t('正在读取 {count} 个文件...', { count: fileCount })
        : t('正在读取文件...');
    els.pdfControls.classList.add('hidden');
    els.sourceViewer.innerHTML = `<div class="empty-result">${escapeHtml(t('正在读取文件，请稍候...'))}</div>`;
    els.sourceViewer.scrollTop = 0;
    els.resultTitle.textContent = t('准备解析');
    els.markdownView.innerHTML = `<div class="empty-result">${escapeHtml(t('正在读取新文件，解析结果会显示在这里。'))}</div>`;
    els.jsonView.textContent = '';
}

function assertUploadWithinLimit(fileOrBlob, filename = '') {
    const size = Number(fileOrBlob?.size || 0);
    if (!size) return;
    // Large files go through chunked upload, checked against maxTotalUploadBytes
    if (size > chunkedUploadThreshold) {
        if (size > maxTotalUploadBytes) {
            throw new Error(t('{name} 超过最大上传限制 {limit}，请压缩或拆分后再试。', {
                name: filename || fileOrBlob.name || t('文件'),
                limit: formatSize(maxTotalUploadBytes)
            }));
        }
        return;
    }
    // Small files use single upload, checked against maxUploadBytes
    if (maxUploadBytes && size > maxUploadBytes) {
        throw new Error(t('{name} 超过上传上限 {limit}，请压缩或拆分后再试。', {
            name: filename || fileOrBlob.name || t('文件'),
            limit: formatSize(maxUploadBytes)
        }));
    }
}

async function createTaskFromFile(file) {
    assertUploadWithinLimit(file);
    const ext = getExtension(file.name);
    const officeExts = ['ppt', 'pptx', 'doc', 'docx'];
    const imageExts = ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff', 'tif', 'gif'];

    if (officeExts.includes(ext)) {
        const converted = await convertOfficeToPdf(file);
        return createPdfTask(converted.blob, file.name.replace(/\.[^.]+$/, '.pdf'), {
            originalName: file.name,
            sourceKind: 'office'
        });
    }

    if (ext === 'pdf' || file.type === 'application/pdf') {
        return createPdfTask(file, file.name, { sourceKind: 'pdf' });
    }

    if (imageExts.includes(ext)) {
        return createImageTask(file);
    }

    alert(t('不支持的文件格式：{name}', { name: file.name }));
    throw new Error(`Unsupported file type: ${file.name}`);
}

async function createImageTask(file) {
    const id = createId();
    const dataUrl = await readAsDataUrl(file);
    const sourceUrl = await uploadTaskSource(id, file, file.name, file.type || 'application/octet-stream');
    const now = Date.now();
    const task = {
        id,
        name: file.name,
        sourceKind: 'image',
        mimeType: file.type || 'image/*',
        size: file.size,
        createdAt: now,
        updatedAt: now,
        status: 'pending',
        pageCount: 1,
        sourceUrl,
        sourceDataUrl: dataUrl,
        thumbnail: dataUrl,
        batches: [{
            id: createId(),
            label: formatPageLabel(1),
            fileType: 1,
            pageCount: 1,
            payloadDataUrl: dataUrl,
            status: 'pending'
        }],
        markdown: '',
        images: {},
        ocrResults: []
    };
    applySelectedModelToTask(task);
    return task;
}

async function createPdfTask(fileOrBlob, name, extra = {}) {
    const id = createId();
    const size = Number(fileOrBlob.size || 0);

    // For large files: upload first, then get page count from server
    // This avoids loading the entire PDF into browser memory
    if (size > chunkedUploadThreshold) {
        const sourceUrl = await uploadTaskSource(id, fileOrBlob, name, 'application/pdf');
        const info = await fetchSourceInfo(id);
        const pageCount = info.pageCount || 1;
        const thumbnail = await renderPDFPageDataUrlFromSource(id, 1, 0.35) || '';
        const pdfBatchSize = getConfiguredPdfBatchSize();
        const batches = createPdfBatchDescriptors(pageCount, pdfBatchSize);

        const now = Date.now();
        const task = {
            id,
            name,
            sourceKind: extra.sourceKind || 'pdf',
            originalName: extra.originalName || name,
            mimeType: 'application/pdf',
            size,
            createdAt: now,
            updatedAt: now,
            status: 'pending',
            pageCount,
            pdfBatchSize,
            sourceUrl,
            thumbnail,
            batches,
            markdown: '',
            images: {},
            ocrResults: []
        };
        applySelectedModelToTask(task);
        return task;
    }

    // Small files: use the existing client-side PDF-lib path
    const arrayBuffer = await fileOrBlob.arrayBuffer();
    const sourceUrl = await uploadTaskSource(id, fileOrBlob, name, 'application/pdf');
    const pdf = await loadPdf(arrayBuffer.slice(0));
    const pageCount = pdf.numPages;
    const thumbnail = await renderPDFPageDataUrl(pdf, 1, 0.35);
    const pdfBatchSize = getConfiguredPdfBatchSize();
    const batches = createPdfBatchDescriptors(pageCount, pdfBatchSize);

    const now = Date.now();
    const task = {
        id,
        name,
        sourceKind: extra.sourceKind || 'pdf',
        originalName: extra.originalName || name,
        mimeType: 'application/pdf',
        size: fileOrBlob.size || arrayBuffer.byteLength,
        createdAt: now,
        updatedAt: now,
        status: 'pending',
        pageCount,
        pdfBatchSize,
        sourceUrl,
        thumbnail,
        batches,
        markdown: '',
        images: {},
        ocrResults: []
    };
    applySelectedModelToTask(task);
    return task;
}

async function uploadTaskSource(taskId, fileOrBlob, filename, mimeType) {
    const size = Number(fileOrBlob?.size || 0);
    // Use chunked upload for large files
    if (size > chunkedUploadThreshold) {
        return chunkedUpload(taskId, fileOrBlob, filename, mimeType);
    }
    assertUploadWithinLimit(fileOrBlob, filename);
    const formData = new FormData();
    const source = fileOrBlob instanceof File
        ? fileOrBlob
        : new File([fileOrBlob], filename, { type: mimeType || fileOrBlob.type || 'application/octet-stream' });
    formData.append('file', source, filename);
    const response = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}/source`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        throw new Error(t('保存源文件失败：{detail}', { detail: await responseErrorText(response) }));
    }
    const data = await response.json();
    return data.url;
}

async function chunkedUpload(taskId, fileOrBlob, filename, mimeType) {
    const size = Number(fileOrBlob.size || 0);
    if (size > maxTotalUploadBytes) {
        throw new Error(t('{name} 超过最大上传限制 {limit}，请压缩或拆分后再试。', {
            name: filename,
            limit: formatSize(maxTotalUploadBytes)
        }));
    }

    // Create upload session
    const createResponse = await apiFetch(`${API_BASE}/uploads`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filename,
            totalSize: size,
            chunkSize: defaultChunkSize,
            taskId
        })
    });
    if (!createResponse.ok) {
        throw new Error(t('创建上传会话失败：{detail}', { detail: await responseErrorText(createResponse) }));
    }
    const session = await createResponse.json();
    const { uploadId, chunkSize, totalChunks } = session;

    // Check for existing chunks (resume support)
    const statusResponse = await apiFetch(`${API_BASE}/uploads/${encodeURIComponent(uploadId)}`);
    let receivedChunks = new Set();
    if (statusResponse.ok) {
        const status = await statusResponse.json();
        receivedChunks = new Set(status.receivedChunks || []);
    }

    // Upload missing chunks sequentially
    for (let index = 0; index < totalChunks; index++) {
        if (receivedChunks.has(index)) continue;

        const offset = index * chunkSize;
        const end = Math.min(offset + chunkSize, size);
        const chunkBlob = fileOrBlob.slice(offset, end);

        const chunkFormData = new FormData();
        chunkFormData.append('file', chunkBlob, `chunk_${index}`);

        const chunkResponse = await apiFetch(
            `${API_BASE}/uploads/${encodeURIComponent(uploadId)}/chunks/${index}`,
            { method: 'PUT', body: chunkFormData }
        );
        if (!chunkResponse.ok) {
            throw new Error(t('上传分片 {index}/{total} 失败：{detail}', {
                index: index + 1,
                total: totalChunks,
                detail: await responseErrorText(chunkResponse)
            }));
        }

        // Update progress
        const uploadedBytes = Math.min((index + 1) * chunkSize, size);
        updateUploadProgress(taskId, uploadedBytes, size);
    }

    // Complete upload
    const completeResponse = await apiFetch(
        `${API_BASE}/uploads/${encodeURIComponent(uploadId)}/complete`,
        { method: 'POST' }
    );
    if (!completeResponse.ok) {
        throw new Error(t('完成上传失败：{detail}', { detail: await responseErrorText(completeResponse) }));
    }

    const result = await completeResponse.json();
    updateUploadProgress(taskId, size, size); // 100%
    return result.url;
}

function updateUploadProgress(taskId, uploadedBytes, totalBytes) {
    const task = tasks.find((t) => t.id === taskId);
    if (!task) return;
    const percent = totalBytes > 0 ? Math.round(uploadedBytes / totalBytes * 100) : 0;
    if (els.sourceMeta && activeTaskId === taskId) {
        els.sourceMeta.textContent = t('上传中 {percent}% ({uploaded}/{total})', {
            percent,
            uploaded: formatSize(uploadedBytes),
            total: formatSize(totalBytes)
        });
    }
}

async function fetchSourceInfo(taskId) {
    const response = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}/source/info`);
    if (!response.ok) {
        throw new Error(t('获取源文件信息失败：{detail}', { detail: await responseErrorText(response) }));
    }
    return response.json();
}

async function convertOfficeToPdf(file) {
    assertUploadWithinLimit(file);
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiFetch(`${API_BASE}/convert/to-pdf`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        const detail = await responseErrorText(response);
        throw new Error(t('Office 转 PDF 失败：{detail}', { detail }));
    }
    return { blob: await response.blob() };
}

function renderTaskList() {
    const keyword = els.taskSearch.value.trim().toLowerCase();
    els.taskList.innerHTML = '';
    const visibleTasks = tasks.filter((task) => {
        if (activeFilter === 'done' && task.status !== 'completed') return false;
        // Folder filter: show only tasks in the selected folder
        if (activeFolderId) {
            if (task.folderId !== activeFolderId) return false;
        } else {
            // "全部文件" shows everything
        }
        return !keyword || task.name.toLowerCase().includes(keyword);
    });

    if (visibleTasks.length === 0) {
        els.taskList.innerHTML = `<div class="task-empty">${escapeHtml(t('暂无任务'))}</div>`;
        return;
    }

    for (const task of visibleTasks) {
        const clone = els.taskTemplate.content.cloneNode(true);
        const item = clone.querySelector('.task-item');
        item.dataset.taskId = task.id;
        item.classList.toggle('active', task.id === activeTaskId);
        item.classList.add(`status-${taskVisualStatus(task)}`);
        const checkbox = item.querySelector('.task-checkbox');
        if (checkbox) {
            checkbox.checked = selectedTaskIds.has(task.id);
            if (checkbox.checked) item.classList.add('checked');
            checkbox.addEventListener('click', (event) => event.stopPropagation());
            checkbox.addEventListener('change', () => {
                if (checkbox.checked) selectedTaskIds.add(task.id);
                else selectedTaskIds.delete(task.id);
                item.classList.toggle('checked', checkbox.checked);
                updateSelectionBar();
            });
        }
        item.querySelector('.task-icon').innerHTML = taskIcon(task);
        item.querySelector('.task-name').textContent = task.name;
        const folderTag = task.folderName
            ? ` · <span class="folder-tag">${escapeHtml(task.folderName)}</span>`
            : '';
        item.querySelector('.task-meta').innerHTML = `${formatDate(task.updatedAt)} · ${formatPageCount(task.pageCount || 1)}${folderTag}`;
        item.querySelector('.task-state').textContent = statusText(task);
        const deleteButton = item.querySelector('.task-delete');
        deleteButton.setAttribute('title', t('删除任务'));
        deleteButton.setAttribute('aria-label', t('删除任务'));
        item.addEventListener('click', (event) => {
            // In selection mode, clicking the row toggles the checkbox
            // rather than opening the task — except clicks on the row's
            // own controls (handled via stopPropagation above).
            if (taskSelectionMode) {
                const cb = item.querySelector('.task-checkbox');
                if (cb) {
                    cb.checked = !cb.checked;
                    cb.dispatchEvent(new Event('change'));
                }
                return;
            }
            selectTask(task.id);
        });
        item.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                if (taskSelectionMode) {
                    const cb = item.querySelector('.task-checkbox');
                    if (cb) {
                        cb.checked = !cb.checked;
                        cb.dispatchEvent(new Event('change'));
                    }
                    return;
                }
                selectTask(task.id);
            }
        });
        deleteButton.addEventListener('click', async (event) => {
            event.stopPropagation();
            await deleteTask(task.id);
        });
        // Drag support for moving task to folder (disabled while
        // multi-selecting so users don't accidentally drag a checked row).
        if (!taskSelectionMode) setupTaskDragDrop(item, task.id);
        // Right-click context menu for folder assignment
        item.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            showTaskFolderMenu(e, task);
        });
        els.taskList.appendChild(item);
    }
}

// ── Task multi-selection ──────────────────────────────────────────
//
// Lets the user pick existing tasks from the sidebar and run them in a
// single batch. Drives the same batchQueue used by the upload path so
// pause / skip / retry all keep working.

function getVisibleTasks() {
    const keyword = els.taskSearch.value.trim().toLowerCase();
    return tasks.filter((task) => {
        if (activeFilter === 'done' && task.status !== 'completed') return false;
        if (activeFolderId && task.folderId !== activeFolderId) return false;
        return !keyword || task.name.toLowerCase().includes(keyword);
    });
}

function toggleTaskSelectionMode(forceOn) {
    const next = typeof forceOn === 'boolean' ? forceOn : !taskSelectionMode;
    taskSelectionMode = next;
    document.body.classList.toggle('task-select-mode', next);
    if (!next) clearTaskSelection();
    if (els.taskSelectBar) els.taskSelectBar.classList.toggle('hidden', !next);
    if (els.taskSelectModeBtn) {
        els.taskSelectModeBtn.classList.toggle('success', next);
        els.taskSelectModeBtn.setAttribute(
            'title',
            next ? t('退出多选') : t('多选 / 批量解析现有文件')
        );
    }
    renderTaskList();
    updateSelectionBar();
}

function clearTaskSelection() {
    selectedTaskIds.clear();
}

function selectAllVisibleTasks() {
    selectedTaskIds.clear();
    for (const task of getVisibleTasks()) selectedTaskIds.add(task.id);
}

function updateSelectionBar() {
    if (!els.taskSelectBar) return;
    const count = selectedTaskIds.size;
    const visibleCount = getVisibleTasks().length;
    if (els.taskSelectCounter) {
        els.taskSelectCounter.textContent = t('已选 {n}', { n: count });
    }
    if (els.taskSelectAllCb) {
        els.taskSelectAllCb.checked = count > 0 && count === visibleCount;
        els.taskSelectAllCb.indeterminate = count > 0 && count < visibleCount;
    }
    const disabled = count === 0;
    if (els.taskSelectParse) els.taskSelectParse.disabled = disabled;
    if (els.taskSelectMove) els.taskSelectMove.disabled = disabled;
    if (els.taskSelectDelete) els.taskSelectDelete.disabled = disabled;
}

function batchParseSelected() {
    if (selectedTaskIds.size === 0) return;
    const ids = Array.from(selectedTaskIds);
    // enqueueTasks → processTask(confirmCompleted:false) — already-done
    // tasks get re-parsed from scratch without an "are you sure?" prompt.
    enqueueTasks(ids);
    toggleTaskSelectionMode(false);
}

async function batchMoveSelected() {
    if (selectedTaskIds.size === 0) return;
    const choices = [t('（移到根目录 / 全部文件）'), ...folders.map((f) => f.name)];
    const message = t('要把选中的 {n} 个文件移到哪个文件夹？\n输入序号：\n{list}', {
        n: selectedTaskIds.size,
        list: choices.map((label, idx) => `${idx}. ${label}`).join('\n'),
    });
    const raw = prompt(message, '0');
    if (raw == null) return;
    const idx = Number.parseInt(raw, 10);
    if (Number.isNaN(idx) || idx < 0 || idx >= choices.length) {
        alert(t('无效的序号。'));
        return;
    }
    const targetFolderId = idx === 0 ? null : folders[idx - 1].id;
    const ids = Array.from(selectedTaskIds);
    for (const id of ids) {
        try {
            await moveTaskToFolder(id, targetFolderId);
        } catch (err) {
            console.error('Batch move failed for task', id, err);
        }
    }
    await loadFolders();
    renderTaskList();
    toggleTaskSelectionMode(false);
}

async function batchDeleteSelected() {
    if (selectedTaskIds.size === 0) return;
    if (!confirm(t('确定删除选中的 {n} 个文件？此操作不可撤销。', { n: selectedTaskIds.size }))) return;
    const ids = Array.from(selectedTaskIds);
    for (const id of ids) {
        try {
            await deleteTaskById(id);
        } catch (err) {
            console.error('Batch delete failed for task', id, err);
        }
    }
    tasks = tasks.filter((task) => !selectedTaskIds.has(task.id));
    if (activeTaskId && selectedTaskIds.has(activeTaskId)) {
        activeTaskId = tasks[0]?.id || null;
        if (activeTaskId) await selectTask(activeTaskId);
        else resetWorkbench();
    }
    toggleTaskSelectionMode(false);
}

async function deleteTask(taskId) {
    const task = tasks.find((item) => item.id === taskId);
    if (!task) return;
    if (isProcessing && task.id === activeTaskId) {
        alert(t('当前文件正在解析中，完成后再删除。'));
        return;
    }
    if (task.status === 'processing' && !shouldResumeTask(task)) {
        alert(t('当前文件正在解析中，完成后再删除。'));
        return;
    }
    if (!confirm(t('确定要删除“{name}”吗？当前操作不可回撤。', { name: task.name }))) return;

    const wasActive = activeTaskId === taskId;
    try {
        await deleteTaskById(taskId);
    } catch (error) {
        console.error(error);
        alert(error.message || t('删除失败，请稍后重试。'));
        return;
    }
    tasks = tasks.filter((item) => item.id !== taskId);

    if (!wasActive) {
        renderTaskList();
        return;
    }

    activeTaskId = tasks[0]?.id || null;
    if (activeTaskId) {
        await selectTask(activeTaskId);
    } else {
        resetWorkbench();
    }
}

// Lightweight selection: populate the right pane and metadata without
// eagerly rendering the source PDF. Used on initial load so a large
// PDF doesn't freeze the page before the user has a chance to look at
// the task list. The full renderSource() still runs once the user
// clicks anywhere on the task again.
async function selectTaskLight(taskId) {
    activeTaskId = taskId;
    lastActiveTaskId = taskId;
    try { localStorage.setItem(ACTIVE_TASK_STORAGE_KEY, taskId); } catch (_) {}
    closeMobileSidebar();
    renderTaskList();
    let task;
    try {
        task = await ensureTaskLoaded(taskId);
    } catch (error) {
        console.error(error);
        return;
    }
    if (activeTaskId !== taskId || !task) return;
    updateActiveModelDisplay(task);
    els.sourceTitle.textContent = task.name;
    els.sourceMeta.textContent = taskSourceMeta(task);
    els.resultTitle.textContent = resultPaneTitle(task);
    renderResultPane(task);
    updateActionState(task);
    // Show a soft "click to load" placeholder in the source pane so
    // the user knows the file is there but unrendered.
    showSourcePreviewPlaceholder(task);
}

function showSourcePreviewPlaceholder(task) {
    const viewer = els.sourceViewer;
    if (!viewer) return;
    currentPdf = null;
    els.pdfControls.classList.add('hidden');
    viewer.innerHTML = '';
    const pageCount = task.pageCount || 1;
    const sizeLabel = formatSize(task.size || 0);
    // Thresholds tuned for the in-browser PDF.js path: rendering 200+
    // pages of placeholder DIVs + IntersectionObservers locks the main
    // thread for a noticeable amount of time, and the user gets little
    // out of it (they're usually about to look at the parsed result).
    // Anything past this size should be viewed via the result pane or
    // downloaded, not rendered in the browser.
    const SOFT_LIMIT_PAGES = 200;
    const SOFT_LIMIT_BYTES = 80 * 1024 * 1024;     // 80 MB
    const isLarge = pageCount > SOFT_LIMIT_PAGES || (task.size || 0) > SOFT_LIMIT_BYTES;
    const placeholder = document.createElement('div');
    placeholder.className = 'source-preview-placeholder';
    placeholder.innerHTML = `
        <div class="source-preview-card">
            <svg viewBox="0 0 24 24" width="44" height="44">
                <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
                <path d="M14 3v6h6"/>
            </svg>
            <div class="source-preview-meta">
                <strong>${escapeHtml(task.name)}</strong>
                <span>${formatPageCount(pageCount)} · ${sizeLabel}${isLarge ? ' · 大文件' : ''}</span>
            </div>
            ${isLarge
                ? `<p class="source-preview-warning">此文件较大（${formatPageCount(pageCount)} / ${sizeLabel}），直接在浏览器中渲染可能卡顿。建议：</p>
                   <ul class="source-preview-tips">
                       <li>直接在右侧查看解析结果</li>
                       <li>点上方「下载」保存 Markdown 后用外部阅读器查看</li>
                       <li>如确需对照源文件，点击下方「仍要加载」</li>
                   </ul>
                   <div class="source-preview-actions">
                       <button type="button" class="primary-button" id="source-preview-load">仍要加载源文件</button>
                   </div>`
                : `<button type="button" class="primary-button" id="source-preview-load">加载源文件</button>`
            }
        </div>
    `;
    viewer.appendChild(placeholder);
    placeholder.querySelector('#source-preview-load')?.addEventListener('click', () => {
        // Re-issue a full selectTask which will actually render the PDF.
        selectTask(taskId);
    });
    viewer.scrollTop = 0;
    viewer.scrollLeft = 0;
    els.markdownView.scrollLeft = 0;
}

function showEmptyWorkbench() {
    activeTaskId = null;
    lastActiveTaskId = null;
    try { localStorage.removeItem(ACTIVE_TASK_STORAGE_KEY); } catch (_) {}
    const viewer = els.sourceViewer;
    if (viewer) {
        currentPdf = null;
        els.pdfControls.classList.add('hidden');
        viewer.innerHTML = `<div class="drop-zone" id="drop-zone">
            <svg viewBox="0 0 24 24"><path d="M12 3v12M7 8l5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2 2v-4"/></svg>
            <h3>拖拽文件到这里</h3>
            <p>支持 PDF、图片、PPT/PPTX、DOC/DOCX；PDF 会逐页解析。</p>
            <button class="primary-button" id="browse-btn">选择文件</button>
        </div>`;
        // Re-wire browse button (innerHTML wipes listeners).
        document.getElementById('browse-btn')?.addEventListener('click', () => els.fileInput.click());
    }
    if (els.sourceTitle) els.sourceTitle.textContent = t('等待上传文件');
    if (els.sourceMeta) els.sourceMeta.textContent = t('PDF、图片、Office 文档');
    if (els.resultTitle) els.resultTitle.textContent = t('解析结果');
    if (els.markdownView) {
        els.markdownView.innerHTML = `<div class="empty-result">${escapeHtml(t('选择左侧任务，或上传一个新文件开始解析。'))}</div>`;
    }
    if (els.jsonView) els.jsonView.textContent = '';
    updateActionState(null);
}

async function selectTask(taskId) {
    activeTaskId = taskId;
    lastActiveTaskId = taskId;
    try { localStorage.setItem(ACTIVE_TASK_STORAGE_KEY, taskId); } catch (_) {}
    closeMobileSidebar();
    renderTaskList();
    let task;
    try {
        task = await ensureTaskLoaded(taskId);
    } catch (error) {
        console.error(error);
        els.sourceTitle.textContent = t('任务加载失败');
        els.sourceMeta.textContent = '';
        els.resultTitle.textContent = t('加载失败');
        els.markdownView.textContent = error.message || t('任务详情加载失败');
        updateActionState(null);
        return;
    }
    if (activeTaskId !== taskId) return;
    if (!task) return;

    // Lazily load heavy result data (images, ocrResults) if available.
    // The lite endpoint omits these to avoid OOM on large results.
    // We load images first (needed for markdown rendering), then defer
    // ocrResults until the user actually needs them.
    const resultState = task._resultState;
    if (!task._resultLoaded) {
        if (resultState && (resultState.hasImages || resultState.hasOcrResults)) {
            try {
                // Load images in batches — they're needed for markdown rendering
                const resultData = await loadTaskResult(taskId, {
                    fields: 'images',
                    imageLimit: 200,
                });
                if (resultData.images && typeof resultData.images === 'object') {
                    task.images = { ...task.images, ...resultData.images };
                }
                task._imagesLoaded = true;
                task._imageTotal = resultData.imageTotal || 0;
                // Defer ocrResults — load on demand when user switches to JSON view
                task._ocrLoaded = false;
                task._ocrTotal = resultState.hasOcrResults ? -1 : 0; // -1 = unknown, needs loading
            } catch (err) {
                console.warn('Failed to lazy-load task images:', err);
            }
        } else {
            // No heavy data to load or already present in lite response
            task._imagesLoaded = true;
            task._ocrLoaded = true;
        }
        task._resultLoaded = true;
    }

    renderTaskList();
    updateActiveModelDisplay(task);
    els.sourceTitle.textContent = task.name;
    els.sourceMeta.textContent = taskSourceMeta(task);
    els.resultTitle.textContent = resultPaneTitle(task);
    const deferPPOCRVisualResult = isPPOCRVisualTask(task) && task.sourceKind !== 'image';
    if (!deferPPOCRVisualResult) {
        renderResultPane(task);
    } else {
        resetResultRenderCache(task.id);
        resetResultScrollPositions();
        updateResultViewLabels(task);
        syncResultMode(task);
        showResultView('markdown');
    }
    updateActionState(task);
    await renderSource(task);
    if (activeTaskId !== taskId) return;
    if (deferPPOCRVisualResult) {
        invalidatePPOCRVisualRender();
    }
    renderResultPane(task);
    updateActionState(task);

    // Auto-show saved translation if available
    if (task.translation && task.translationLang) {
        isViewingTranslation = true;
        showTranslationResult(task);
    } else {
        isViewingTranslation = false;
    }
}

function getActiveTask() {
    return tasks.find((task) => task.id === activeTaskId);
}

async function renderSource(task) {
    const renderToken = ++sourceRenderToken;
    currentPdf = null;
    els.pdfControls.classList.add('hidden');
    els.sourceViewer.innerHTML = '';
    els.sourceViewer.scrollTop = 0;
    els.sourceViewer.scrollLeft = 0;
    els.markdownView.scrollLeft = 0;

    if (task.sourceKind === 'image') {
        const wrap = document.createElement('div');
        wrap.className = 'source-image-wrap';
        wrap.dataset.page = '1';
        const imageBox = document.createElement('div');
        imageBox.className = 'source-image-box';
        const img = document.createElement('img');
        img.className = 'source-image';
        img.src = task.sourceDataUrl || task.sourceUrl;
        imageBox.appendChild(img);
        const highlightLayer = document.createElement('div');
        highlightLayer.className = 'pdf-highlight-layer';
        imageBox.appendChild(highlightLayer);
        wrap.appendChild(imageBox);
        els.sourceViewer.appendChild(wrap);
        await waitForImageReady(img);
        return;
    }

    currentPage = Math.min(Math.max(currentPage, 1), task.pageCount || 1);
    els.pdfControls.classList.add('hidden');

    // Two paths:
    //  1. Server-rendered thumbs (default for large / normal PDFs):
    //     the backend pre-renders every page to a small PNG. The
    //     browser never sees the whole PDF — no PDF.js, no decode
    //     storm, no main-thread freeze.
    //  2. PDF.js (fallback for dataUrl tasks where we already have
    //     the bytes client-side, or for environments where the
    //     server doesn't expose /thumbs).
    let thumbsMeta = null;
    if (!task.sourceDataUrl) {
        try {
            const metaResp = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(task.id)}/thumbs`);
            if (metaResp.ok) {
                thumbsMeta = await metaResp.json();
            }
        } catch (_) { thumbsMeta = null; }
    }
    if (thumbsMeta && thumbsMeta.pages && thumbsMeta.pages.length > 0) {
        await renderThumbsDocument(task, thumbsMeta, renderToken);
        return;
    }
    // Fallback: PDF.js (kept for sourceDataUrl uploads).
    await renderPdfJsSource(task, renderToken);
}

// Render every page of a PDF as a list of <img> tags. Each image is
// lazy-loaded by IntersectionObserver so the browser only fetches the
// page the user is actually looking at. The source PDF never enters
// browser memory — only the small PNGs do.
async function renderThumbsDocument(task, meta, renderToken) {
    if (renderToken !== sourceRenderToken) return;
    currentPdf = null;
    els.pdfControls.classList.remove('hidden');
    els.sourceViewer.innerHTML = '';
    els.sourceViewer.scrollTop = 0;
    els.sourceViewer.scrollLeft = 0;
    els.markdownView.scrollLeft = 0;

    const flow = document.createElement('div');
    flow.className = 'pdf-document-flow';
    els.sourceViewer.appendChild(flow);

    const pages = meta.pages;
    const A4_HEIGHT_PX = 1240;        // matches the default 90-DPI thumb
    const pageWrappers = [];
    for (const page of pages) {
        const wrap = document.createElement('div');
        wrap.className = 'pdf-page-wrap pdf-page-placeholder';
        wrap.dataset.page = String(page.page);
        // Use the page's actual aspect ratio so the placeholder is the
        // right height before the image arrives — no jumpy layout.
        const aspect = page.heightPt / Math.max(1, page.widthPt);
        wrap.style.minHeight = `${Math.round(A4_HEIGHT_PX * aspect)}px`;
        const canvasBox = document.createElement('div');
        canvasBox.className = 'pdf-canvas-box';
        const placeholderText = document.createElement('div');
        placeholderText.className = 'pdf-page-placeholder-text';
        placeholderText.textContent = `${page.page}`;
        canvasBox.appendChild(placeholderText);
        const highlightLayer = document.createElement('div');
        highlightLayer.className = 'pdf-highlight-layer';
        canvasBox.appendChild(highlightLayer);
        wrap.appendChild(canvasBox);
        flow.appendChild(wrap);
        pageWrappers[page.page] = wrap;
    }
    observeThumbPages(task, pageWrappers);
    scrollPdfPageIntoView(currentPage, 'auto');
    resetSplitHorizontalScroll();
    updateCurrentPageFromScroll();
}

// Lazy-load thumbs as they enter the viewport. ~3-page buffer either
// side keeps scrolling smooth without preloading the whole book.
function observeThumbPages(task, pageWrappers) {
    if (thumbObserver) thumbObserver.disconnect();
    thumbObserver = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (entry.isIntersecting) {
                const wrap = entry.target;
                if (wrap.dataset.thumbLoaded === '1') continue;
                wrap.dataset.thumbLoaded = '1';
                const pageNumber = Number(wrap.dataset.page);
                const img = document.createElement('img');
                img.className = 'thumb-image';
                img.alt = `${task.name} - page ${pageNumber}`;
                img.loading = 'lazy';
                img.src = `${API_BASE}/tasks/${encodeURIComponent(task.id)}/thumb/${pageNumber}`;
                img.onload = () => {
                    wrap.classList.remove('pdf-page-placeholder');
                    const text = wrap.querySelector('.pdf-page-placeholder-text');
                    if (text) text.remove();
                };
                const box = wrap.querySelector('.pdf-canvas-box');
                if (box) box.appendChild(img);
            }
        }
    }, { root: els.sourceViewer, rootMargin: '300px 0px', threshold: 0.01 });
    Object.values(pageWrappers).forEach((wrap) => thumbObserver.observe(wrap));
}

let thumbObserver = null;

// Keep renderSource for the dataUrl / PDF.js fallback path.
async function renderPdfJsSource(task, renderToken) {
    // Show a progress bar with a Cancel button so a slow render
    // doesn't look like a frozen page.
    const showProgress = (label) => {
        els.sourceViewer.innerHTML = `
            <div class="source-load-progress">
                <div class="source-load-spinner"></div>
                <div class="source-load-label">${escapeHtml(label)}</div>
                <div class="source-load-bar"><div class="source-load-bar-fill"></div></div>
                <button type="button" class="secondary-button" id="source-load-cancel">取消</button>
            </div>
        `;
    };
    showProgress(t('正在加载 PDF...'));
    const cancelBtn = () => document.getElementById('source-load-cancel');
    let cancelled = false;
    const onCancel = () => { cancelled = true; sourceRenderToken++; };
    cancelBtn()?.addEventListener('click', onCancel);

    let pdf;
    try {
        if (task.sourceDataUrl) {
            pdf = await loadPdf(dataUrlToUint8Array(task.sourceDataUrl));
        } else {
            pdf = await loadPdfWithProgress(task.sourceUrl, ({ loaded, total }) => {
                if (cancelled) throw new Error('cancelled');
                if (total > 0) {
                    const fill = els.sourceViewer.querySelector('.source-load-bar-fill');
                    const label = els.sourceViewer.querySelector('.source-load-label');
                    if (fill) fill.style.width = `${Math.min(100, Math.round(loaded / total * 100))}%`;
                    if (label) {
                        label.textContent = t('正在下载 {done} / {total}', {
                            done: formatSize(loaded), total: formatSize(total)
                        });
                    }
                } else {
                    const label = els.sourceViewer.querySelector('.source-load-label');
                    if (label) label.textContent = t('正在下载… {done}', { done: formatSize(loaded) });
                }
            });
        }
    } catch (err) {
        if (cancelled) {
            if (renderToken === sourceRenderToken) showSourcePreviewPlaceholder(task);
            return;
        }
        console.error('PDF load failed', err);
        if (renderToken === sourceRenderToken) {
            els.sourceViewer.innerHTML = `<div class="empty-result">${escapeHtml(err.message || t('PDF 加载失败'))}</div>`;
        }
        return;
    }
    if (renderToken !== sourceRenderToken) return;
    const firstPage = await pdf.getPage(1);
    if (renderToken !== sourceRenderToken) return;
    pdfDefaultPageWidth = firstPage.getViewport({ scale: 1 }).width || PDF_DEFAULT_PAGE_WIDTH;
    currentZoom = getDefaultPdfZoom();
    currentPdf = pdf;
    els.pdfControls.classList.remove('hidden');
    await renderPdfDocument(renderToken);
}

async function renderPdfDocument(renderToken = sourceRenderToken, scrollAnchor = null) {
    if (renderToken !== sourceRenderToken) return;
    if (!currentPdf) return;
    currentPage = Math.min(Math.max(currentPage, 1), currentPdf.numPages);
    updatePdfControls();
    els.sourceViewer.innerHTML = '';
    const flow = document.createElement('div');
    flow.className = 'pdf-document-flow';
    els.sourceViewer.appendChild(flow);

    const totalPages = currentPdf.numPages;
    const PDF_PLACEHOLDER_HEIGHT = 842;
    const estimatedHeight = PDF_PLACEHOLDER_HEIGHT * currentZoom;

    // Build all wrappers in a DocumentFragment for single DOM insertion
    const fragment = document.createDocumentFragment();
    const pageWrappers = []; // Direct reference — no querySelector needed
    for (let pageNumber = 1; pageNumber <= totalPages; pageNumber += 1) {
        const wrap = document.createElement('div');
        wrap.className = 'pdf-page-wrap pdf-page-placeholder';
        wrap.dataset.page = String(pageNumber);
        wrap.dataset.rendered = 'false';
        wrap.style.minHeight = `${estimatedHeight}px`;
        const canvasBox = document.createElement('div');
        canvasBox.className = 'pdf-canvas-box';
        const placeholderText = document.createElement('div');
        placeholderText.className = 'pdf-page-placeholder-text';
        placeholderText.textContent = `${pageNumber}`;
        canvasBox.appendChild(placeholderText);
        const highlightLayer = document.createElement('div');
        highlightLayer.className = 'pdf-highlight-layer';
        canvasBox.appendChild(highlightLayer);
        wrap.appendChild(canvasBox);
        fragment.appendChild(wrap);
        pageWrappers[pageNumber] = wrap;
    }
    flow.appendChild(fragment);

    observePdfPages(renderToken, pageWrappers);

    if (scrollAnchor) {
        restoreSourceScrollAnchor(scrollAnchor, 'auto');
    } else {
        scrollPdfPageIntoView(currentPage, 'auto');
        resetSplitHorizontalScroll();
    }
    updateCurrentPageFromScroll();
}

// Virtual scrolling: priority queue + cancel-on-scroll-away
const PDF_RENDER_BUFFER = 2;
const PDF_PREFETCH_BUFFER = 6;
const RECYCLE_THRESHOLD = 12;
let pdfPageObserver = null;
let pdfRenderedPages = new Set();
let pdfPageWrappers = [];
let pdfRenderQueue = [];
let pdfRenderInFlight = false;

function observePdfPages(renderToken, pageWrappers) {
    if (pdfPageObserver) pdfPageObserver.disconnect();
    pdfRenderedPages.clear();
    pdfPageWrappers = pageWrappers || [];
    pdfRenderQueue = [];
    pdfRenderInFlight = false;

    pdfPageObserver = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            const pageNumber = parseInt(entry.target.dataset.page, 10);
            if (pageNumber) schedulePdfPageRender(pageNumber, renderToken);
        }
    }, {
        root: els.sourceViewer,
        rootMargin: '150% 0px',
        threshold: 0
    });

    for (let i = 1; i < pdfPageWrappers.length; i++) {
        if (pdfPageWrappers[i]) pdfPageObserver.observe(pdfPageWrappers[i]);
    }
}

function schedulePdfPageRender(centerPage, renderToken) {
    if (!currentPdf) return;
    const totalPages = currentPdf.numPages;

    // Drop queue items that are now far from current viewport
    pdfRenderQueue = pdfRenderQueue.filter((item) =>
        Math.abs(item.page - centerPage) <= PDF_PREFETCH_BUFFER + 2
    );

    for (let offset = 0; offset <= PDF_PREFETCH_BUFFER; offset++) {
        for (const page of [centerPage + offset, centerPage - offset]) {
            if (page < 1 || page > totalPages) continue;
            if (pdfRenderedPages.has(page)) continue;
            const wrap = pdfPageWrappers[page];
            if (!wrap || wrap.dataset.rendered === 'true') continue;
            const priority = Math.abs(page - centerPage);
            const existing = pdfRenderQueue.find((item) => item.page === page);
            if (existing) {
                if (priority < existing.priority) existing.priority = priority;
            } else {
                pdfRenderQueue.push({ page, priority, renderToken });
            }
        }
    }

    pdfRenderQueue.sort((a, b) => a.priority - b.priority);
    if (!pdfRenderInFlight) processPdfRenderQueue();
}

async function processPdfRenderQueue() {
    if (pdfRenderInFlight) return;
    pdfRenderInFlight = true;
    let count = 0;
    while (pdfRenderQueue.length > 0) {
        const item = pdfRenderQueue.shift();
        if (item.renderToken !== sourceRenderToken) continue;
        if (pdfRenderedPages.has(item.page)) continue;
        const wrap = pdfPageWrappers[item.page];
        if (!wrap || wrap.dataset.rendered === 'true') continue;
        await renderSinglePdfPage(item.page, wrap, item.renderToken);
        count++;
        if (count % 3 === 0) recycleDistantPages(item.page);
    }
    pdfRenderInFlight = false;
}

async function renderSinglePdfPage(pageNumber, wrap, renderToken) {
    if (renderToken !== sourceRenderToken || pdfRenderedPages.has(pageNumber)) return;
    pdfRenderedPages.add(pageNumber);
    try {
        const page = await currentPdf.getPage(pageNumber);
        const outputScale = window.devicePixelRatio || 1;
        const viewport = page.getViewport({ scale: currentZoom * outputScale });
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        canvas.style.width = `${viewport.width / outputScale}px`;
        canvas.style.height = `${viewport.height / outputScale}px`;
        const canvasBox = wrap.querySelector('.pdf-canvas-box');
        if (!canvasBox) { pdfRenderedPages.delete(pageNumber); return; }
        const placeholderText = canvasBox.querySelector('.pdf-page-placeholder-text');
        if (placeholderText) placeholderText.remove();
        canvasBox.insertBefore(canvas, canvasBox.firstChild);
        wrap.style.minHeight = '';
        wrap.classList.remove('pdf-page-placeholder');
        wrap.dataset.rendered = 'true';
        await page.render({ canvasContext: context, viewport }).promise;
    } catch (err) {
        console.warn(`Failed to render PDF page ${pageNumber}`, err);
        pdfRenderedPages.delete(pageNumber);
    }
}

function recycleDistantPages(centerPage) {
    for (let i = 1; i < pdfPageWrappers.length; i++) {
        const wrap = pdfPageWrappers[i];
        if (!wrap || wrap.dataset.rendered !== 'true') continue;
        if (Math.abs(i - centerPage) > RECYCLE_THRESHOLD) {
            const canvas = wrap.querySelector('canvas');
            if (canvas) { canvas.width = 0; canvas.height = 0; canvas.remove(); }
            wrap.dataset.rendered = 'false';
            pdfRenderedPages.delete(i);
            const canvasBox = wrap.querySelector('.pdf-canvas-box');
            if (canvasBox && !canvasBox.querySelector('.pdf-page-placeholder-text')) {
                const placeholderText = document.createElement('div');
                placeholderText.className = 'pdf-page-placeholder-text';
                placeholderText.textContent = `${i}`;
                canvasBox.insertBefore(placeholderText, canvasBox.firstChild);
            }
            wrap.style.minHeight = `${842 * currentZoom}px`;
            wrap.classList.add('pdf-page-placeholder');
        }
    }
}

function setActiveResultView(view) {
    if (!view || view === activeResultView) return;
    activeResultView = view;
    document.querySelectorAll('.view-tab').forEach((tab) => {
        tab.classList.toggle('active', tab.dataset.view === view);
    });
    renderResultPane(getActiveTask(), { deferJson: true });
    updateActionState(getActiveTask());
}

function resultDataKey(task) {
    if (!task) return '';
    return [
        task.id,
        task.status,
        task.updatedAt || 0,
        task.markdown?.length || 0,
        task.ocrResults?.length || 0
    ].join(':');
}

function markdownRenderKey(task) {
    return `${resultDataKey(task)}:${sourceRenderToken}:${currentZoom}:${currentLanguage}`;
}

function resetResultRenderCache(taskId = null) {
    renderedResultTaskId = taskId;
    renderedMarkdownKey = '';
    renderedOfficialLayoutContext = '';
    renderedPPOCRVisualContext = '';
    renderedJsonKey = '';
    cachedJsonLines = [];
    cachedJsonMaxLineLength = 0;
    jsonRenderToken += 1;
}

function invalidatePPOCRVisualRender() {
    renderedMarkdownKey = '';
    renderedPPOCRVisualContext = '';
}

function resetResultScrollPositions() {
    els.markdownView.scrollTop = 0;
    els.markdownView.scrollLeft = 0;
    els.jsonView.scrollTop = 0;
    els.jsonView.scrollLeft = 0;
}

function captureResultScrollState() {
    const element = activeResultView === 'json' ? els.jsonView : els.markdownView;
    const maxScrollTop = Math.max(0, element.scrollHeight - element.clientHeight);
    const bottomOffset = maxScrollTop - element.scrollTop;
    return {
        element,
        scrollTop: element.scrollTop,
        scrollLeft: element.scrollLeft,
        stickToBottom: bottomOffset <= 32
    };
}

function restoreResultScrollState(state) {
    if (!state?.element) return;

    const restore = () => {
        const maxScrollTop = Math.max(0, state.element.scrollHeight - state.element.clientHeight);
        state.element.scrollTop = state.stickToBottom
            ? maxScrollTop
            : Math.min(state.scrollTop, maxScrollTop);
        const maxScrollLeft = Math.max(0, state.element.scrollWidth - state.element.clientWidth);
        state.element.scrollLeft = Math.min(state.scrollLeft || 0, maxScrollLeft);
    };

    restore();
    requestAnimationFrame(restore);
    setTimeout(restore, 80);
}

function showResultView(view) {
    const showJson = view === 'json';
    els.markdownView.classList.toggle('hidden', showJson);
    els.jsonView.classList.toggle('hidden', !showJson);
    // Free up the source PDF memory when the user is clearly more
    // interested in the parsed JSON. They can re-open the source via
    // the task list whenever they want — PDF.js will re-stream it.
    if (showJson && currentPdf) {
        try { currentPdf.destroy(); } catch (_) {}
        currentPdf = null;
        if (els.sourceViewer && !els.sourceViewer.querySelector('.source-preview-placeholder')) {
            // Re-render the preview placeholder so the workbench
            // doesn't show a blank pane for the source viewer.
            const task = getActiveTask();
            if (task) showSourcePreviewPlaceholder(task);
        }
    }
}

function renderResultPane(task, { deferJson = false, preserveScroll = true } = {}) {
    if (!task) {
        resetResultRenderCache();
        resetResultScrollPositions();
        updateResultViewLabels(null);
        syncResultMode(null);
        showResultView('markdown');
        els.resultTitle.textContent = t('解析结果');
        els.markdownView.innerHTML = `<div class="empty-result">${escapeHtml(t('选择左侧任务，或上传一个新文件开始解析。'))}</div>`;
        els.jsonView.textContent = '';
        return;
    }

    const isSameRenderedTask = renderedResultTaskId === task.id;
    const scrollState = preserveScroll && isSameRenderedTask
        ? captureResultScrollState()
        : null;

    if (!isSameRenderedTask) {
        resetResultRenderCache(task.id);
        resetResultScrollPositions();
    }

    els.resultTitle.textContent = resultPaneTitle(task);
    updateResultViewLabels(task);
    syncResultMode(task);

    if (activeResultView === 'json') {
        showResultView('json');
        renderJsonResult(task, { defer: deferJson, scrollState });
        return;
    }

    showResultView('markdown');
    const markdownKey = markdownRenderKey(task);
    const ppocrVisualTask = isPPOCRVisualTask(task);
    const ppocrContext = ppocrVisualTask ? ppocrVisualRenderContext(task) : '';
    if (renderedMarkdownKey === markdownKey && (!ppocrVisualTask || renderedPPOCRVisualContext === ppocrContext)) {
        warmJsonResultCache(task);
        restoreResultScrollState(scrollState);
        return;
    }

    if (ppocrVisualTask) {
        renderedOfficialLayoutContext = '';
        renderPPOCRVisualResult(task, markdownKey, scrollState);
        warmJsonResultCache(task);
        return;
    }

    const isDocumentParser = ['mineru', 'glm-ocr'].includes(task.modelId) || (task.ocrResults?.length && task.ocrResults.some(p => ['mineru', 'glm-ocr'].includes(p.parser)));
    
    if (!isDocumentParser) {
        const officialRender = renderOfficialLayoutResult(task);
        if (officialRender.rendered) {
            renderedMarkdownKey = markdownKey;
            if (officialRender.changed) {
                officialRender.mathRoots.forEach((root) => renderMathWhenReady(root));
            }
            warmJsonResultCache(task);
            restoreResultScrollState(scrollState);
            return;
        }
    }

    const markdown = prepareMarkdownForRender(task.markdown || '');
    if (!markdown) {
        renderedOfficialLayoutContext = '';
        clearSourceHighlight();
        clearSourceHotspots();
        els.markdownView.innerHTML = `<div class="empty-result">${escapeHtml(emptyResultText(task))}</div>`;
        renderedMarkdownKey = markdownKey;
        warmJsonResultCache(task);
        restoreResultScrollState(scrollState);
        return;
    }

    let renderMarkdown = markdown;
    renderedOfficialLayoutContext = '';
    // Image source replacement — handle both Markdown ![](path) and HTML <img src="path">
    const images = task.images || {};
    const imageKeys = Object.keys(images);
    if (imageKeys.length > 0) {
        // Replace Markdown-style image refs: ![alt](path)
        renderMarkdown = renderMarkdown.split(/(!\[[^\]]*\]\()([^)\s]+)/g).map((part, i) => {
            if (i % 3 === 2 && images[part]) {
                return `data:image/jpeg;base64,${images[part]}`;
            }
            return part;
        }).join('');
        // Replace HTML-style image refs: <img src="path"> or <img src='path'>
        for (const [path, base64] of Object.entries(images)) {
            const dataUrl = `data:image/jpeg;base64,${base64}`;
            // Double quotes
            renderMarkdown = renderMarkdown.split(`src="${path}"`).join(`src="${dataUrl}"`);
            // Single quotes
            renderMarkdown = renderMarkdown.split(`src='${path}'`).join(`src='${dataUrl}'`);
        }
    }
    const html = renderMarkdownHtml(renderMarkdown);
    // Skip DOM update if rendered HTML hasn't changed (cheap string reference check)
    if (html === lastRenderedHtml) {
        warmJsonResultCache(task);
        restoreResultScrollState(scrollState);
        return;
    }
    lastRenderedHtml = html;
    els.markdownView.innerHTML = html;
    renderedMarkdownKey = markdownKey;
    renderMathWhenReady(els.markdownView);
    linkMarkdownToSourceBlocks(task);
    if (task.modelId === 'mineru' && Array.isArray(task.contentList)) {
        bindMineruContentClicks(task);
    }
    warmJsonResultCache(task);
    restoreResultScrollState(scrollState);
}

async function renderJsonResult(task, { defer = false, scrollState = null } = {}) {
    // Lazy-load ocrResults if not yet loaded
    if (task._ocrLoaded === false && task._ocrTotal !== 0) {
        try {
            const resultData = await loadTaskResult(task.id, {
                fields: 'ocrResults',
                ocrLimit: 1000,
            });
            if (Array.isArray(resultData.ocrResults)) {
                task.ocrResults = resultData.ocrResults;
            }
            task._ocrLoaded = true;
            task._ocrTotal = resultData.ocrTotal || task.ocrResults.length;
        } catch (err) {
            console.warn('Failed to lazy-load ocrResults:', err);
            task._ocrLoaded = true; // Don't retry on error
        }
    }

    const key = resultDataKey(task);
    if (renderedJsonKey === key) {
        renderVisibleJsonLines();
        restoreResultScrollState(scrollState);
        return;
    }

    const render = () => {
        cacheJsonLines(JSON.stringify(toOfficialJson(task), null, 2));
        renderedJsonKey = key;
        if (!scrollState) {
            els.jsonView.scrollTop = 0;
        }
        renderVisibleJsonLines();
        restoreResultScrollState(scrollState);
    };

    if (!defer) {
        render();
        return;
    }

    const token = ++jsonRenderToken;
    renderVisibleJsonLines();
    requestAnimationFrame(() => {
        if (token !== jsonRenderToken || activeResultView !== 'json' || getActiveTask()?.id !== task.id) return;
        render();
    });
}

function warmJsonResultCache(task) {
    const key = resultDataKey(task);
    // Skip if ocrResults not yet loaded (will be loaded on demand in renderJsonResult)
    if (renderedJsonKey === key || !task?.ocrResults?.length || task._ocrLoaded === false) return;

    const warm = () => {
        if (renderedJsonKey === key || activeResultView === 'json' || getActiveTask()?.id !== task.id) return;
        cacheJsonLines(JSON.stringify(toOfficialJson(task), null, 2));
        renderedJsonKey = key;
    };

    if (window.requestIdleCallback) {
        requestIdleCallback(warm, { timeout: 1200 });
    } else {
        setTimeout(warm, 80);
    }
}

function cacheJsonLines(text) {
    cachedJsonLines = String(text || '').split('\n');
    cachedJsonMaxLineLength = cachedJsonLines.reduce((max, line) => Math.max(max, line.length), 0);
}

function updateResultViewLabels(task) {
    const markdownTab = document.querySelector('.view-tab[data-view="markdown"]');
    if (!markdownTab) return;
    markdownTab.textContent = isPPOCRVisualTask(task) ? t('文字识别') : t('文档解析');
}

function syncResultMode(task) {
    const visualMode = isPPOCRVisualTask(task) && activeResultView === 'markdown';
    els.resultPane?.classList.toggle('ppocr-result-mode', visualMode);
    els.markdownView.classList.toggle('ocr-visual-mode', visualMode);
}

function isPPOCRVisualTask(task) {
    return task?.modelId === 'pp-ocrv6'
        || Boolean(task?.ocrResults?.some((pageResult) => pageResult?.parser === 'pp-ocrv6'));
}

function renderPPOCRVisualResult(task, markdownKey, scrollState = null) {
    const pages = collectPPOCRVisualPages(task);
    const context = ppocrVisualRenderContext(task);
    const visualScrollState = freezeVisualScrollState(scrollState);

    if (!pages.length) {
        const hasEmptyResult = els.markdownView.children.length === 1
            && els.markdownView.firstElementChild?.classList.contains('empty-result')
            && renderedPPOCRVisualContext === context;
        if (!hasEmptyResult) {
            clearSourceHighlight();
            clearSourceHotspots();
            els.markdownView.innerHTML = `<div class="empty-result">${escapeHtml(emptyResultText(task))}</div>`;
        }
        renderedPPOCRVisualContext = context;
        renderedMarkdownKey = markdownKey;
        restoreResultScrollState(visualScrollState);
        return;
    }

    const expectedKeys = pages.map(ppocrVisualPageKey);
    let flow = els.markdownView.querySelector(':scope > .ocr-visual-flow');
    const existingPages = flow
        ? Array.from(flow.children).filter((element) => element.classList.contains('ocr-visual-page'))
        : [];
    const existingKeys = existingPages.map((element) => element.dataset.pageKey || '');
    const canAppend = Boolean(flow)
        && els.markdownView.children.length === 1
        && renderedPPOCRVisualContext === context
        && existingKeys.length <= expectedKeys.length
        && existingKeys.every((key, index) => key === expectedKeys[index]);

    if (canAppend && existingKeys.length === expectedKeys.length) {
        renderedMarkdownKey = markdownKey;
        restoreResultScrollState(visualScrollState);
        return;
    }

    if (!canAppend) {
        clearSourceHighlight();
        clearSourceHotspots();
        flow = document.createElement('div');
        flow.className = 'ocr-visual-flow';
        els.markdownView.replaceChildren(flow);
        renderedPPOCRVisualContext = context;
    }

    const startIndex = canAppend ? existingKeys.length : 0;
    pages.slice(startIndex).forEach((page, offset) => {
        const pageIndex = startIndex + offset;
        flow.appendChild(createPPOCRVisualPage(page, pageIndex, expectedKeys[pageIndex]));
    });
    renderedMarkdownKey = markdownKey;
    restoreResultScrollState(visualScrollState);
}

function freezeVisualScrollState(scrollState) {
    if (!scrollState) return null;
    return {
        ...scrollState,
        stickToBottom: false
    };
}

function ppocrVisualRenderContext(task) {
    return [
        task?.id || '',
        sourceRenderToken,
        currentZoom,
        currentLanguage
    ].join(':');
}

function ppocrVisualPageKey(page) {
    const firstLine = page.lines[0] || {};
    const lastLine = page.lines[page.lines.length - 1] || {};
    const signature = [
        firstLine.text || '',
        Array.isArray(firstLine.box) ? firstLine.box.join(',') : '',
        lastLine.text || '',
        Array.isArray(lastLine.box) ? lastLine.box.join(',') : ''
    ].join('|');
    return [
        page.pageNumber || '',
        page.index,
        page.pageImage ? String(page.pageImage).length : 0,
        page.lines.length,
        hashString(signature)
    ].join(':');
}

function hashString(value) {
    let hash = 0;
    const text = String(value || '');
    for (let index = 0; index < text.length; index += 1) {
        hash = ((hash << 5) - hash + text.charCodeAt(index)) >>> 0;
    }
    return hash.toString(36);
}

function collectPPOCRVisualPages(task) {
    if (!Array.isArray(task?.ocrResults)) return [];
    return task.ocrResults
        .map((pageResult, index) => {
            const lines = collectPPOCRLines(pageResult);
            const pageImage = pageResult?.pageImage || pageResult?.inputImage || null;
            return {
                index,
                pageNumber: Number(pageResult?.sourcePage || pageResult?.page_index || index + 1),
                pageImage,
                lines
            };
        })
        .filter((page) => page.pageImage || page.lines.length > 0)
        .sort((a, b) => (a.pageNumber - b.pageNumber) || (a.index - b.index));
}

function collectPPOCRLines(pageResult) {
    if (Array.isArray(pageResult?.ocrLines)) {
        return pageResult.ocrLines
            .map((line, index) => normalizePPOCRLine(line, index))
            .filter(Boolean);
    }
    if (Array.isArray(pageResult?.lines)) {
        return pageResult.lines
            .map((line, index) => normalizePPOCRLine(line, index))
            .filter(Boolean);
    }

    const pruned = pageResult?.prunedResult || pageResult || {};
    const texts = Array.isArray(pruned.rec_texts) ? pruned.rec_texts : [];
    const scores = Array.isArray(pruned.rec_scores) ? pruned.rec_scores : [];
    const boxes = Array.isArray(pruned.rec_boxes) ? pruned.rec_boxes : [];
    const polys = Array.isArray(pruned.rec_polys) ? pruned.rec_polys : [];
    return texts.map((text, index) => normalizePPOCRLine({
        text,
        score: scores[index],
        box: boxes[index],
        poly: polys[index]
    }, index)).filter(Boolean);
}

function normalizePPOCRLine(line, index) {
    const text = String(line?.text || '').trim();
    if (!text) return null;
    const box = normalizePPOCRBox(line.box || boxFromPoly(line.poly));
    if (!box) return null;
    return {
        index,
        text,
        score: line.score,
        box
    };
}

function boxFromPoly(poly) {
    if (!Array.isArray(poly) || poly.length === 0) return null;
    const xs = [];
    const ys = [];
    poly.forEach((point) => {
        if (!Array.isArray(point) || point.length < 2) return;
        xs.push(Number(point[0]));
        ys.push(Number(point[1]));
    });
    if (!xs.length || !ys.length) return null;
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function normalizePPOCRBox(box) {
    if (!Array.isArray(box) || box.length < 4) return null;
    const values = box.slice(0, 4).map(Number);
    if (values.some((value) => !Number.isFinite(value))) return null;
    const [x1, y1, x2, y2] = values;
    if (x2 <= x1 || y2 <= y1) return null;
    return values;
}

function createPPOCRVisualPage(page, pageIndex, pageKey = '') {
    const pageElement = document.createElement('section');
    pageElement.className = 'ocr-visual-page';
    pageElement.dataset.page = String(page.pageNumber || pageIndex + 1);
    if (pageKey) pageElement.dataset.pageKey = pageKey;

    const stage = document.createElement('div');
    stage.className = 'ocr-page-stage loading';
    const displaySize = getPPOCRPageDisplaySize(page);
    if (displaySize) {
        applyPPOCRStageDisplaySize(stage, displaySize.width, displaySize.height);
    }
    pageElement.appendChild(stage);

    const toolbar = createPPOCRFloatingToolbar();
    stage.appendChild(toolbar);

    if (page.pageImage) {
        const img = document.createElement('img');
        img.className = 'ocr-page-image';
        img.alt = `OCR page ${page.pageNumber || pageIndex + 1}`;
        img.addEventListener('load', () => {
            stage.classList.remove('loading');
            const coordinateWidth = img.naturalWidth || displaySize?.width || 1;
            const coordinateHeight = img.naturalHeight || displaySize?.height || 1;
            layoutPPOCRTextLayer(stage, page, coordinateWidth, coordinateHeight, toolbar, img);
            syncPPOCRVisualScrollFromSource();
        }, { once: true });
        img.addEventListener('error', () => {
            stage.classList.remove('loading');
            createPPOCRTextOnlyLayer(stage, page.lines, toolbar);
        }, { once: true });
        if (displaySize) {
            applyPPOCRStageDisplaySize(stage, displaySize.width, displaySize.height, img);
        }
        stage.appendChild(img);
        img.src = imageValueToSrc(page.pageImage);
    } else {
        stage.classList.remove('loading');
        createPPOCRTextOnlyLayer(stage, page.lines, toolbar);
    }

    return pageElement;
}

function getPPOCRPageDisplaySize(page) {
    const sourceSize = getSourcePageDisplaySize(page.pageNumber);
    if (sourceSize?.width && sourceSize?.height) return sourceSize;
    return null;
}

function applyPPOCRStageDisplaySize(stage, width, height, imageElement = null) {
    if (!stage || !width || !height) return;
    const roundedWidth = Math.round(width);
    const roundedHeight = Math.round(height);
    stage.classList.add('sized');
    stage.style.width = `${roundedWidth}px`;
    stage.style.height = `${roundedHeight}px`;
    if (imageElement) {
        imageElement.style.width = `${roundedWidth}px`;
        imageElement.style.height = `${roundedHeight}px`;
    }
}

function createPPOCRFloatingToolbar() {
    const toolbar = document.createElement('div');
    toolbar.className = 'ocr-floating-toolbar hidden';
    toolbar.innerHTML = `
        <button type="button" data-action="copy">
            <svg viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            <span data-label>${escapeHtml(t('复制'))}</span>
        </button>
        <button type="button" data-action="correct">
            <svg viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
            <span data-label>${escapeHtml(t('纠正'))}</span>
        </button>
    `;
    const handleToolbarAction = async (event) => {
        const button = event.target.closest('button');
        if (!button) return;
        event.stopPropagation();
        if (event.type === 'pointerdown') {
            event.preventDefault();
            toolbar._lastPointerAction = {
                action: button.dataset.action,
                time: Date.now()
            };
        } else if (
            toolbar._lastPointerAction?.action === button.dataset.action
            && Date.now() - toolbar._lastPointerAction.time < 500
        ) {
            return;
        }
        if (button.dataset.action === 'copy') {
            await copyPPOCRToolbarText(toolbar, button);
        }
        if (button.dataset.action === 'correct') {
            openPPOCRCorrectionEditor(toolbar);
        }
    };
    toolbar.querySelectorAll('button').forEach((button) => {
        button.addEventListener('pointerdown', handleToolbarAction);
        button.addEventListener('click', handleToolbarAction);
    });
    return toolbar;
}

async function copyPPOCRToolbarText(toolbar, button) {
    const text = toolbar.dataset.text || '';
    if (!text) return;
    try {
        await writeClipboardText(text);
        flashToolbarButtonLabel(button, t('已复制'), t('复制'));
    } catch (error) {
        console.error(error);
        flashToolbarButtonLabel(button, t('复制失败'), t('复制'));
    }
}

async function writeClipboardText(text) {
    if (navigator.clipboard?.writeText && window.isSecureContext) {
        try {
            await navigator.clipboard.writeText(text);
            return;
        } catch (error) {
            console.warn('Clipboard API write failed, falling back to textarea copy.', error);
        }
    }
    // iOS Safari refuses to copy from an off-screen / display:none element
    // (it must be selectable, visible, and editable), so we render the
    // textarea in-flow but visually inert before issuing execCommand.
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.contentEditable = 'true';
    textarea.style.position = 'fixed';
    textarea.style.top = '0';
    textarea.style.left = '0';
    textarea.style.width = '1px';
    textarea.style.height = '1px';
    textarea.style.padding = '0';
    textarea.style.border = '0';
    textarea.style.opacity = '0';
    textarea.style.pointerEvents = 'none';
    document.body.appendChild(textarea);
    // iOS needs an explicit selection range, not just .select().
    const range = document.createRange();
    range.selectNodeContents(textarea);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    textarea.setSelectionRange(0, text.length);
    const copied = document.execCommand('copy');
    selection.removeAllRanges();
    textarea.remove();
    if (!copied) {
        throw new Error('copy command failed');
    }
}

function flashToolbarButtonLabel(button, text, restoreText) {
    const label = button.querySelector('[data-label]');
    if (!label) return;
    label.textContent = text;
    window.setTimeout(() => {
        label.textContent = restoreText;
    }, 900);
}

function openPPOCRCorrectionEditor(toolbar) {
    const stage = toolbar.closest('.ocr-page-stage');
    const active = toolbar._activePPOCR;
    if (!stage || !active?.element || !active?.line) return;
    stage.querySelector('.ocr-correction-popover')?.remove();

    const popover = document.createElement('form');
    popover.className = 'ocr-correction-popover';
    popover.innerHTML = `
        <input type="text" name="text" aria-label="${escapeHtml(t('纠正文字'))}">
        <button type="submit">${escapeHtml(t('保存'))}</button>
        <button type="button" data-action="cancel">${escapeHtml(t('取消'))}</button>
    `;
    const input = popover.querySelector('input');
    input.value = active.line.text || '';
    popover.addEventListener('submit', async (event) => {
        event.preventDefault();
        const nextText = input.value.trim();
        if (!nextText) return;
        await applyPPOCRCorrection(active.element, active.line, nextText, toolbar);
        popover.remove();
    });
    popover.querySelector('[data-action="cancel"]').addEventListener('click', () => popover.remove());
    stage.appendChild(popover);

    const toolbarRect = toolbar.getBoundingClientRect();
    const stageRect = stage.getBoundingClientRect();
    const viewerRect = els.markdownView.getBoundingClientRect();
    const minLeft = viewerRect.left - stageRect.left + 8;
    const maxLeft = viewerRect.right - stageRect.left - 286;
    const minTop = viewerRect.top - stageRect.top + 8;
    const maxTop = viewerRect.bottom - stageRect.top - 48;
    const left = Math.min(
        Math.max(toolbarRect.left - stageRect.left, minLeft),
        Math.max(maxLeft, minLeft)
    );
    const top = Math.min(
        Math.max(toolbarRect.bottom - stageRect.top + 8, minTop),
        Math.max(maxTop, minTop)
    );
    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
    input.focus();
    input.select();
}

async function applyPPOCRCorrection(element, line, nextText, toolbar) {
    const previousText = line.text || '';
    if (nextText === previousText) return;

    line.text = nextText;
    updatePPOCRLineElementText(element, nextText);
    element.classList.toggle('ocr-text-line-code', isPPOCRCodeToken(nextText));
    toolbar.dataset.text = nextText;
    updateStoredPPOCRLineText(line, nextText);
    fitPPOCRLineElement(element, line);
    await saveCorrectedPPOCRTask();
}

function updatePPOCRLineElementText(element, text) {
    const label = element.querySelector('.ocr-text-line-label');
    if (label) label.textContent = text;
    element.title = text;
    element.setAttribute('aria-label', text);
}

function updateStoredPPOCRLineText(line, text) {
    const task = getActiveTask();
    const pageResult = task?.ocrResults?.[line.pageResultIndex];
    if (!pageResult) return;

    if (Array.isArray(pageResult.ocrLines) && pageResult.ocrLines[line.index]) {
        pageResult.ocrLines[line.index].text = text;
    }
    const pruned = pageResult.prunedResult || pageResult;
    if (Array.isArray(pruned?.rec_texts) && pruned.rec_texts[line.index] !== undefined) {
        pruned.rec_texts[line.index] = text;
    }
    task.updatedAt = Date.now();
}

async function saveCorrectedPPOCRTask() {
    const task = getActiveTask();
    if (!task) return;
    renderedJsonKey = '';
    warmJsonResultCache(task);
    try {
        await saveTask(task);
    } catch (error) {
        console.error(error);
        alert(error.message || '\u4fdd\u5b58\u7ea0\u6b63\u5931\u8d25');
    }
}

function layoutPPOCRTextLayer(stage, page, width, height, toolbar, imageElement = null) {
    stage.querySelector('.ocr-text-layer')?.remove();
    const layer = document.createElement('div');
    layer.className = 'ocr-text-layer';
    stage.appendChild(layer);

    const lines = page.lines || [];
    let renderWidth = width || 1;
    let renderHeight = height || 1;
    const sourceSize = getSourcePageDisplaySize(page.pageNumber);
    if (sourceSize) {
        renderWidth = sourceSize.width;
        renderHeight = sourceSize.height;
    }
    if (imageElement) {
        renderWidth = renderWidth || imageElement.clientWidth || imageElement.naturalWidth || width || 1;
        renderHeight = renderHeight || imageElement.clientHeight || imageElement.naturalHeight || height || 1;
        stage.style.width = `${renderWidth}px`;
        stage.style.height = `${renderHeight}px`;
        layer.style.width = `${renderWidth}px`;
        layer.style.height = `${renderHeight}px`;
        imageElement.style.width = `${renderWidth}px`;
        imageElement.style.height = `${renderHeight}px`;
    }

    const bounds = inferPPOCRCoordinateBounds(lines, width, height);
    lines.forEach((line) => {
        hydratePPOCRLineGeometry(line, page, bounds);
        const element = document.createElement('button');
        element.type = 'button';
        element.className = 'ocr-text-line';
        element.appendChild(createPPOCRLineLabel(line.text));
        element.title = line.text;
        element.setAttribute('aria-label', line.text);
        element.dataset.page = String(line.sourcePage || page.pageNumber || '');
        element.dataset.pageResultIndex = String(line.pageResultIndex ?? '');
        element.dataset.lineIndex = String(line.index ?? '');
        positionPPOCRLine(element, line, bounds, renderWidth, renderHeight);
        bindPPOCRLineEvents(element, toolbar, line);
        layer.appendChild(element);
        addPPOCRSourceHotspot(line, element, toolbar);
        fitPPOCRLineElement(element, line);
    });
}

function getSourcePageDisplaySize(pageNumber) {
    const pageWrap = els.sourceViewer.querySelector(`.pdf-page-wrap[data-page="${pageNumber}"]`);
    const canvas = pageWrap?.querySelector('canvas');
    if (!canvas) return null;
    return {
        width: canvas.clientWidth || canvas.width,
        height: canvas.clientHeight || canvas.height
    };
}

function hydratePPOCRLineGeometry(line, page, bounds) {
    line.sourcePage = Number(page.pageNumber || line.sourcePage || 1);
    line.pageResultIndex = page.index;
    line.pageWidth = bounds.width;
    line.pageHeight = bounds.height;
}

function createPPOCRLineLabel(text) {
    const label = document.createElement('span');
    label.className = 'ocr-text-line-label';
    label.textContent = text;
    return label;
}

function createPPOCRTextOnlyLayer(stage, lines, toolbar) {
    const fallback = document.createElement('div');
    fallback.className = 'ocr-text-only';
    lines.forEach((line) => {
        const element = document.createElement('button');
        element.type = 'button';
        element.className = 'ocr-text-only-line';
        element.textContent = line.text;
        bindPPOCRLineEvents(element, toolbar, line);
        fallback.appendChild(element);
    });
    stage.appendChild(fallback);
}

function inferPPOCRCoordinateBounds(lines, width, height) {
    const maxX = Math.max(width, ...lines.map((line) => line.box[2]));
    const maxY = Math.max(height, ...lines.map((line) => line.box[3]));
    return {
        width: maxX || width || 1,
        height: maxY || height || 1
    };
}

function positionPPOCRLine(element, line, bounds, renderWidth, renderHeight) {
    const box = line.box;
    const [x1, y1, x2, y2] = box;
    const left = (x1 / bounds.width) * 100;
    const top = (y1 / bounds.height) * 100;
    const width = ((x2 - x1) / bounds.width) * 100;
    const height = ((y2 - y1) / bounds.height) * 100;
    const boxWidth = Math.max(1, (width / 100) * renderWidth);
    const boxHeight = Math.max(1, (height / 100) * renderHeight);
    const isCodeToken = isPPOCRCodeToken(line.text);
    const fontSize = fittedPPOCRFontSize(line.text, boxWidth, boxHeight);

    element.style.left = `${left}%`;
    element.style.top = `${top}%`;
    element.style.width = `${width}%`;
    element.style.height = `${height}%`;
    element.style.fontSize = `${fontSize}px`;
    if (isCodeToken) {
        element.classList.add('ocr-text-line-code');
    }
    if (!isCodeToken && (boxHeight < 6 || boxWidth < 6)) {
        element.classList.add('ocr-text-line-compact');
    }
}

function fittedPPOCRFontSize(text, boxWidth, boxHeight) {
    const availableHeight = Math.max(1, boxHeight - 2);
    const byHeight = availableHeight * 0.92;
    if (isPPOCRCodeToken(text)) {
        return Math.round(Math.max(4.2, Math.min(12, byHeight)) * 10) / 10;
    }
    const minReadable = boxWidth >= 120 ? 9.4 : 6;
    const byNarrowWidth = boxWidth < 18 ? Math.max(5, boxWidth * 0.72) : 14;
    return Math.round(Math.max(minReadable, Math.min(14, byHeight, byNarrowWidth)) * 10) / 10;
}

function isPPOCRCodeToken(text) {
    const value = String(text || '').trim();
    return /^[A-Za-z]{2,8}\d{2,8}[A-Za-z0-9-]*$/.test(value);
}

function fitPPOCRLineElement(element, line) {
    const label = element.querySelector('.ocr-text-line-label');
    if (!label) return;

    label.style.transform = 'none';
    const isCodeToken = isPPOCRCodeToken(line?.text);
    const isWideTextLine = element.clientWidth >= 120;
    const minScale = isCodeToken || isWideTextLine ? 0.48 : 0.62;
    const minHeightScale = isWideTextLine && !isCodeToken ? 0.82 : 1;
    const minFontSize = isCodeToken ? 3.8 : 4.8;

    for (let attempt = 0; attempt < 4; attempt += 1) {
        const availableWidth = Math.max(1, element.clientWidth - 1);
        const availableHeight = Math.max(1, element.clientHeight - 1);
        const naturalWidth = Math.max(1, label.scrollWidth || label.getBoundingClientRect().width);
        const naturalHeight = Math.max(1, label.scrollHeight || label.getBoundingClientRect().height);
        const widthScale = Math.min(1, availableWidth / naturalWidth);
        const heightScale = Math.min(1, availableHeight / naturalHeight);

        if (widthScale >= minScale && heightScale >= minHeightScale) {
            label.style.transform = widthScale < 1 ? `scaleX(${roundPPOCRScale(widthScale)})` : 'none';
            return;
        }

        const currentSize = Number.parseFloat(element.style.fontSize || getComputedStyle(element).fontSize) || 6;
        const targetWidthRatio = widthScale < minScale ? widthScale / minScale : 1;
        const targetHeightRatio = heightScale < minHeightScale ? heightScale / minHeightScale : 1;
        const ratio = Math.min(targetHeightRatio, targetWidthRatio, 1);
        const nextSize = Math.round(Math.max(minFontSize, currentSize * ratio * 0.98) * 10) / 10;
        if (nextSize >= currentSize - 0.05) break;
        element.style.fontSize = `${nextSize}px`;
    }

    const finalAvailableWidth = Math.max(1, element.clientWidth - 1);
    const finalNaturalWidth = Math.max(1, label.scrollWidth || label.getBoundingClientRect().width);
    const finalScale = Math.min(1, finalAvailableWidth / finalNaturalWidth);
    label.style.transform = finalScale < 1 ? `scaleX(${roundPPOCRScale(finalScale)})` : 'none';
}

function roundPPOCRScale(value) {
    return Math.round(Math.max(0.35, Math.min(1, value)) * 1000) / 1000;
}

function bindPPOCRLineEvents(element, toolbar, line) {
    const activate = () => activatePPOCRLine(element, toolbar, line);
    element.addEventListener('mouseenter', activate);
    element.addEventListener('focus', activate);
    element.addEventListener('click', activate);
}

function activatePPOCRLine(element, toolbar, line, { scrollSource = false } = {}) {
    const stage = element.closest('.ocr-page-stage');
    if (!stage) return;
    stage.querySelectorAll('.ocr-text-line.active, .ocr-text-only-line.active').forEach((item) => {
        item.classList.remove('active');
    });
    element.classList.add('active');
    toolbar.dataset.text = line.text;
    toolbar._activePPOCR = { element, line };
    toolbar.classList.remove('hidden');
    showPPOCRSourceHighlight(line);
    if (scrollSource) {
        const page = els.sourceViewer.querySelector(`.pdf-page-wrap[data-page="${line.sourcePage}"]`);
        if (page && !isElementMostlyVisible(page, els.sourceViewer)) {
            scrollPdfPageIntoView(line.sourcePage, 'smooth');
        }
    }

    const stageRect = stage.getBoundingClientRect();
    const elementRect = element.getBoundingClientRect();
    const viewerRect = els.markdownView.getBoundingClientRect();
    const minLeft = viewerRect.left - stageRect.left + 8;
    const maxLeft = viewerRect.right - stageRect.left - 152;
    const minTop = viewerRect.top - stageRect.top + 8;
    const maxTop = viewerRect.bottom - stageRect.top - 44;
    const left = Math.min(
        Math.max(elementRect.left - stageRect.left + elementRect.width - 142, minLeft),
        Math.max(maxLeft, minLeft)
    );
    const top = Math.min(
        Math.max(elementRect.bottom - stageRect.top + 8, minTop),
        Math.max(maxTop, minTop)
    );
    toolbar.style.left = `${left}px`;
    toolbar.style.top = `${top}px`;
}

function schedulePPOCRSourceScrollSync() {
    if (splitScrollSyncLocked || ppocrScrollSyncFrame) return;
    ppocrScrollSyncFrame = requestAnimationFrame(() => {
        ppocrScrollSyncFrame = 0;
        syncSourceScrollFromPPOCRVisual();
    });
}

function handlePPOCRMarkdownScroll() {
    schedulePPOCRSourceScrollSync();
}

function syncSourceScrollFromPPOCRVisual() {
    const task = getActiveTask();
    if (!isPPOCRVisualTask(task) || activeResultView !== 'markdown') return;
    if (!currentPdf || !els.sourceViewer || !els.markdownView) return;
    syncPairedPPOCRScroll(els.markdownView, els.sourceViewer);
    updateCurrentPageFromScroll();
}

function syncPPOCRVisualScrollFromSource() {
    const task = getActiveTask();
    if (!isPPOCRVisualTask(task) || activeResultView !== 'markdown') return;
    if (!currentPdf || !els.sourceViewer || !els.markdownView) return;
    syncPairedPPOCRScroll(els.sourceViewer, els.markdownView);
}

function syncPairedPPOCRScroll(fromContainer, toContainer) {
    if (splitScrollSyncLocked || !fromContainer || !toContainer) return;
    const targetTop = directScrollTarget(toContainer, fromContainer.scrollTop || 0, 'top');
    const targetLeft = directScrollTarget(toContainer, fromContainer.scrollLeft || 0, 'left');
    if (
        Math.abs((toContainer.scrollTop || 0) - targetTop) < 1
        && Math.abs((toContainer.scrollLeft || 0) - targetLeft) < 1
    ) {
        return;
    }
    withSplitScrollLock(() => {
        toContainer.scrollTo({
            top: targetTop,
            left: targetLeft,
            behavior: 'auto'
        });
    });
}

function directScrollTarget(container, value, axis) {
    const maxScroll = axis === 'left'
        ? Math.max(0, container.scrollWidth - container.clientWidth)
        : Math.max(0, container.scrollHeight - container.clientHeight);
    return Math.min(Math.max(value, 0), maxScroll);
}

function ensureJsonVirtualDom() {
    let spacer = els.jsonView.querySelector('.json-virtual-spacer');
    let lines = els.jsonView.querySelector('.json-virtual-lines');
    if (spacer && lines) return { spacer, lines };

    els.jsonView.textContent = '';
    spacer = document.createElement('div');
    spacer.className = 'json-virtual-spacer';
    lines = document.createElement('code');
    lines.className = 'json-virtual-lines';
    els.jsonView.append(spacer, lines);
    return { spacer, lines };
}

function renderVisibleJsonLines() {
    if (!cachedJsonLines.length) {
        els.jsonView.textContent = '';
        return;
    }

    const { spacer, lines } = ensureJsonVirtualDom();
    const totalHeight = cachedJsonLines.length * JSON_LINE_HEIGHT + JSON_PADDING_TOP + JSON_PADDING_BOTTOM;
    spacer.style.height = `${totalHeight}px`;
    spacer.style.width = `calc(${Math.max(cachedJsonMaxLineLength, 1)}ch + ${JSON_PADDING_LEFT + JSON_PADDING_RIGHT}px)`;

    const viewportHeight = els.jsonView.clientHeight || 1;
    const firstVisibleLine = Math.max(0, Math.floor((els.jsonView.scrollTop - JSON_PADDING_TOP) / JSON_LINE_HEIGHT));
    const visibleLineCount = Math.ceil(viewportHeight / JSON_LINE_HEIGHT) + JSON_OVERSCAN_LINES * 2;
    const start = Math.max(0, firstVisibleLine - JSON_OVERSCAN_LINES);
    const end = Math.min(cachedJsonLines.length, start + visibleLineCount);

    lines.style.transform = `translateY(${JSON_PADDING_TOP + start * JSON_LINE_HEIGHT}px)`;
    lines.textContent = cachedJsonLines.slice(start, end).join('\n');
}

async function processActiveTask() {
    const task = getActiveTask();
    await processTask(task, { confirmCompleted: true });
}

async function processTask(task, { confirmCompleted = true } = {}) {
    if (!task) return;
    // If another task is already in flight, release the queue so it can
    // come back to us later (or, for non-batch callers, just no-op).
    if (isProcessing) {
        signalBatchJobDone(task.id, 'error');
        return;
    }
    if (confirmCompleted && task.status === 'completed' && !confirm(t('这个任务已经解析完成，要重新解析吗？'))) {
        signalBatchJobDone(task.id, 'skipped');
        return;
    }

    let resumeExistingResults = shouldResumeTask(task);
    let targetModel = resumeExistingResults ? getTaskModel(task) : getSelectedModel();

    if (task.modelId && task.modelId !== selectedModelId) {
        if (resumeExistingResults && task.batches.some(b => b.status === 'completed')) {
            if (confirm(t('你已切换到新的解析模型，是否要清空旧模型的解析结果并重新开始？\n\n点击“确定”使用新模型重新解析，点击“取消”继续使用旧模型解析剩余页面。'))) {
                resumeExistingResults = false;
                targetModel = getSelectedModel();
                applySelectedModelToTask(task);
            } else {
                els.modelSelect.value = task.modelId;
                selectedModelId = task.modelId;
                localStorage.setItem(MODEL_STORAGE_KEY, selectedModelId);
                updateActiveModelDisplay(task);
            }
        } else {
            resumeExistingResults = false;
            targetModel = getSelectedModel();
            applySelectedModelToTask(task);
        }
    } else if (confirmCompleted && task.status === 'completed') {
        resumeExistingResults = false;
        targetModel = getSelectedModel();
        applySelectedModelToTask(task);
    } else if (!task.modelId) {
        applySelectedModelToTask(task);
        targetModel = getSelectedModel();
    }

    const modelReady = await ensureModelRuntimeReadyForTask(task, targetModel);
    if (!modelReady) {
        signalBatchJobDone(task.id, 'error');
        return;
    }

    // Decide processing mode: server-side (background) for tasks with sourceUrl,
    // client-side for small tasks without server source
    const useServerProcessing = backgroundProcessingEnabled && task.sourceUrl && !task.sourceDataUrl;

    isProcessing = true;
    try {
        if (shouldRebuildPdfBatchPlan(task)) {
            rebuildPdfBatchPlan(task);
        }
        if (resumeExistingResults) {
            task.batches.forEach((batch) => {
                if (batch.status === 'processing') batch.status = 'pending';
            });
            rebuildTaskResultFromCompletedBatches(task);
        } else {
            if (confirmCompleted || !task.modelId) {
                applySelectedModelToTask(task);
            }
            task.markdown = '';
            task.images = {};
            task.ocrResults = [];
            task.contentList = [];
            task.batches.forEach((batch) => {
                batch.status = 'pending';
                batch.markdown = '';
            });
        }
        task.status = 'processing';
        task.error = null;
        task.updatedAt = Date.now();
        await saveTask(task);
        refreshTaskUi(task);

        if (useServerProcessing) {
            // Server-side background processing with SSE progress
            // Do NOT clear isProcessing in finally — SSE/polling handler will do it
            await startServerProcessing(task, targetModel);
            return; // processTask is done; SSE handler will update UI
        }

        // Client-side sequential batch processing (original logic)
        for (const batch of task.batches) {
            if (batch.status === 'completed') continue;
            batch.status = 'processing';
            task.updatedAt = Date.now();
            await saveTask(task, { includeResults: false });
            refreshTaskUi(task);

            let result;
            try {
                await ensureBatchPayload(task, batch);
                result = await callOCR(batch, task);
            } finally {
                releaseBatchPayload(batch);
            }
            const prepared = prepareBatchResult(result, batch.id);
            batch.status = 'completed';
            batch.markdown = prepared.markdown;
            appendTaskMarkdown(task, prepared.markdown);
            Object.assign(task.images, prepared.images);
            task.ocrResults.push(...normalizeOCRJsonResults(result).map((pageResult, pageIndex) => (
                compactOCRJsonResult(pageResult, batch, pageIndex)
            )));
            if (Array.isArray(result.contentList)) {
                if (!Array.isArray(task.contentList)) task.contentList = [];
                task.contentList.push(...result.contentList);
            }
            task.updatedAt = Date.now();
            await saveTask(task);
            refreshTaskUi(task);
        }
        task.status = 'completed';
    } catch (error) {
        console.error(error);
        task.status = 'error';
        task.error = error.message;
        // Always clear isProcessing and save/refresh on error
        isProcessing = false;
        task.updatedAt = Date.now();
        await saveTask(task);
        refreshTaskUi(task);
    } finally {
        if (useServerProcessing && task.status !== 'error') {
            // For server processing (non-error), isProcessing is managed by SSE/polling handlers
            task.updatedAt = Date.now();
        } else if (task.status !== 'error') {
            // Client-side processing: normal cleanup
            isProcessing = false;
            task.updatedAt = Date.now();
            await saveTask(task, { includeResults: false });
            refreshTaskUi(task);
            // Notify the batch driver — client-side jobs don't hit SSE.
            signalBatchJobDone(task.id, 'completed');
        } else {
            // Error path: still need to release the queue.
            signalBatchJobDone(task.id, 'error');
        }
    }
}

async function startServerProcessing(task, model) {
    // Close any previous SSE/polling connections
    cleanupServerProcessingListeners();

    // Collect OCR options from UI controls
    const ocrOptions = collectOcrOptions(model.id);

    // Start background processing on server
    const response = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(task.id)}/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelId: model.id, ocrOptions })
    });
    if (!response.ok) {
        const detail = await responseErrorText(response);
        throw new Error(t('启动服务端处理失败：{detail}', { detail }));
    }

    // Listen to SSE for progress
    const progressUrl = `${API_BASE}/tasks/${encodeURIComponent(task.id)}/progress`;
    try {
        activeEventSource = new EventSource(progressUrl);
    } catch (_) {
        // SSE fallback: poll task status
        pollTaskProgress(task.id);
        return;
    }

    activeEventSource.onmessage = (event) => {
        try {
            const progress = JSON.parse(event.data);
            updateProcessingProgress(task.id, progress);
            if (progress.status === 'completed') {
                activeEventSource.close();
                activeEventSource = null;
                finishServerProcessing(task.id, 'completed');
            } else if (progress.status === 'error') {
                activeEventSource.close();
                activeEventSource = null;
                finishServerProcessing(task.id, 'error', progress.error);
            } else if (progress.status === 'cancelled') {
                activeEventSource.close();
                activeEventSource = null;
                finishServerProcessing(task.id, 'paused');
            } else if (progress.status === 'running') {
                // Refresh results from server every few progress updates
                refreshServerResults(task.id);
            }
        } catch (err) {
            console.warn('Failed to parse SSE progress event', err);
        }
    };

    activeEventSource.onerror = () => {
        if (activeEventSource) {
            activeEventSource.close();
            activeEventSource = null;
        }
        // Fallback to polling
        pollTaskProgress(task.id);
    };
}

function cleanupServerProcessingListeners() {
    if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
    }
    if (activePollInterval) {
        clearInterval(activePollInterval);
        activePollInterval = null;
    }
}

function collectOcrOptions(modelId) {
    const isMineru = modelId === 'mineru';
    const isGlmOcr = modelId === 'glm-ocr';
    const options = {};
    if (isMineru) {
        options.useChartRecognition = document.getElementById('mineru-image-switch')?.checked ?? true;
        options.useLayoutDetection = true;
        options.useSealRecognition = true;
        options.formatBlockContent = true;
        options.showFormulaNumber = true;
    } else if (isGlmOcr) {
        const glmLayoutSwitch = document.getElementById('glm-ocr-layout-switch');
        options.useLayoutDetection = glmLayoutSwitch?.checked ?? true;
        options.formatBlockContent = true;
    } else {
        options.useLayoutDetection = true;
        options.useChartRecognition = els.chartRecognitionSwitch?.checked ?? false;
        options.useDocUnwarping = els.docUnwarpingSwitch?.checked ?? false;
        options.useDocOrientationClassify = els.docOrientationSwitch?.checked ?? false;
        options.useSealRecognition = els.sealRecognitionSwitch?.checked ?? true;
        options.formatBlockContent = true;
        options.showFormulaNumber = els.formulaNumberSwitch?.checked ?? true;
        const ignoreLabels = [];
        if (els.ignoreNumberSwitch?.checked) ignoreLabels.push('number');
        ignoreLabels.push('footnote');
        if (els.ignoreHeaderSwitch?.checked) ignoreLabels.push('header', 'header_image');
        if (els.ignoreFooterSwitch?.checked) ignoreLabels.push('footer', 'footer_image');
        ignoreLabels.push('aside_text');
        options.markdownIgnoreLabels = ignoreLabels;
    }
    return options;
}

function updateProcessingProgress(taskId, progress) {
    const task = getActiveTask();
    if (!task || task.id !== taskId) return;

    const percent = progress.percent || 0;
    const batchIndex = progress.currentBatchIndex || 0;
    const totalBatches = progress.totalBatches || 0;
    const label = progress.currentBatchLabel || '';

    if (progress.status === 'running') {
        if (els.startBtn) {
            els.startBtn.innerHTML = `<span class="spinner"></span>${t('解析中')} ${percent}%`;
        }
        if (els.sourceMeta) {
            els.sourceMeta.textContent = t('解析批次 {current}/{total}: {label}', {
                current: batchIndex + 1,
                total: totalBatches,
                label
            });
        }
        // Drive the batch-queue stats line: every running progress
        // event advances the live page count by `currentBatchIndex`,
        // which the server reports as the number of batches already
        // finished. This is the only signal the stats bar has to stay
        // moving while a long task is mid-flight.
        if (batchQueue.running === taskId) {
            // Use batches already done in the running task + 1 for the
            // in-flight batch. We re-derive rather than increment so
            // out-of-order progress events can't drift the counter up.
            const runningTask = tasks.find((t) => t.id === taskId);
            const liveBatches = Number(batchIndex) + 1;
            const pagesFromRunning = Math.max(0, Math.min(liveBatches, totalBatches || liveBatches));
            // Subtract whatever this task contributed before the
            // running update so the run total stays consistent with
            // what the driver set when the job started.
            const taskContrib = Number(runningTask?.pageCount || 1);
            batchQueue.completedPages = Math.max(
                0,
                batchQueue.completedPages - taskContrib + pagesFromRunning
            );
            renderBatchQueueBar();
        }
    } else if (progress.status === 'cancelling') {
        if (els.startBtn) {
            els.startBtn.innerHTML = `<span class="spinner"></span>${t('正在取消...')}`;
        }
    }
}

let lastResultRefreshTime = 0;
const RESULT_REFRESH_INTERVAL = 3000; // 3 seconds
let mdRenderThrottle = null; // Throttle markdown re-renders during processing

async function refreshServerResults(taskId) {
    // Throttle: don't refresh more often than every 3 seconds
    const now = Date.now();
    if (now - lastResultRefreshTime < RESULT_REFRESH_INTERVAL) return;
    lastResultRefreshTime = now;

    try {
        const response = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}?lite=true`);
        if (!response.ok) return;
        const serverTask = await response.json();

        // Update local task data, preserving already-loaded heavy data
        const localIndex = tasks.findIndex((t) => t.id === taskId);
        if (localIndex >= 0) {
            const localTask = tasks[localIndex];
            if (localTask.images && Object.keys(localTask.images).length) {
                serverTask.images = { ...serverTask.images, ...localTask.images };
            }
            if (localTask.ocrResults?.length) {
                serverTask.ocrResults = localTask.ocrResults;
            }
            if (localTask._resultLoaded) serverTask._resultLoaded = true;
            if (localTask._imagesLoaded) serverTask._imagesLoaded = true;
            if (localTask._ocrLoaded) serverTask._ocrLoaded = true;
            tasks[localIndex] = serverTask;
        }

        // Refresh result pane if this task is active — with throttle
        if (activeTaskId === taskId) {
            // During processing, new images may have been added — load them incrementally
            if (serverTask._resultState?.hasImages && serverTask._imagesLoaded) {
                const localImageCount = Object.keys(serverTask.images || {}).length;
                const serverImageTotal = serverTask._imageTotal || 0;
                if (serverImageTotal > localImageCount) {
                    // More images available on server — fetch the new ones
                    loadTaskResult(taskId, {
                        fields: 'images',
                        imageOffset: localImageCount,
                        imageLimit: 200,
                    }).then((newData) => {
                        if (newData.images && typeof newData.images === 'object') {
                            const t = tasks.find((x) => x.id === taskId);
                            if (t) {
                                t.images = { ...t.images, ...newData.images };
                                t._imageTotal = newData.imageTotal || t._imageTotal;
                            }
                        }
                    }).catch(() => {});
                }
            } else if (serverTask._resultState?.hasImages && !serverTask._imagesLoaded) {
                // Images not loaded yet — load first batch
                serverTask._imagesLoaded = true; // prevent re-trigger
                loadTaskResult(taskId, {
                    fields: 'images',
                    imageLimit: 200,
                }).then((newData) => {
                    if (newData.images && typeof newData.images === 'object') {
                        serverTask.images = { ...serverTask.images, ...newData.images };
                        serverTask._imageTotal = newData.imageTotal || 0;
                    }
                }).catch(() => {});
            }
            scheduleMdRender(serverTask);
        }

        // Update task list in sidebar (lightweight)
        renderTaskList();
    } catch (err) {
        // Silent fail — don't disrupt processing
    }
}

function scheduleMdRender(task) {
    // Cancel any pending render
    if (mdRenderThrottle) {
        cancelAnimationFrame(mdRenderThrottle);
    }
    // Defer render to next animation frame — coalesces multiple SSE events
    mdRenderThrottle = requestAnimationFrame(() => {
        mdRenderThrottle = null;
        renderResultPane(task);
        updateActionState(task);
    });
}

async function finishServerProcessing(taskId, status, error = null) {
    isProcessing = false;
    // Clean up SSE/polling listeners
    cleanupServerProcessingListeners();
    // Reload task from server to get updated results
    const response = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}?lite=true`);
    if (response.ok) {
        const serverTask = await response.json();
        // Merge server results into local task, preserving already-loaded heavy data
        const localIndex = tasks.findIndex((t) => t.id === taskId);
        if (localIndex >= 0) {
            const localTask = tasks[localIndex];
            if (localTask.images && Object.keys(localTask.images).length) {
                serverTask.images = { ...serverTask.images, ...localTask.images };
            }
            if (localTask.ocrResults?.length) {
                serverTask.ocrResults = localTask.ocrResults;
            }
            if (localTask._resultLoaded) serverTask._resultLoaded = true;
            if (localTask._imagesLoaded) serverTask._imagesLoaded = true;
            if (localTask._ocrLoaded) serverTask._ocrLoaded = true;
            tasks[localIndex] = serverTask;
        }
        if (activeTaskId === taskId) {
            refreshTaskUi(serverTask);
        }
    } else {
        // Fallback: update local task
        const task = tasks.find((t) => t.id === taskId);
        if (task) {
            task.status = status;
            task.error = error;
            task.updatedAt = Date.now();
            await saveTask(task);
            refreshTaskUi(task);
        }
    }
    updateActionState(tasks.find((t) => t.id === taskId));
    // Release the batch queue driver — this is the signal it's been
    // waiting for to start the next file.
    const batchStatus = status === 'completed' ? 'completed' : (status === 'paused' ? 'skipped' : 'error');
    signalBatchJobDone(taskId, batchStatus);
}

function pollTaskProgress(taskId) {
    // Fallback polling when SSE is unavailable
    let lastPollRefresh = 0;
    activePollInterval = setInterval(async () => {
        try {
            const response = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}?lite=true`);
            if (!response.ok) return;
            const serverTask = await response.json();
            const localIndex = tasks.findIndex((t) => t.id === taskId);
            if (localIndex >= 0) {
                // Merge: preserve already-loaded heavy data from local task
                const localTask = tasks[localIndex];
                if (localTask.images && Object.keys(localTask.images).length) {
                    serverTask.images = { ...serverTask.images, ...localTask.images };
                }
                if (localTask.ocrResults?.length) {
                    serverTask.ocrResults = localTask.ocrResults;
                }
                if (localTask._resultLoaded) serverTask._resultLoaded = true;
                if (localTask._imagesLoaded) serverTask._imagesLoaded = true;
                if (localTask._ocrLoaded) serverTask._ocrLoaded = true;
                tasks[localIndex] = serverTask;
            }
            if (serverTask.status === 'completed' || serverTask.status === 'error' || serverTask.status === 'paused') {
                clearInterval(activePollInterval);
                activePollInterval = null;
                isProcessing = false;
                if (activeTaskId === taskId) {
                    refreshTaskUi(serverTask);
                }
                renderTaskList();
                updateActionState(serverTask);
                const batchStatus = serverTask.status === 'completed'
                    ? 'completed'
                    : (serverTask.status === 'paused' ? 'skipped' : 'error');
                signalBatchJobDone(taskId, batchStatus);
            } else if (activeTaskId === taskId) {
                // Update progress display
                const completedBatches = (serverTask.batches || []).filter((b) => b.status === 'completed').length;
                const totalBatches = (serverTask.batches || []).length;
                const percent = totalBatches > 0 ? Math.round(completedBatches / totalBatches * 100) : 0;
                // Show cancel button during polling
                if (els.startBtn) {
                    els.startBtn.disabled = false;
                    els.startBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M6 6h12v12H6z"/></svg>${t('停止')} ${percent}%`;
                }
                // Refresh results every 3 seconds
                const now = Date.now();
                if (now - lastPollRefresh >= RESULT_REFRESH_INTERVAL) {
                    lastPollRefresh = now;
                    renderResultPane(serverTask);
                    renderTaskList();
                }
            }
        } catch (err) {
            console.warn('Polling failed', err);
        }
    }, 2000);
}

async function cancelServerProcessing(taskId) {
    const response = await apiFetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}/cancel`, {
        method: 'POST'
    });
    if (!response.ok) {
        const detail = await responseErrorText(response);
        console.warn('Cancel failed:', detail);
    }
}

function shouldResumeTask(task) {
    if (isTaskActivelyProcessing(task)) return false;
    // Allow resume from any non-completed status (pending, error, paused, etc.)
    // as long as some batches are already done and some are not.
    const canResumeStatus = task?.status !== 'completed';
    if (!canResumeStatus) return false;

    if (Array.isArray(task?.batches)) {
        const completedBatchCount = task.batches.filter((batch) => batch.status === 'completed').length;
        const hasPendingBatch = task.batches.some((batch) => batch.status !== 'completed');
        return completedBatchCount > 0 && hasPendingBatch;
    }

    const completedPages = Number(task?.completedPages || 0);
    const pageCount = Number(task?.pageCount || 0);
    return completedPages > 0 && (!pageCount || completedPages < pageCount);
}

async function resetTaskForFullRerun(task) {
    // Reset all batch statuses and clear results so the task can be re-parsed from scratch.
    task.markdown = '';
    task.images = {};
    task.ocrResults = [];
    task.contentList = [];
    if (Array.isArray(task.batches)) {
        task.batches.forEach((batch) => {
            batch.status = 'pending';
            batch.markdown = '';
        });
    }
    task.status = 'pending';
    task.error = null;
    task.updatedAt = Date.now();
    await saveTask(task);
}

function isTaskActivelyProcessing(task) {
    return task?.status === 'processing'
        || Boolean(task?.batches?.some((batch) => batch.status === 'processing'));
}

function shouldRebuildPdfBatchPlan(task) {
    if (!task || !(task.sourceDataUrl || task.sourceUrl) || !['pdf', 'office'].includes(task.sourceKind)) return false;
    const pageCount = Number(task.pageCount || 0);
    if (pageCount <= 0) return false;
    const batches = Array.isArray(task.batches) ? task.batches : [];
    const completedCount = batches.filter((batch) => batch.status === 'completed').length;
    if (completedCount > 0) return false;
    const configuredBatchSize = getConfiguredPdfBatchSize();
    if (batches.length === 0) return true;
    if (Number(task.pdfBatchSize || 0) !== configuredBatchSize) return true;
    return Number(task.pdfBatchSize || 0) > MAX_PDF_BATCH_SIZE
        || batches.some((batch) => Number(batch.pageCount || 0) > MAX_PDF_BATCH_SIZE);
}

function rebuildPdfBatchPlan(task) {
    const pageCount = Number(task.pageCount || 1);
    const batchSize = getConfiguredPdfBatchSize();
    task.pdfBatchSize = batchSize;
    task.batches = createPdfBatchDescriptors(pageCount, batchSize, task.sourceDataUrl);
    task.markdown = '';
    task.images = {};
    task.ocrResults = [];
}

function taskVisualStatus(task) {
    if (isTaskActivelyProcessing(task)) return 'processing';
    return shouldResumeTask(task) ? 'pending' : (task?.status || 'pending');
}

function rebuildTaskResultFromCompletedBatches(task) {
    const completedBatches = task.batches.filter((batch) => batch.status === 'completed');
    if (completedBatches.length === 0) return;

    const existingMarkdown = task.markdown || '';
    const hasBatchMarkdown = completedBatches.some((batch) => batch.markdown);
    if (!existingMarkdown && hasBatchMarkdown) {
        task.markdown = completedBatches
            .map((batch) => batch.markdown || '')
            .filter(Boolean)
            .join('\n\n');
    }

    if (!task.images || typeof task.images !== 'object') {
        task.images = {};
    }
    if (!Array.isArray(task.ocrResults)) {
        task.ocrResults = [];
    }
}

function appendTaskMarkdown(task, markdown) {
    const text = String(markdown || '');
    if (!text) return;
    if (task.markdown && !task.markdown.endsWith('\n\n')) {
        task.markdown += '\n\n';
    }
    task.markdown += `${text}\n\n`;
}

function refreshTaskUi(task) {
    renderTaskList();
    const activeTask = getActiveTask();
    if (task?.id === activeTaskId) {
        updateActiveModelDisplay(task);
        renderResultPane(task);
    }
    updateActionState(activeTask);
}

async function callOCR(batch, task) {
    const model = getTaskModel(task);
    const isMineru = model.id === 'mineru';
    const isGlmOcr = model.id === 'glm-ocr';
    const ignoreLabels = [];
    if (!isMineru && !isGlmOcr) {
        if (els.ignoreNumberSwitch.checked) ignoreLabels.push('number');
        ignoreLabels.push('footnote');
        if (els.ignoreHeaderSwitch.checked) ignoreLabels.push('header', 'header_image');
        if (els.ignoreFooterSwitch.checked) ignoreLabels.push('footer', 'footer_image');
        ignoreLabels.push('aside_text');
    }

    const formData = new FormData();
    const filename = batch.fileType === 0 ? `${batch.id}.pdf` : `${batch.id}.image`;
    if (batch.payloadBlob) {
        formData.append('file', batch.payloadBlob, filename);
    } else if (batch.payloadDataUrl) {
        formData.append('file', dataUrlToBlob(batch.payloadDataUrl), filename);
    } else {
        throw new Error(t('无法重建当前批次的解析 payload'));
    }
    formData.append('fileType', String(batch.fileType));

    if (isMineru) {
        formData.append('useChartRecognition', String(document.getElementById('mineru-image-switch')?.checked ?? true));
        formData.append('useLayoutDetection', 'true');
        formData.append('useSealRecognition', 'true');
        formData.append('formatBlockContent', 'true');
        formData.append('showFormulaNumber', 'true');
        formData.append('modelId', model.id);
    } else if (isGlmOcr) {
        const glmLayoutSwitch = document.getElementById('glm-ocr-layout-switch');
        formData.append('useLayoutDetection', String(glmLayoutSwitch?.checked ?? true));
        formData.append('formatBlockContent', 'true');
        formData.append('modelId', model.id);
    } else {
        formData.append('useLayoutDetection', 'true');
        formData.append('useChartRecognition', String(els.chartRecognitionSwitch.checked));
        formData.append('useDocUnwarping', String(els.docUnwarpingSwitch.checked));
        formData.append('useDocOrientationClassify', String(els.docOrientationSwitch.checked));
        formData.append('useSealRecognition', String(els.sealRecognitionSwitch.checked));
        formData.append('formatBlockContent', 'true');
        formData.append('showFormulaNumber', String(els.formulaNumberSwitch.checked));
        formData.append('markdownIgnoreLabels', JSON.stringify(ignoreLabels));
        formData.append('modelId', model.id);
    }

    const response = await apiFetch(modelApiUrl(model), {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        throw new Error(await responseErrorText(response));
    }
    const text = await response.text();
    if (!text.trim()) {
        throw new Error(t('OCR 服务返回了空响应，请降低每批页数后重试：{label}', { label: batch.label || '' }));
    }
    try {
        return JSON.parse(text);
    } catch (error) {
        const preview = text.slice(0, 500);
        throw new Error(
            t('OCR 服务返回的 JSON 不完整或格式异常，请降低每批页数后重试：{label}。响应长度 {length} 字符，片段：{preview}', {
                label: batch.label || '',
                length: text.length,
                preview
            })
        );
    }
}

function createPdfBatchDescriptors(pageCount, pdfBatchSize, sourceDataUrl = '') {
    const batches = [];
    for (let startPage = 1; startPage <= pageCount; startPage += pdfBatchSize) {
        const endPage = Math.min(startPage + pdfBatchSize - 1, pageCount);
        const batch = {
            id: createId(),
            label: formatPageLabel(startPage, endPage),
            fileType: 0,
            startPage,
            endPage,
            pageCount: endPage - startPage + 1,
            status: 'pending'
        };
        if (pageCount === 1 && sourceDataUrl) {
            batch.payloadDataUrl = sourceDataUrl;
        }
        batches.push(batch);
    }
    return batches;
}

function taskForPersistence(task, { includeResults = true } = {}) {
    const persisted = { ...task };
    delete persisted.detailLoaded;
    delete persisted._storage;
    delete persisted._resultState;
    delete persisted._resultLoaded;
    delete persisted._imagesLoaded;
    delete persisted._ocrLoaded;
    delete persisted._imageTotal;
    delete persisted._ocrTotal;
    if (persisted.sourceUrl) {
        delete persisted.sourceDataUrl;
    }
    if (!includeResults) {
        delete persisted.markdown;
        delete persisted.images;
        delete persisted.ocrResults;
        persisted._preserveResult = true;
    }
    persisted.batches = Array.isArray(task.batches)
        ? task.batches.map((batch) => {
            const copy = { ...batch };
            delete copy.payloadBlob;
            delete copy.payloadDataUrl;
            if (!includeResults) delete copy.markdown;
            return copy;
        })
        : [];
    return persisted;
}

function updateActionState(task) {
    const hasResult = Boolean(task?.markdown) || Boolean(task?.ocrResults?.length);
    const taskModel = task ? getTaskModel(task) : getSelectedModel();
    const modelReady = !task || isModelRuntimeReady(taskModel.id);
    const canStartAfterSwitch = task && !modelReady && canSwitchModelRuntime(taskModel.id);
    const modelStarting = task && !modelReady && isModelRuntimeSwitching(taskModel.id);
    const isServerProcessing = isProcessing && task?.sourceUrl && !task?.sourceDataUrl;
    const isBackendProcessing = task?.status === 'processing' && task?.sourceUrl && !task?.sourceDataUrl;
    if (els.modelSelect) {
        els.modelSelect.disabled = isProcessing || modelSwitchInFlight || isModelRuntimeSwitching();
    }
    if ((isServerProcessing || isBackendProcessing) && task?.status === 'processing') {
        // Show cancel button during server-side processing
        els.startBtn.disabled = false;
        els.startBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M6 6h12v12H6z"/></svg>${t('停止')}`;
    } else {
        els.startBtn.disabled = !task || !isTaskDetailLoaded(task) || isProcessing || (!modelReady && !canStartAfterSwitch);
        const startLabel = startButtonLabel(task);
        const showProcessing = (isProcessing && task?.status === 'processing') || modelStarting;
        els.startBtn.innerHTML = showProcessing
            ? `<span class="spinner"></span>${modelStarting ? t('模型启动中') : t('解析中')}`
            : `<svg viewBox="0 0 24 24"><path d="m8 5 11 7-11 7V5Z"/></svg>${startLabel}`;
    }
    updateCopyButtonState(task);
    els.downloadBtn.disabled = !hasResult;
}

function startButtonLabel(task) {
    if (!task) return t('开始解析');
    const taskModel = getTaskModel(task);
    if (!isModelRuntimeReady(taskModel.id)) return t('启动模型并解析');
    if (task.status === 'completed') return t('重新解析');
    if (task.status === 'error') return t('重试解析');
    if (shouldResumeTask(task)) return t('继续解析');
    return t('开始解析');
}

function hasJsonResult(task) {
    return Boolean(task?.ocrResults?.length);
}

function canCopyActiveResult(task) {
    if (activeResultView === 'json') return hasJsonResult(task);
    return Boolean(task?.markdown);
}

function copyButtonLabel() {
    return activeResultView === 'json' ? t('复制 JSON') : t('复制 Markdown');
}

function updateCopyButtonState(task) {
    const label = copyButtonLabel();
    els.copyBtn.disabled = !canCopyActiveResult(task);
    els.copyBtn.setAttribute('title', label);
    els.copyBtn.setAttribute('aria-label', label);
    // Show clear-result button only when there are results or task is in error/paused state
    const hasResults = Boolean(task?.markdown) || Boolean(task?.ocrResults?.length);
    const canClear = hasResults || task?.status === 'error' || task?.status === 'paused';
    if (els.clearResultBtn) {
        els.clearResultBtn.style.display = canClear ? '' : 'none';
    }
    if (els.translateBtn) {
        if (!hasResults && !isTranslating) {
            els.translateBtn.style.display = 'none';
        } else if (isTranslating) {
            // Show translate button as stop button during translation
            els.translateBtn.style.display = '';
            els.translateBtn.title = t('停止翻译');
        } else {
            els.translateBtn.style.display = '';
            if (task?.translation) {
                // Already translated — toggle between original / translation
                if (isViewingTranslation) {
                    els.translateBtn.title = t('查看原文');
                } else {
                    els.translateBtn.title = t('查看翻译');
                }
            } else if (translateAvailable) {
                els.translateBtn.title = t('翻译');
            } else {
                els.translateBtn.style.display = 'none';
            }
        }
    }
}

function activeResultCopyText(task) {
    if (!task) return '';
    if (activeResultView === 'json') {
        if (!hasJsonResult(task)) return '';
        return JSON.stringify(toOfficialJson(task), null, 2);
    }
    return task.markdown ? normalizeOCRMarkdown(task.markdown) : '';
}

async function copyActiveResult() {
    const task = getActiveTask();
    const text = activeResultCopyText(task);
    if (!text) return;
    try {
        await writeClipboardText(text);
        els.copyBtn.classList.add('success');
        setTimeout(() => els.copyBtn.classList.remove('success'), 900);
    } catch (error) {
        console.error(error);
        alert(error.message || t('复制失败'));
    }
}

async function clearActiveResult() {
    const task = getActiveTask();
    if (!task) return;
    if (!task.markdown && !task.ocrResults?.length && task.status !== 'error' && task.status !== 'paused') return;
    if (!confirm(t('确定清空当前任务的解析结果？此操作不可撤销。'))) return;
    await resetTaskForFullRerun(task);
    const updated = await loadTaskFromServer(task.id);
    replaceTask(updated);
    refreshTaskUi(updated);
}

// ----- Translation -----

const TRANSLATE_LANGUAGES = [
    { code: 'zh-CN', name: '简体中文' },
    { code: 'zh-TW', name: '繁體中文' },
    { code: 'en', name: 'English' },
    { code: 'ja', name: '日本語' },
    { code: 'ko', name: '한국어' },
    { code: 'fr', name: 'Français' },
    { code: 'de', name: 'Deutsch' },
    { code: 'es', name: 'Español' },
    { code: 'ru', name: 'Русский' },
];

let translateAvailable = false;
let isTranslating = false;
let isViewingTranslation = false; // whether we are currently showing the translation
let translateAbortController = null; // for cancelling translation

async function checkTranslateAvailable() {
    try {
        const resp = await apiFetch(`${API_BASE}/translate/config`);
        if (resp.ok) {
            const config = await resp.json();
            translateAvailable = config.available;
        }
    } catch (_) {}
}

function showTranslateDialog() {
    const task = getActiveTask();
    if (!task || !task.markdown) return;

    // If currently translating, click = stop
    if (isTranslating && translateAbortController) {
        translateAbortController.abort();
        return;
    }

    if (isTranslating) return;

    // If we already have a saved translation, show options: view / re-translate
    if (task.translation) {
        if (isViewingTranslation) {
            // Currently viewing translation — switch back to original
            isViewingTranslation = false;
            const indicator = document.querySelector('.translation-indicator');
            if (indicator) indicator.remove();
            resetResultRenderCache(task.id);
            lastRenderedHtml = '';
            renderResultPane(task);
        } else {
            // Not viewing translation — show it
            isViewingTranslation = true;
            showTranslationResult(task);
        }
        updateActionState(task);
        return;
    }

    // No existing translation — show the language picker dialog
    if (!translateAvailable) {
        alert(t('翻译功能未配置。请在环境变量中设置 PANDOCR_TRANSLATE_API_URL 和 PANDOCR_TRANSLATE_API_KEY。'));
        return;
    }

    showTranslateLangDialog(task);
}

function showTranslateLangDialog(task, isRetranslate = false) {
    const existing = document.getElementById('translate-dialog');
    if (existing) existing.remove();

    const dialog = document.createElement('div');
    dialog.id = 'translate-dialog';
    dialog.style.cssText = 'position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.5)';

    const currentLang = task.translationLang || '';
    const title = isRetranslate ? t('重新翻译') : t('翻译');
    const options = TRANSLATE_LANGUAGES.map((lang) =>
        `<option value="${lang.code}" ${lang.code === currentLang ? 'selected' : ''} ${lang.code === 'zh-CN' && !currentLang ? 'selected' : ''}>${lang.name}</option>`
    ).join('');

    dialog.innerHTML = `
        <div style="background:var(--surface-color,#fff);border-radius:12px;padding:24px;min-width:300px;box-shadow:0 8px 32px rgba(0,0,0,0.3)">
            <h3 style="margin:0 0 16px;font-size:16px">${title}</h3>
            <label style="display:block;margin-bottom:8px;font-size:13px;color:var(--text-secondary,#666)">${t('目标语言')}</label>
            <select id="translate-target-lang" style="width:100%;padding:8px;border-radius:6px;border:1px solid var(--border-color,#ddd);font-size:14px;background:var(--input-bg,#fff);color:var(--text-color,#333)">${options}</select>
            <div style="display:flex;gap:8px;margin-top:16px;justify-content:flex-end">
                <button id="translate-cancel" style="padding:6px 16px;border-radius:6px;border:1px solid var(--border-color,#ddd);cursor:pointer;background:var(--surface-color,#fff);color:var(--text-color,#333)">${t('取消')}</button>
                <button id="translate-start" style="padding:6px 16px;border-radius:6px;border:none;cursor:pointer;background:var(--primary-color,#4f46e5);color:#fff;font-weight:500">${title}</button>
            </div>
        </div>`;

    document.body.appendChild(dialog);

    document.getElementById('translate-cancel').onclick = () => dialog.remove();
    dialog.onclick = (e) => { if (e.target === dialog) dialog.remove(); };
    document.getElementById('translate-start').onclick = () => {
        const targetLang = document.getElementById('translate-target-lang').value;
        dialog.remove();
        startTranslation(task.id, targetLang);
    };
}

async function startTranslation(taskId, targetLang) {
    if (isTranslating) return;
    isTranslating = true;

    // Create abort controller for cancellation
    translateAbortController = new AbortController();

    // Show progress bar on translate button
    const btn = els.translateBtn;
    if (btn) {
        btn.style.color = 'var(--primary-color, #4f46e5)';
        btn.title = t('翻译中... 0%');
        btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;vertical-align:middle"></span><span style="vertical-align:middle;font-size:12px;margin-left:3px">0%</span>`;
        btn.style.minWidth = '72px';
        btn.style.whiteSpace = 'nowrap';
    }

    try {
        const url = `${API_BASE}/tasks/${encodeURIComponent(taskId)}/translate`;
        const resp = await apiFetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ targetLang }),
            signal: translateAbortController.signal,
        });

        if (!resp.ok) {
            const detail = await responseErrorText(resp);
            throw new Error(detail);
        }

        // Read SSE stream
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let lastPercent = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.type === 'progress') {
                        lastPercent = data.percent;
                        if (btn) {
                            btn.title = t('翻译中... {p}%', { p: data.percent });
                            btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;border-width:2px;display:inline-block;vertical-align:middle"></span><span style="vertical-align:middle;font-size:12px;margin-left:3px">${data.percent}%</span>`;
                        }
                    } else if (data.type === 'error') {
                        console.warn('Translation chunk error:', data.error);
                    } else if (data.type === 'done') {
                        // Translation complete — but only update if user is still on this task
                        if (activeTaskId !== taskId) break;
                        const updated = await loadTaskFromServer(taskId);
                        replaceTask(updated);
                        isViewingTranslation = true;
                        showTranslationResult(updated);
                        updateActionState(updated);
                    }
                } catch (_) {}
            }
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            // User cancelled — translation was stopped mid-way
            // Reload task to get whatever was saved
            const updated = await loadTaskFromServer(taskId);
            replaceTask(updated);
            if (updated.translation) {
                isViewingTranslation = true;
                showTranslationResult(updated);
            }
        } else {
            alert(t('翻译失败：{err}', { err: err.message }));
        }
    } finally {
        isTranslating = false;
        translateAbortController = null;
        if (btn) {
            btn.style.color = '';
            btn.style.minWidth = '';
            btn.style.whiteSpace = '';
            btn.title = t('翻译');
            // Restore the translate SVG icon
            btn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M12.87 15.07l-2.54-2.51.03-.03A17.52 17.52 0 0014.07 6H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"/></svg>`;
        }
        updateActionState(getActiveTask());
    }
}

function showTranslationResult(task) {
    if (!task.translation) return;
    // Safety: only show if this task is still the active one
    if (activeTaskId && task.id !== activeTaskId) return;

    // Clear render caches
    resetResultRenderCache(task.id);
    lastRenderedHtml = '';
    renderedOfficialLayoutContext = '';
    renderedPPOCRVisualContext = '';

    // Render the translation as plain markdown (bypass layout/ppocr visual modes)
    const html = renderMarkdownHtml(prepareMarkdownForRender(task.translation));
    els.markdownView.innerHTML = html;
    renderMathWhenReady(els.markdownView);

    // Remove any existing indicator
    const old = document.querySelector('.translation-indicator');
    if (old) old.remove();

    // Add indicator bar: language label | download | re-translate | view original
    const indicator = document.createElement('div');
    indicator.className = 'translation-indicator';
    indicator.style.cssText = 'padding:6px 16px;background:var(--primary-color,#4f46e5);color:#fff;font-size:13px;display:flex;align-items:center;gap:6px;border-radius:8px 8px 0 0;position:sticky;top:0;z-index:10;flex-wrap:wrap';
    const langLabel = LANG_DISPLAY_NAMES?.[task.translationLang] || task.translationLang || '';
    const btnStyle = 'background:rgba(255,255,255,0.2);border:none;color:#fff;padding:3px 10px;border-radius:4px;cursor:pointer;font-size:12px;white-space:nowrap';
    indicator.innerHTML = `
        <span style="flex:1">${t('{lang} 翻译', { lang: langLabel })}</span>
        <button id="download-translation-btn" style="${btnStyle}" title="${t('下载翻译MD')}">⬇ ${t('下载')}</button>
        <button id="retranslate-btn" style="${btnStyle}" title="${t('重新翻译')}">🔄 ${t('重翻')}</button>
        <button id="view-original-btn" style="${btnStyle}" title="${t('查看原文')}">↩ ${t('原文')}</button>`;

    // Download translation as .md
    indicator.querySelector('#download-translation-btn').onclick = () => {
        const name = (task.name || 'translation').replace(/\.[^.]+$/, '');
        const lang = task.translationLang || 'translated';
        downloadBlob(new Blob([task.translation], { type: 'text/markdown;charset=utf-8' }), `${name}_${lang}.md`);
    };

    // Re-translate — open language dialog
    indicator.querySelector('#retranslate-btn').onclick = () => {
        indicator.remove();
        isViewingTranslation = false;
        resetResultRenderCache(task.id);
        lastRenderedHtml = '';
        renderResultPane(task);
        showTranslateLangDialog(task, true);
    };

    // View original — just toggle state, don't remove indicator tracking
    indicator.querySelector('#view-original-btn').onclick = () => {
        isViewingTranslation = false;
        indicator.remove();
        resetResultRenderCache(task.id);
        lastRenderedHtml = '';
        renderResultPane(task);
        updateActionState(task);
    };
    els.markdownView.parentElement.insertBefore(indicator, els.markdownView);
}

// Display names for translation target languages
const LANG_DISPLAY_NAMES = {
    'zh-CN': '简体中文', 'zh-TW': '繁體中文',
    'en': 'English', 'ja': '日本語', 'ko': '한국어',
    'fr': 'Français', 'de': 'Deutsch', 'es': 'Español',
    'pt': 'Português', 'ru': 'Русский', 'ar': 'العربية',
    'it': 'Italiano', 'nl': 'Nederlands', 'pl': 'Polski',
    'tr': 'Türkçe', 'vi': 'Tiếng Việt', 'th': 'ไทย',
    'id': 'Bahasa Indonesia', 'ms': 'Bahasa Melayu', 'hi': 'हिन्दी',
};

async function downloadActiveTask() {
    const task = getActiveTask();
    if (!task?.markdown && !task?.ocrResults?.length) return;

    if (activeResultView === 'json') {
        const json = JSON.stringify(toOfficialJson(task), null, 2);
        downloadBlob(new Blob([json], { type: 'application/json' }), safeDownloadName(task.name, 'json'));
        return;
    }

    const markdown = normalizeOCRMarkdown(task.markdown);
    const imageEntries = Object.entries(task.images || {});
    if (imageEntries.length === 0) {
        downloadBlob(new Blob([markdown], { type: 'text/markdown' }), safeDownloadName(task.name, 'md'));
        return;
    }

    const zip = new JSZip();
    let rewritten = markdown;
    const folder = zip.folder('ocr_images');
    for (const [path, base64] of imageEntries) {
        const filename = path.split('/').pop();
        rewritten = rewritten.split(path).join(`ocr_images/${filename}`);
        folder.file(filename, base64ToBytes(base64));
    }
    zip.file('README.md', rewritten);
    const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 6 } });
    downloadBlob(blob, safeDownloadName(task.name, 'zip'));
}

async function clearHistory() {
    if (!confirm(t('确认清空所有本地任务历史吗？'))) return;
    try {
        await deleteAllTasks();
    } catch (error) {
        console.error(error);
        alert(error.message || t('清空失败，请稍后重试。'));
        return;
    }
    tasks = [];
    activeTaskId = null;
    resetWorkbench();
}

function resetWorkbench() {
    renderTaskList();
    sourceRenderToken += 1;
    currentPdf = null;
    els.sourceTitle.textContent = t('等待上传文件');
    els.sourceMeta.textContent = t('PDF、图片、Office 文档');
    els.pdfControls.classList.add('hidden');
    els.sourceViewer.innerHTML = emptyDropZoneHtml();
    els.dropZone = document.getElementById('drop-zone');
    els.browseBtn = document.getElementById('browse-btn');
    els.browseBtn.addEventListener('click', () => els.fileInput.click());
    renderResultPane(null);
    updateActiveModelDisplay(null);
    updateActionState(null);
}

function changePdfPage(delta) {
    if (!currentPdf) return;
    currentPage = Math.min(Math.max(currentPage + delta, 1), currentPdf.numPages);
    resetSplitHorizontalScroll();
    scrollPdfPageIntoView(currentPage, 'smooth');
    syncPPOCRVisualScrollFromSource();
    updatePdfControls();
}

async function changeZoom(delta) {
    if (!currentPdf) return;
    const scrollAnchor = captureSourceScrollAnchor();
    currentZoom = Math.min(2.2, Math.max(0.55, currentZoom + delta));
    await renderPdfDocument(++sourceRenderToken, scrollAnchor);
    const task = getActiveTask();
    if (task && activeResultView === 'markdown') {
        if (isPPOCRVisualTask(task)) {
            invalidatePPOCRVisualRender();
        }
        renderResultPane(task);
        queueSyncedScrollRestore(scrollAnchor);
    }
}

async function resetZoom() {
    if (!currentPdf) return;
    const scrollAnchor = resetAnchorHorizontal(captureSourceScrollAnchor());
    currentZoom = getDefaultPdfZoom();
    await renderPdfDocument(++sourceRenderToken, scrollAnchor);
    const task = getActiveTask();
    if (task && activeResultView === 'markdown') {
        if (isPPOCRVisualTask(task)) {
            invalidatePPOCRVisualRender();
        }
        renderResultPane(task);
        queueSyncedScrollRestore(scrollAnchor);
    }
}

function scrollPdfPageIntoView(pageNumber, behavior = 'smooth') {
    const page = sourcePageSurface(pageNumber)?.container;
    if (!page) return;
    const top = Number(pageNumber) <= 1 ? 0 : page.offsetTop - els.sourceViewer.offsetTop - 12;
    els.sourceViewer.scrollTo({ top: Math.max(top, 0), behavior });
}

function handleSourceViewerScroll() {
    updateCurrentPageFromScroll();
    scheduleSourceToPPOCRScrollSync();
}

function updateCurrentPageFromScroll() {
    if (!currentPdf) return;
    const pages = Array.from(els.sourceViewer.querySelectorAll('.pdf-page-wrap'));
    if (!pages.length) return;

    const viewerTop = els.sourceViewer.getBoundingClientRect().top;
    let nearestPage = currentPage;
    let nearestDistance = Infinity;

    pages.forEach((page) => {
        const distance = Math.abs(page.getBoundingClientRect().top - viewerTop - 16);
        if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestPage = Number(page.dataset.page);
        }
    });

    if (nearestPage !== currentPage) {
        currentPage = nearestPage;
        updatePdfControls();
    }
}

function updatePdfControls() {
    if (!currentPdf) return;
    els.pageIndicator.textContent = `${currentPage} / ${currentPdf.numPages}`;
    els.prevPageBtn.disabled = currentPage <= 1;
    els.nextPageBtn.disabled = currentPage >= currentPdf.numPages;
    if (els.resetZoomBtn) {
        els.resetZoomBtn.disabled = Math.abs(currentZoom - getDefaultPdfZoom()) < 0.01;
    }
}

function getDefaultPdfZoom() {
    const viewer = els.sourceViewer;
    if (!viewer) return DEFAULT_PDF_ZOOM;
    const styles = getComputedStyle(viewer);
    const horizontalPadding = (Number.parseFloat(styles.paddingLeft) || 0)
        + (Number.parseFloat(styles.paddingRight) || 0);
    const availableWidth = Math.max(0, viewer.clientWidth - horizontalPadding - PDF_FIT_WIDTH_GUTTER);
    if (!availableWidth || !pdfDefaultPageWidth) return DEFAULT_PDF_ZOOM;
    const fitZoom = availableWidth / pdfDefaultPageWidth;
    return roundPdfZoom(Math.max(DEFAULT_PDF_ZOOM, Math.min(MAX_DEFAULT_PDF_ZOOM, fitZoom)));
}

function roundPdfZoom(value) {
    return Math.round(value * 100) / 100;
}

function captureSourceScrollAnchor() {
    const page = getActiveSourcePage();
    if (!page) {
        return {
            pageNumber: currentPage,
            progress: 0,
            xRatio: horizontalScrollRatio(els.sourceViewer)
        };
    }

    const pageNumber = Number(page.dataset.page || currentPage);
    const pageTop = sourcePageTop(page);
    const scrollable = Math.max(1, page.offsetHeight - els.sourceViewer.clientHeight);
    return {
        pageNumber,
        progress: Math.min(Math.max((els.sourceViewer.scrollTop - pageTop) / scrollable, 0), 1),
        xRatio: horizontalScrollRatio(els.sourceViewer)
    };
}

function restoreSourceScrollAnchor(anchor, behavior = 'auto') {
    if (!anchor) return;
    const page = els.sourceViewer.querySelector(`.pdf-page-wrap[data-page="${anchor.pageNumber}"]`);
    if (!page) return;
    const scrollable = Math.max(0, page.offsetHeight - els.sourceViewer.clientHeight);
    const targetTop = sourcePageTop(page) + (anchor.progress || 0) * scrollable;
    els.sourceViewer.scrollTo({
        top: Math.max(targetTop, 0),
        left: horizontalScrollTarget(els.sourceViewer, anchor.xRatio || 0),
        behavior
    });
    currentPage = Number(anchor.pageNumber || currentPage);
    updatePdfControls();
}

function queueSyncedScrollRestore(anchor) {
    window.setTimeout(() => {
        restoreSourceScrollAnchor(anchor, 'auto');
        syncPPOCRVisualScrollFromSource();
    }, 120);
    window.setTimeout(() => {
        restoreSourceScrollAnchor(anchor, 'auto');
        syncPPOCRVisualScrollFromSource();
    }, 360);
}

function getActiveSourcePage() {
    const pages = Array.from(els.sourceViewer.querySelectorAll('.pdf-page-wrap'));
    if (!pages.length) return null;

    const viewerTop = els.sourceViewer.getBoundingClientRect().top;
    let nearestPage = pages[0];
    let nearestDistance = Infinity;
    pages.forEach((page) => {
        const distance = Math.abs(page.getBoundingClientRect().top - viewerTop - 16);
        if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestPage = page;
        }
    });
    return nearestPage;
}

function sourcePageTop(page) {
    if (!page) return 0;
    const pageNumber = Number(page.dataset.page || 1);
    return pageNumber <= 1 ? 0 : page.offsetTop - els.sourceViewer.offsetTop - 12;
}

function horizontalScrollRatio(container) {
    const maxScroll = Math.max(0, container.scrollWidth - container.clientWidth);
    return maxScroll ? Math.min(Math.max(container.scrollLeft / maxScroll, 0), 1) : 0;
}

function horizontalScrollTarget(container, ratio) {
    const maxScroll = Math.max(0, container.scrollWidth - container.clientWidth);
    return maxScroll * Math.min(Math.max(ratio, 0), 1);
}

function resetAnchorHorizontal(anchor) {
    if (!anchor) return anchor;
    return {
        ...anchor,
        xRatio: 0
    };
}

function resetSplitHorizontalScroll() {
    if (!els.sourceViewer || !els.markdownView) return;
    withSplitScrollLock(() => {
        els.sourceViewer.scrollLeft = 0;
        els.markdownView.scrollLeft = 0;
    });
}

function withSplitScrollLock(callback) {
    splitScrollSyncLocked = true;
    try {
        callback();
    } finally {
        requestAnimationFrame(() => {
            splitScrollSyncLocked = false;
        });
    }
}

function scheduleSourceToPPOCRScrollSync() {
    if (splitScrollSyncLocked || sourceScrollSyncFrame) return;
    sourceScrollSyncFrame = requestAnimationFrame(() => {
        sourceScrollSyncFrame = 0;
        syncPPOCRVisualScrollFromSource();
    });
}

async function renderPDFPageDataUrl(pdf, pageNumber, scale) {
    const page = await pdf.getPage(pageNumber);
    const viewport = page.getViewport({ scale });
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    await page.render({ canvasContext: context, viewport }).promise;
    return canvas.toDataURL('image/jpeg', 0.78);
}

async function renderPDFPageDataUrlFromSource(taskId, pageNumber, scale) {
    // Render a PDF page thumbnail from the server — extract just one page
    // to avoid downloading the entire PDF into browser memory
    try {
        const pageUrl = `${API_BASE}/tasks/${encodeURIComponent(taskId)}/source/pages?start_page=${pageNumber}&end_page=${pageNumber}`;
        const response = await apiFetch(pageUrl);
        if (!response.ok) throw new Error(t('无法加载 PDF 页面'));
        const arrayBuffer = await response.arrayBuffer();
        const pdf = await loadPdf(arrayBuffer.slice(0));
        return renderPDFPageDataUrl(pdf, 1, scale);
    } catch (err) {
        // Fallback: return empty thumbnail
        console.warn('Failed to render thumbnail from server', err);
        return null;
    }
}

async function loadPdfFromUrl(url) {
    const response = await apiFetch(url);
    if (!response.ok) throw new Error(t('无法加载 PDF'));
    const arrayBuffer = await response.arrayBuffer();
    return loadPdf(arrayBuffer);
}

async function createPDFBatchBlob(sourcePdf, startPage, endPage) {
    return new Blob([await createPDFBatchBytes(sourcePdf, startPage, endPage)], { type: 'application/pdf' });
}

async function createPDFBatchBytes(sourcePdf, startPage, endPage) {
    const batchPdf = await PDFLib.PDFDocument.create();
    const pageIndices = [];
    for (let i = startPage - 1; i <= endPage - 1; i++) {
        pageIndices.push(i);
    }
    const copiedPages = await batchPdf.copyPages(sourcePdf, pageIndices);
    copiedPages.forEach((page) => batchPdf.addPage(page));
    return batchPdf.save();
}

async function ensureBatchPayload(task, batch) {
    if (batch.payloadBlob || batch.payloadDataUrl) return;
    if (batch.fileType === 1) {
        batch.payloadBlob = await getTaskSourceBlob(task, task.mimeType || 'image/jpeg');
        return;
    }

    if (batch.fileType !== 0) {
        throw new Error(t('无法重建当前批次的解析 payload'));
    }
    if (!(task.sourceDataUrl || task.sourceUrl)) {
        throw new Error(t('缺少源 PDF，无法继续解析'));
    }
    if ((task.pageCount || 1) === 1) {
        batch.payloadBlob = await getTaskSourceBlob(task, 'application/pdf');
        return;
    }

    // Always use server-side page extraction for PDF batches.
    // This avoids loading the entire PDF into browser memory (PDF-lib).
    if (task.sourceUrl) {
        batch.payloadBlob = await fetchPdfBatchBlob(task, batch.startPage, batch.endPage);
        return;
    }

    // Fallback: only use client-side PDF-lib for small PDFs with sourceDataUrl
    if (task.sourceDataUrl && (task.size || 0) <= chunkedUploadThreshold) {
        let sourcePdf = sourcePdfCache.get(task.id);
        if (!sourcePdf) {
            sourcePdf = await PDFLib.PDFDocument.load(await getTaskSourceBytes(task));
            sourcePdfCache.set(task.id, sourcePdf);
        }
        batch.payloadBlob = await createPDFBatchBlob(sourcePdf, batch.startPage, batch.endPage);
        return;
    }

    throw new Error(t('缺少源 PDF，无法继续解析'));
}

function releaseBatchPayload(batch) {
    delete batch.payloadBlob;
    delete batch.payloadDataUrl;
}

async function fetchPdfBatchBlob(task, startPage, endPage) {
    const url = `${API_BASE}/tasks/${encodeURIComponent(task.id)}/source/pages?start_page=${startPage}&end_page=${endPage}`;
    const response = await apiFetch(url);
    if (!response.ok) {
        throw new Error(t('读取 PDF 分页失败：{detail}', { detail: await responseErrorText(response) }));
    }
    return response.blob();
}

async function getTaskSourceBytes(task) {
    if (sourceBytesCache.has(task.id)) {
        return sourceBytesCache.get(task.id);
    }
    if (task.sourceDataUrl) {
        const bytes = dataUrlToUint8Array(task.sourceDataUrl);
        sourceBytesCache.set(task.id, bytes);
        return bytes;
    }
    if (!task.sourceUrl) {
        throw new Error(t('缺少源文件，无法继续解析'));
    }
    const response = await apiFetch(task.sourceUrl);
    if (!response.ok) {
        throw new Error(t('读取源文件失败：{detail}', { detail: await responseErrorText(response) }));
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    sourceBytesCache.set(task.id, bytes);
    return bytes;
}

async function getTaskSourceBlob(task, mimeType) {
    if (task.sourceDataUrl) {
        return dataUrlToBlob(task.sourceDataUrl);
    }
    if (!task.sourceUrl) {
        throw new Error(t('缺少源文件，无法继续解析'));
    }
    const response = await apiFetch(task.sourceUrl);
    if (!response.ok) {
        throw new Error(t('读取源文件失败：{detail}', { detail: await responseErrorText(response) }));
    }
    const blob = await response.blob();
    if (mimeType && blob.type !== mimeType) {
        return new Blob([blob], { type: mimeType });
    }
    return blob;
}

function normalizeOCRMarkdown(markdown) {
    return String(markdown)
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n')
        .replace(/\\r\\n/g, '\n')
        .replace(/\\n/g, '\n');
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderMarkdownHtml(markdown) {
    const { text, math } = stashMathSegments(markdown);
    if (!marked._pandocrConfigured) {
        marked.setOptions({
            breaks: true,
            gfm: true,
        });
        marked._pandocrConfigured = true;
    }
    let html = marked.parse(text);
    math.forEach((value, index) => {
        html = html.split(mathToken(index)).join(value);
    });
    if (!window.DOMPurify) return html;
    // For large documents (>50KB), use a faster sanitize pass.
    // OCR results come from our own server — not user-generated XSS vectors.
    // Math blocks are already stashed/replaced with safe tokens.
    if (html.length > 50000) {
        return DOMPurify.sanitize(html, {
            ADD_TAGS: ['math', 'semantics', 'mrow', 'mi', 'mo', 'mn', 'msup', 'msub', 'mfrac', 'msqrt', 'mroot', 'munder', 'mover', 'munderover', 'mtable', 'mtr', 'mtd', 'mtext', 'mspace', 'mpadded', 'menclose', 'mfenced', 'mstyle', 'annotation'],
            ADD_ATTR: ['display', 'mathvariant', 'encoding', 'stretchy', 'lspace', 'rspace', 'minsize', 'maxsize', 'movablelimits', 'symmetric', 'largeop', 'accent', 'linethickness', 'scriptlevel', 'displaystyle', 'xmlns'],
            ALLOW_DATA_ATTR: true,
            // Allow data:image src attributes for inline images
            ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|cid|xmpp|data):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i,
        });
    }
    return DOMPurify.sanitize(html, {
        ADD_TAGS: ['math', 'semantics', 'mrow', 'mi', 'mo', 'mn', 'msup', 'msub', 'mfrac', 'msqrt', 'mroot', 'munder', 'mover', 'munderover', 'mtable', 'mtr', 'mtd', 'mtext', 'mspace', 'mpadded', 'menclose', 'mfenced', 'mstyle', 'annotation'],
        ADD_ATTR: ['display', 'mathvariant', 'encoding', 'stretchy', 'lspace', 'rspace', 'minsize', 'maxsize', 'movablelimits', 'symmetric', 'largeop', 'accent', 'linethickness', 'scriptlevel', 'displaystyle', 'xmlns'],
        ALLOW_DATA_ATTR: true,
        ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto|tel|callto|cid|xmpp|data):|[^a-z]|[a-z+.\-]+(?:[^a-z+.\-:]|$))/i,
    });
}

function stashMathSegments(markdown) {
    const math = [];
    const store = (value) => {
        const token = mathToken(math.length);
        math.push(value);
        return token;
    };
    let text = markdown.replace(/\$\$[\s\S]*?\$\$/g, store);
    text = text.replace(/\\\[[\s\S]*?\\\]/g, store);
    text = text.replace(/\\\([\s\S]*?\\\)/g, store);
    text = text.replace(/\$([^$\n]|\\.)+?\$/g, store);
    return { text, math };
}

function mathToken(index) {
    return `PANDOCRMATHTOKEN${index}X`;
}

function prepareMarkdownForRender(markdown) {
    return normalizeOCRMarkdown(markdown);
}

function renderMathWhenReady(container, retries = 20) {
    if (!window.renderMathInElement) {
        if (retries > 0) setTimeout(() => renderMathWhenReady(container, retries - 1), 150);
        return;
    }
    try {
        renderMathInElement(container, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '\\[', right: '\\]', display: true },
                { left: '\\(', right: '\\)', display: false },
                { left: '$', right: '$', display: false }
            ],
            // Only render elements not already processed by KaTeX
            preProcess: (el) => {
                if (el.querySelector && el.querySelector('.katex')) return null;
                return el;
            },
            ignoredTags: ['script', 'noscript', 'style', 'textarea', 'code', 'pre', 'katex'],
            throwOnError: false,
            strict: false,
            trust: true,
        });
    } catch (e) {
        console.warn('KaTeX render error:', e);
    }
}

function renderOfficialLayoutResult(task) {
    const blocks = collectOfficialRenderBlocks(task);
    if (!blocks.length) return { rendered: false, changed: false, mathRoots: [] };

    const context = officialLayoutRenderContext(task);
    const expectedKeys = blocks.map(officialLayoutBlockKey);
    const children = Array.from(els.markdownView.children);
    const existingBlocks = children.filter((element) => element.classList.contains('official-layout-block'));
    const hasOnlyOfficialBlocks = children.length === existingBlocks.length;
    const existingKeys = existingBlocks.map((element) => element.dataset.blockKey || '');
    const canAppend = hasOnlyOfficialBlocks
        && renderedOfficialLayoutContext === context
        && existingKeys.length <= expectedKeys.length
        && existingKeys.every((key, index) => key === expectedKeys[index]);

    const appendedElements = [];
    const fullRebuild = !canAppend;
    if (fullRebuild) {
        clearSourceHighlight();
        clearSourceHotspots();
        els.markdownView.replaceChildren();
        renderedOfficialLayoutContext = context;
    }

    const startIndex = canAppend ? existingKeys.length : 0;
    if (startIndex === expectedKeys.length) {
        return { rendered: true, changed: false, mathRoots: [] };
    }

    const fragment = document.createDocumentFragment();
    blocks.slice(startIndex).forEach((block, offset) => {
        const blockIndex = startIndex + offset;
        const element = createOfficialLayoutBlockElement(block, expectedKeys[blockIndex], task);
        appendedElements.push(element);
        fragment.appendChild(element);
    });
    els.markdownView.appendChild(fragment);
    renderedOfficialLayoutContext = context;

    return {
        rendered: true,
        changed: true,
        mathRoots: fullRebuild ? [els.markdownView] : appendedElements
    };
}

function createOfficialLayoutBlockElement(block, blockKey, task) {
    const element = document.createElement('section');
    element.className = 'layout-linked-block official-layout-block';
    element.dataset.blockKey = blockKey;
    element.dataset.layoutLabel = layoutLabelText(block.label);
    element.dataset.page = String(block.page);
    element.dataset.blockIndex = String(block.blockIndex);

    const content = rewriteBlockImageSources(block.content || fallbackBlockContent(block), block.pageResult, task);
    element.innerHTML = renderMarkdownHtml(content);

    addSourceHotspot(block, element);
    bindLinkedBlockEvents(element, block);
    return element;
}

function officialLayoutRenderContext(task) {
    return [
        task?.id || '',
        sourceRenderToken,
        currentZoom,
        currentLanguage,
        els.ignoreNumberSwitch.checked,
        els.ignoreHeaderSwitch.checked,
        els.ignoreFooterSwitch.checked
    ].join(':');
}

function officialLayoutBlockKey(block) {
    return [
        block.page,
        block.blockIndex,
        String(block.label || '').toLowerCase(),
        Array.isArray(block.bbox) ? block.bbox.join(',') : '',
        shortHash(block.content || fallbackBlockContent(block))
    ].join(':');
}

function shortHash(value) {
    const text = String(value || '');
    let hash = 5381;
    for (let index = 0; index < text.length; index += 1) {
        hash = ((hash << 5) + hash) ^ text.charCodeAt(index);
    }
    return (hash >>> 0).toString(36);
}

function collectOfficialRenderBlocks(task) {
    const blocks = [];
    if (!Array.isArray(task?.ocrResults)) return blocks;

    task.ocrResults.forEach((pageResult, pageIndex) => {
        const pruned = pageResult?.prunedResult || pageResult;
        const pageWidth = Number(pruned?.width);
        const pageHeight = Number(pruned?.height);
        const parsingBlocks = Array.isArray(pruned?.parsing_res_list) ? pruned.parsing_res_list : [];

        parsingBlocks.forEach((sourceBlock, blockIndex) => {
            const bbox = sourceBlock.block_bbox || sourceBlock.coordinate || sourceBlock.bbox;
            const label = sourceBlock.block_label || sourceBlock.label || '';
            const content = sourceBlock.block_content ?? sourceBlock.text ?? sourceBlock.content ?? '';
            if (!Array.isArray(bbox) || bbox.length < 4 || !pageWidth || !pageHeight) return;
            if (isIgnoredLayoutLabel(label)) return;
            if (!String(content || '').trim() && !isVisualLayoutLabel(label)) return;

            blocks.push({
                page: pageIndex + 1,
                blockIndex,
                label,
                bbox,
                pageWidth,
                pageHeight,
                content: String(content || ''),
                pageResult,
                sourceBlock
            });
        });
    });

    return blocks;
}

function isIgnoredLayoutLabel(label) {
    const normalized = String(label || '').toLowerCase();
    const ignored = new Set(['footnote', 'aside_text']);
    if (els.ignoreNumberSwitch.checked) ignored.add('number');
    if (els.ignoreHeaderSwitch.checked) {
        ignored.add('header');
        ignored.add('header_image');
    }
    if (els.ignoreFooterSwitch.checked) {
        ignored.add('footer');
        ignored.add('footer_image');
    }
    return ignored.has(normalized);
}

function isVisualLayoutLabel(label) {
    return ['image', 'chart', 'table', 'algorithm'].includes(String(label || '').toLowerCase());
}

function fallbackBlockContent(block) {
    const label = layoutLabelText(block.label);
    return label ? `<div class="layout-block-placeholder">${escapeHtml(label)}</div>` : '';
}

function rewriteBlockImageSources(content, pageResult, task) {
    let output = normalizeOCRMarkdown(String(content || ''));
    const imageMaps = [
        pageResult?.markdown?.images,
        pageResult?.prunedResult?.markdown?.images,
        task?.images
    ];

    imageMaps.forEach((images) => {
        if (!images || typeof images !== 'object') return;
        Object.entries(images).forEach(([path, value]) => {
            if (!path || value == null) return;
            output = output.split(path).join(imageValueToSrc(value));
        });
    });

    return output;
}

function imageValueToSrc(value) {
    const text = String(value || '');
    if (/^(https?:|data:|blob:)/i.test(text)) return text;
    if (/^ocr_images\//i.test(text)) return text;
    return `data:image/jpeg;base64,${text}`;
}

function bindLinkedBlockEvents(element, block) {
    const preview = () => activateLinkedBlock(element, block);
    const locate = () => activateLinkedBlock(element, block, { scrollSource: true });
    const deactivate = () => deactivateLinkedBlocks();
    element.addEventListener('mouseenter', preview);
    element.addEventListener('mouseover', preview);
    element.addEventListener('pointerenter', preview);
    element.addEventListener('focusin', preview);
    element.addEventListener('click', locate);
    element.addEventListener('mouseleave', deactivate);
    element.addEventListener('pointerleave', deactivate);
    element.addEventListener('focusout', deactivate);
}

function linkMarkdownToSourceBlocks(task) {
    clearSourceHighlight();
    clearSourceHotspots();
    if (!task?.ocrResults?.length) return;

    const blocks = collectLayoutBlocks(task);
    if (!blocks.length) return;

    const elements = collectMarkdownBlockElements(els.markdownView);
    let cursor = 0;

    elements.forEach((element) => {
        const isImageBlock = isMarkdownImageBlock(element);
        const text = normalizeMatchText(element.innerText || element.textContent || '');
        if (!isImageBlock && text.length < 2) return;

        const match = isImageBlock
            ? findNextLayoutBlockByLabel(blocks, cursor, ['image', 'chart', 'table'])
            : isAlgorithmText(element.innerText || element.textContent || '')
                ? findNextLayoutBlockByLabel(blocks, cursor, ['algorithm'])
            : isFigureTitleText(element.innerText || element.textContent || '')
                ? findNextLayoutBlockByLabel(blocks, cursor, ['figure_title'])
            : findBestLayoutBlock(text, blocks, cursor);
        if (!match) return;

        cursor = match.index;
        element.classList.add('layout-linked-block');
        element.dataset.layoutLabel = layoutLabelText(match.block.label);
        addSourceHotspot(match.block, element);
        const preview = () => activateLinkedBlock(element, match.block);
        const locate = () => activateLinkedBlock(element, match.block, { scrollSource: true });
        const deactivate = () => deactivateLinkedBlocks();
        element.addEventListener('mouseenter', preview);
        element.addEventListener('mouseover', preview);
        element.addEventListener('pointerenter', preview);
        element.addEventListener('focusin', preview);
        element.addEventListener('click', locate);
        element.addEventListener('mouseleave', deactivate);
        element.addEventListener('pointerleave', deactivate);
        element.addEventListener('focusout', deactivate);
    });
}

function collectLayoutBlocks(task) {
    const blocks = [];
    task.ocrResults.forEach((pageResult, pageIndex) => {
        const pruned = pageResult?.prunedResult || pageResult;
        const pageWidth = Number(pruned?.width);
        const pageHeight = Number(pruned?.height);
        const parsingBlocks = Array.isArray(pruned?.parsing_res_list) ? pruned.parsing_res_list : [];

        parsingBlocks.forEach((block, blockIndex) => {
            const bbox = block.block_bbox || block.coordinate || block.bbox;
            const content = block.block_content || block.text || block.content || '';
            const label = block.block_label || block.label || '';
            if (!Array.isArray(bbox) || bbox.length < 4 || !pageWidth || !pageHeight) return;
            if (!content && !['image', 'chart', 'table'].includes(label)) return;
            blocks.push({
                page: pageIndex + 1,
                order: Number(block.block_order ?? blockIndex),
                label,
                bbox,
                pageWidth,
                pageHeight,
                text: normalizeMatchText(content || label)
            });
        });
    });
    return blocks.sort((a, b) => (a.page - b.page) || (a.order - b.order));
}

function collectMarkdownBlockElements(container) {
    const selector = 'h1,h2,h3,h4,h5,h6,p,li,table,pre,blockquote,div,img';
    const seen = new Set();
    const elements = [];

    Array.from(container.querySelectorAll(selector)).forEach((element) => {
        if (element.closest('.empty-result')) return false;
        if (element.parentElement?.closest('li,table,pre,blockquote')) return false;
        if (element.tagName === 'DIV' && !isMarkdownImageBlock(element) && !isFigureTitleText(element.innerText || element.textContent || '')) {
            return false;
        }

        const imageHost = element.tagName === 'IMG' ? element.closest('p,div') || element : element;
        const target = ['P', 'DIV'].includes(imageHost.tagName) && imageHost.querySelector('img') ? imageHost : element;
        if (seen.has(target)) return false;

        const hasText = Boolean((target.innerText || target.textContent || '').trim());
        const hasImage = isMarkdownImageBlock(target);
        if (!hasText && !hasImage) return false;

        seen.add(target);
        elements.push(target);
        return true;
    });

    return elements;
}

function isMarkdownImageBlock(element) {
    return element?.tagName === 'IMG' || Boolean(element?.querySelector?.('img'));
}

function isFigureTitleText(value) {
    return /^Figure\s+\d+\s*[:：]/i.test(String(value || '').trim());
}

function isAlgorithmText(value) {
    return /^Algorithm\s+\d+\s*[:：]/i.test(String(value || '').trim());
}

function findBestLayoutBlock(text, blocks, cursor) {
    let best = null;
    const start = Math.max(0, cursor - 1);
    const end = Math.min(blocks.length, cursor + 18);

    for (let index = start; index < end; index += 1) {
        const score = matchScore(text, blocks[index].text);
        if (score < 0.55) continue;
        if (!best || score > best.score) best = { index, block: blocks[index], score };
        if (score >= 0.92) break;
    }

    return best;
}

function findNextLayoutBlockByLabel(blocks, cursor, labels) {
    const wanted = new Set(labels);
    const start = Math.max(0, cursor - 1);
    for (let index = start; index < blocks.length; index += 1) {
        if (wanted.has(String(blocks[index].label || '').toLowerCase())) {
            return { index, block: blocks[index], score: 1 };
        }
    }
    return null;
}

function matchScore(elementText, blockText) {
    if (!elementText || !blockText) return 0;
    if (elementText === blockText) return 1;
    if (blockText.includes(elementText)) return Math.min(0.98, elementText.length / Math.max(blockText.length, 1) + 0.62);
    if (elementText.includes(blockText)) return Math.min(0.96, blockText.length / Math.max(elementText.length, 1) + 0.55);

    const words = elementText.split(' ').filter((word) => word.length > 2);
    if (!words.length) return 0;
    const hitCount = words.filter((word) => blockText.includes(word)).length;
    return hitCount / words.length;
}

function normalizeMatchText(value) {
    return String(value)
        .replace(/\$+/g, ' ')
        .replace(/\\[a-zA-Z]+/g, ' ')
        .replace(/[{}^_`~|()[\]<>#*_.,:;'"!?，。；：！？、]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .toLowerCase();
}

function showSourceHighlight(block) {
    clearSourceHighlight();
    const surface = sourcePageSurface(block.page);
    if (!surface) return;

    const box = document.createElement('div');
    box.className = 'source-highlight-box';
    positionSourceOverlayBox(box, block, surface.element);
    const label = document.createElement('span');
    label.className = 'source-highlight-label';
    label.textContent = layoutLabelText(block.label);
    box.appendChild(label);
    surface.layer.appendChild(box);

}

function showPPOCRSourceHighlight(line) {
    if (!line?.box || !line.pageWidth || !line.pageHeight) return;
    clearSourceHighlight();
    const surface = sourcePageSurface(line.sourcePage);
    if (!surface) return;

    const box = document.createElement('div');
    box.className = 'source-highlight-box source-highlight-box-ocr';
    positionSourceOverlayBox(box, {
        bbox: line.box,
        pageWidth: line.pageWidth,
        pageHeight: line.pageHeight
    }, surface.element);
    surface.layer.appendChild(box);
}

function clearSourceHighlight() {
    els.sourceViewer.querySelectorAll('.source-highlight-box').forEach((box) => box.remove());
}

function addPPOCRSourceHotspot(line, markdownElement, toolbar) {
    if (!line?.box || !line.pageWidth || !line.pageHeight) return;
    const surface = sourcePageSurface(line.sourcePage);
    if (!surface) return;

    const hotspot = document.createElement('button');
    hotspot.type = 'button';
    hotspot.className = 'source-link-hotspot source-ocr-hotspot';
    hotspot.setAttribute('aria-label', line.text || 'OCR');
    hotspot.dataset.page = String(line.sourcePage || '');
    hotspot.dataset.pageResultIndex = String(line.pageResultIndex ?? '');
    hotspot.dataset.lineIndex = String(line.index ?? '');
    positionSourceOverlayBox(hotspot, {
        bbox: line.box,
        pageWidth: line.pageWidth,
        pageHeight: line.pageHeight
    }, surface.element);

    const preview = () => {
        activatePPOCRLine(markdownElement, toolbar, line, { scrollSource: false });
    };
    const locate = () => {
        scrollElementIntoContainer(markdownElement, els.markdownView, 'smooth');
        activatePPOCRLine(markdownElement, toolbar, line, { scrollSource: false });
    };
    hotspot.addEventListener('mouseenter', preview);
    hotspot.addEventListener('mouseover', preview);
    hotspot.addEventListener('pointerenter', preview);
    hotspot.addEventListener('focusin', preview);
    hotspot.addEventListener('click', locate);
    surface.layer.appendChild(hotspot);
}

function addSourceHotspot(block, markdownElement) {
    const surface = sourcePageSurface(block.page);
    if (!surface) return;

    const hotspot = document.createElement('button');
    hotspot.type = 'button';
    hotspot.className = 'source-link-hotspot';
    hotspot.setAttribute('aria-label', layoutLabelText(block.label));
    positionSourceOverlayBox(hotspot, block, surface.element);

    const preview = () => activateLinkedBlock(markdownElement, block);
    const locate = () => activateLinkedBlock(markdownElement, block, { scrollMarkdown: true });
    const deactivate = () => deactivateLinkedBlocks();
    hotspot.addEventListener('mouseenter', preview);
    hotspot.addEventListener('mouseover', preview);
    hotspot.addEventListener('pointerenter', preview);
    hotspot.addEventListener('focusin', preview);
    hotspot.addEventListener('click', locate);
    hotspot.addEventListener('mouseleave', deactivate);
    hotspot.addEventListener('pointerleave', deactivate);
    hotspot.addEventListener('focusout', deactivate);

    surface.layer.appendChild(hotspot);
}

function sourcePageSurface(pageNumber = 1) {
    const page = String(pageNumber || 1);
    const pdfPage = els.sourceViewer.querySelector(`.pdf-page-wrap[data-page="${page}"]`);
    if (pdfPage) {
        const element = pdfPage.querySelector('canvas');
        const layer = pdfPage.querySelector('.pdf-highlight-layer');
        if (element && layer) return { container: pdfPage, element, layer };
    }

    const imagePage = els.sourceViewer.querySelector(`.source-image-wrap[data-page="${page}"]`);
    if (imagePage) {
        const element = imagePage.querySelector('.source-image');
        const layer = imagePage.querySelector('.pdf-highlight-layer');
        if (element && layer) return { container: imagePage, element, layer };
    }

    return null;
}

function positionSourceOverlayBox(element, block, sourceElement) {
    const [x1, y1, x2, y2] = block.bbox.map(Number);
    const sourceWidth = sourceElement.clientWidth || sourceElement.width || sourceElement.naturalWidth;
    const sourceHeight = sourceElement.clientHeight || sourceElement.height || sourceElement.naturalHeight;
    element.style.left = `${(x1 / block.pageWidth) * sourceWidth}px`;
    element.style.top = `${(y1 / block.pageHeight) * sourceHeight}px`;
    element.style.width = `${((x2 - x1) / block.pageWidth) * sourceWidth}px`;
    element.style.height = `${((y2 - y1) / block.pageHeight) * sourceHeight}px`;
}

function activateLinkedBlock(markdownElement, block, { scrollMarkdown = false, scrollSource = false } = {}) {
    deactivateLinkedBlocks();
    markdownElement.classList.add('layout-linked-block-active');
    showSourceHighlight(block);
    if (scrollMarkdown && !isElementMostlyVisible(markdownElement, els.markdownView)) {
        scrollElementIntoContainer(markdownElement, els.markdownView, 'smooth');
    }
    if (scrollSource) {
        const sourceSurface = sourcePageSurface(block.page);
        if (sourceSurface?.container && !isElementMostlyVisible(sourceSurface.container, els.sourceViewer)) {
            scrollPdfPageIntoView(block.page, 'smooth');
        }
    }
}

function deactivateLinkedBlocks() {
    els.markdownView.querySelectorAll('.layout-linked-block-active').forEach((element) => {
        element.classList.remove('layout-linked-block-active');
    });
    clearSourceHighlight();
}

function clearSourceHotspots() {
    els.sourceViewer.querySelectorAll('.source-link-hotspot').forEach((hotspot) => hotspot.remove());
}

function isElementMostlyVisible(element, container) {
    const elementRect = element.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    return elementRect.top >= containerRect.top - 20 && elementRect.top <= containerRect.bottom - 80;
}

function scrollElementIntoContainer(element, container, behavior = 'smooth') {
    const elementRect = element.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();
    const offset = elementRect.top - containerRect.top;
    const centeredTop = container.scrollTop + offset - (container.clientHeight / 2) + (elementRect.height / 2);
    container.scrollTo({ top: Math.max(centeredTop, 0), behavior });
}

function layoutLabelText(label) {
    const normalized = String(label || '').trim().toLowerCase();
    const labels = {
        abstract: '摘要',
        doc_title: '标题',
        title: '标题',
        paragraph_title: '段落标题',
        text: '文本',
        image: '图片',
        figure_title: '图表标题',
        table: '表格',
        formula: '公式',
        display_formula: '行间公式',
        formula_number: '公式编号',
        footer: '页脚',
        header: '页眉',
        number: '页码',
        reference: '参考文献',
        reference_content: '参考文献',
        footnote: '脚注',
        algorithm: '算法',
        chart: '图表'
    };
    return labels[normalized] ? t(labels[normalized]) : (normalized || t('版面块'));
}

function prepareBatchResult(result, batchId) {
    let markdown = normalizeOCRMarkdown(result.markdown || '');
    const images = {};
    Object.entries(result.images || {}).forEach(([path, base64]) => {
        const safePath = safeImagePath(batchId, path);
        markdown = markdown.split(path).join(safePath);
        images[safePath] = base64;
    });
    return { markdown, images };
}

function safeImagePath(batchId, path) {
    const filename = String(path || '').split('/').pop() || 'image';
    return `ocr_images/${batchId}_${filename}`;
}

function compactOCRJsonResult(pageResult, batchOrId, pageIndex = 0) {
    const batch = typeof batchOrId === 'object' ? batchOrId : null;
    const batchId = batch?.id || batchOrId;
    const compact = stripLargeOCRFields(pageResult);
    if (batch && compact?.parser === 'pp-ocrv6') {
        compact.sourcePage = Number(batch.startPage || 1) + pageIndex;
        compact.batchId = batch.id;
    }
    rewriteMarkdownImageMaps(compact, batchId);
    return compact;
}

function stripLargeOCRFields(value) {
    if (Array.isArray(value)) {
        return value.map(stripLargeOCRFields);
    }
    if (!value || typeof value !== 'object') {
        return value;
    }

    const output = {};
    Object.entries(value).forEach(([key, nestedValue]) => {
        if (key === 'inputImage' || key === 'outputImages') return;
        output[key] = stripLargeOCRFields(nestedValue);
    });
    return output;
}

function rewriteMarkdownImageMaps(value, batchId) {
    if (!value || typeof value !== 'object') return;
    if (value.images && typeof value.images === 'object' && typeof value.text === 'string') {
        value.images = Object.fromEntries(
            Object.keys(value.images).map((path) => [path, safeImagePath(batchId, path)])
        );
    }
    Object.values(value).forEach((nestedValue) => rewriteMarkdownImageMaps(nestedValue, batchId));
}

function normalizeOCRJsonResults(result) {
    if (Array.isArray(result.pages)) {
        return result.pages;
    }
    if (Array.isArray(result.layoutParsingResults)) {
        return result.layoutParsingResults;
    }
    if (Array.isArray(result.results)) {
        return result.results;
    }
    return [{
        markdown: {
            text: result.markdown || '',
            images: result.images || {}
        }
    }];
}

function toOfficialJson(task) {
    if (Array.isArray(task.ocrResults) && task.ocrResults.length > 0) {
        return task.ocrResults;
    }

    return [];
}

function readAsDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
    });
}

async function waitForImageReady(img) {
    if (!img || (img.complete && img.naturalWidth > 0)) return;
    if (typeof img.decode === 'function') {
        try {
            await img.decode();
            return;
        } catch (error) {
            if (img.complete) return;
        }
    }
    await new Promise((resolve) => {
        img.addEventListener('load', resolve, { once: true });
        img.addEventListener('error', resolve, { once: true });
    });
}

function dataUrlToUint8Array(dataUrl) {
    const base64 = dataUrl.split('base64,')[1];
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
}

function dataUrlToBlob(dataUrl) {
    const mimeMatch = String(dataUrl).match(/^data:([^;,]+)[;,]/i);
    return new Blob([dataUrlToUint8Array(dataUrl)], {
        type: mimeMatch?.[1] || 'application/octet-stream'
    });
}

function base64ToBytes(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function emptyDropZoneHtml() {
    return `
        <div class="drop-zone" id="drop-zone">
            <svg viewBox="0 0 24 24"><path d="M12 3v12M7 8l5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/></svg>
            <h3>${escapeHtml(t('拖拽文件到这里'))}</h3>
            <p>${escapeHtml(t('支持 PDF、图片、PPT/PPTX、DOC/DOCX；PDF 会逐页解析。'))}</p>
            <button class="primary-button" id="browse-btn">${escapeHtml(t('选择文件'))}</button>
        </div>
    `;
}

function taskIcon(task) {
    if (task.sourceKind === 'image') {
        return '<svg viewBox="0 0 24 24"><path d="M4 5h16v14H4z"/><path d="m4 16 5-5 4 4 2-2 5 5"/><circle cx="16" cy="9" r="1.5"/></svg>';
    }
    return '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8M8 17h6"/></svg>';
}

function statusText(task) {
    const donePages = task.batches?.filter((batch) => batch.status === 'completed').reduce((sum, batch) => sum + batch.pageCount, 0) || task.completedPages || 0;
    if (task.status === 'completed') return t('完成');
    if (isTaskActivelyProcessing(task)) return t('{done}/{total} 解析中', { done: donePages, total: task.pageCount || 1 });
    if (shouldResumeTask(task)) return t('{done}/{total} 可继续', { done: donePages, total: task.pageCount || 1 });
    if (task.status === 'error') return t('失败');
    return t('待解析');
}

function resultPaneTitle(task) {
    if (task.status === 'completed') return t('解析结果');
    if (task.status === 'processing') return t('解析中');
    if (shouldResumeTask(task)) return t('解析中断');
    if (task.status === 'error') return t('解析失败');
    return t('待解析');
}

function emptyResultText(task) {
    if (task.status === 'processing') return t('正在解析，结果会实时追加到这里。');
    if (shouldResumeTask(task)) return t('上次解析中断，点击“继续解析”从未完成页面恢复。');
    if (task.status === 'error') return t('解析失败：{detail}', { detail: task.error || t('未知错误') });
    return t('点击“开始解析”生成 Markdown 或 JSON 结果。');
}

function sourceLabel(task) {
    if (task.sourceKind === 'office') return t('Office 已转 PDF · {name}', { name: task.originalName });
    if (task.sourceKind === 'image') return t('图片');
    return 'PDF';
}

function taskSourceMeta(task) {
    return `${sourceLabel(task)} · ${formatSize(task.size)} · ${formatPageCount(task.pageCount || 1)}`;
}

function formatPageCount(count) {
    if (currentLanguage === 'en') {
        return `${count} ${Number(count) === 1 ? 'page' : 'pages'}`;
    }
    return t('{count} 页', { count });
}

function getExtension(filename) {
    return filename.split('.').pop().toLowerCase();
}

function initPdfBatchSizeSetting() {
    if (!els.pdfBatchSizeInput) return;
    syncPdfBatchSizeSetting();
}

function syncPdfBatchSizeSetting() {
    if (!els.pdfBatchSizeInput) return DEFAULT_PDF_BATCH_SIZE;
    const batchSize = getConfiguredPdfBatchSize();
    els.pdfBatchSizeInput.value = String(batchSize);
    localStorage.setItem(PDF_BATCH_SIZE_STORAGE_KEY, String(batchSize));
    return batchSize;
}

function handlePdfBatchSizeInput(event) {
    const input = event?.target || els.pdfBatchSizeInput;
    if (!input) return;
    const rawValue = input.value;
    if (rawValue === '') return;
    const parsed = Number.parseInt(rawValue, 10);
    if (!Number.isFinite(parsed)) return;
    const batchSize = clampPdfBatchSize(parsed);
    if (String(parsed) !== String(batchSize)) {
        input.value = String(batchSize);
    }
    localStorage.setItem(PDF_BATCH_SIZE_STORAGE_KEY, String(batchSize));
}

function getConfiguredPdfBatchSize() {
    // Check model-specific batch size inputs first, then fall back to the shared one
    const modelId = selectedModelId;
    const modelInput = document.getElementById(`pdf-batch-size-input-${modelId}`);
    const rawValue = modelInput?.value || els.pdfBatchSizeInput?.value || localStorage.getItem(PDF_BATCH_SIZE_STORAGE_KEY);
    return clampPdfBatchSize(rawValue);
}

function clampPdfBatchSize(value) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed)) return DEFAULT_PDF_BATCH_SIZE;
    return Math.min(MAX_PDF_BATCH_SIZE, Math.max(1, parsed));
}

function formatDate(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const sameYear = date.getFullYear() === now.getFullYear();
    return date.toLocaleString(languageLocale(), {
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        ...(sameYear ? {} : { year: 'numeric' })
    });
}

function formatSize(bytes = 0) {
    if (!bytes) return t('未知大小');
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)}MB`;
}

function formatPageLabel(startPage, endPage = startPage) {
    return startPage === endPage
        ? t('第 {start} 页', { start: startPage })
        : t('第 {start}-{end} 页', { start: startPage, end: endPage });
}

function safeDownloadName(name, ext) {
    return `${name.replace(/\.[^.]+$/, '').replace(/[\\/:*?"<>|]/g, '_')}.${ext}`;
}

function loadPdf(source) {
    const opts = typeof source === 'string'
        ? { url: source }
        : { data: source };
    opts.cMapUrl = '/static/vendor/pdfjs/cmaps/';
    opts.cMapPacked = true;
    opts.standardFontDataUrl = '/static/vendor/pdfjs/standard_fonts/';
    opts.useSystemFonts = true;
    return pdfjsLib.getDocument(opts).promise;
}

// Streaming variant for large remote PDFs — onProgress fires as the
// PDF.js worker pulls bytes over the network, so we can show a real
// progress bar. Throwing from onProgress aborts the load (used by the
// Cancel button).
function loadPdfWithProgress(url, onProgress) {
    return new Promise((resolve, reject) => {
        const loadingTask = pdfjsLib.getDocument({
            url,
            cMapUrl: '/static/vendor/pdfjs/cmaps/',
            cMapPacked: true,
            standardFontDataUrl: '/static/vendor/pdfjs/standard_fonts/',
            useSystemFonts: true,
        });
        loadingTask.onProgress = onProgress || (() => {});
        loadingTask.promise.then(resolve, reject);
    });
}

function createId() {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

/* MinerU: click-to-locate & editable text */
function bindMineruContentClicks(task) {
    const contentList = task.contentList || [];
    const elements = els.markdownView.querySelectorAll('h1,h2,h3,h4,h5,h6,p,li,td,th,pre,blockquote');
    let contentIdx = 0;

    elements.forEach((el) => {
        const text = el.textContent.trim();
        if (!text) return;

        let matchedItem = null;
        for (let i = contentIdx; i < contentList.length; i++) {
            const item = contentList[i];
            if (item.type === 'text' && item.text && text.includes(item.text.trim())) {
                matchedItem = item;
                contentIdx = i;
                break;
            }
        }

        if (matchedItem && matchedItem.bbox && matchedItem.page_idx != null) {
            el.dataset.mineruPage = String(matchedItem.page_idx);
            el.dataset.mineruBbox = JSON.stringify(matchedItem.bbox);
            el.classList.add('mineru-locatable');
            el.title = t('点击定位到原文，双击编辑');

            el.addEventListener('click', (e) => {
                if (el.isContentEditable) return;
                e.preventDefault();
                highlightMineruSource(matchedItem.page_idx, matchedItem.bbox);
            });

            el.addEventListener('dblclick', (e) => {
                e.preventDefault();
                el.contentEditable = 'true';
                el.focus();
                el.classList.add('mineru-editing');
            });

            el.addEventListener('blur', () => {
                el.contentEditable = 'false';
                el.classList.remove('mineru-editing');
                const newText = el.textContent.trim();
                if (newText !== text && matchedItem) {
                    matchedItem.text = newText;
                    updateMineruMarkdown(task);
                }
            });

            el.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    el.blur();
                } else if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    el.blur();
                }
            });
        }
    });
}

function highlightMineruSource(pageIdx, bbox) {
    if (pageIdx == null || !Array.isArray(bbox) || bbox.length < 4) return;

    const page = pageIdx + 1;
    scrollPdfPageIntoView(page, 'smooth');

    clearSourceHighlight();

    requestAnimationFrame(() => {
        const surface = sourcePageSurface(page);
        if (!surface) return;
        const { element, layer } = surface;

        const [x0, y0, x1, y1] = bbox.map(Number);
        const sw = element.clientWidth || element.width || 1;
        const sh = element.clientHeight || element.height || 1;

        const highlight = document.createElement('div');
        highlight.className = 'source-highlight-box source-highlight-box-mineru';
        highlight.style.left = `${(x0 / 1000) * sw}px`;
        highlight.style.top = `${(y0 / 1000) * sh}px`;
        highlight.style.width = `${((x1 - x0) / 1000) * sw}px`;
        highlight.style.height = `${((y1 - y0) / 1000) * sh}px`;
        layer.appendChild(highlight);
    });
}

function updateMineruMarkdown(task) {
    const contentList = task.contentList || [];
    let md = '';

    contentList.forEach((item) => {
        if (item.type === 'text' && item.text) {
            const level = item.level;
            if (level === 1) md += `# ${item.text}\n\n`;
            else if (level === 2) md += `## ${item.text}\n\n`;
            else if (level === 3) md += `### ${item.text}\n\n`;
            else md += `${item.text}\n\n`;
        } else if (item.type === 'image' && item.img_idx != null) {
            const imgKey = `images/${item.img_idx}.png`;
            md += `![](${imgKey})\n\n`;
        } else if (item.type === 'table') {
            if (item.text) md += `${item.text}\n\n`;
        } else if (item.type === 'equation' && item.text) {
            md += `$$\n${item.text}\n$$\n\n`;
        }
    });

    task.markdown = md;
    saveTask(task);
}

// ── Folder Management ─────────────────────────────────────────

async function loadFolders() {
    try {
        const resp = await fetch('/api/folders');
        if (!resp.ok) return;
        const data = await resp.json();
        folders = data.folders || [];
        renderFolderSelect();
    } catch (err) {
        console.error('Failed to load folders:', err);
    }
}

function renderFolderSelect() {
    const sel = els.folderSelect;
    if (!sel) return;
    const current = sel.value;
    // Keep first option ("全部文件")
    while (sel.options.length > 1) sel.remove(1);
    for (const f of folders) {
        const opt = document.createElement('option');
        opt.value = f.id;
        opt.textContent = `${f.name} (${f.taskCount ?? 0})`;
        sel.appendChild(opt);
    }
    // Restore selection if still valid
    if (current && folders.some(f => f.id === current)) {
        sel.value = current;
    }
}

async function createFolderDialog() {
    const name = prompt(t('请输入文件夹名称：'));
    if (!name) return;
    try {
        const resp = await fetch('/api/folders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        if (!resp.ok) {
            const data = await resp.json();
            alert(data.detail || t('创建文件夹失败'));
            return;
        }
        await loadFolders();
        // Auto-select the new folder
        const newFolder = folders[folders.length - 1];
        if (newFolder) {
            els.folderSelect.value = newFolder.id;
            activeFolderId = newFolder.id;
            renderTaskList();
        }
    } catch (err) {
        console.error('Failed to create folder:', err);
        alert(t('创建文件夹失败'));
    }
}

function handleFolderSelectChange() {
    activeFolderId = els.folderSelect.value || null;
    renderTaskList();
}

function handleFolderContextMenu(e) {
    e.preventDefault();
    const folderId = els.folderSelect.value;
    if (!folderId) return;
    const folder = folders.find(f => f.id === folderId);
    if (!folder) return;

    const menu = document.createElement('div');
    menu.className = 'folder-context-menu';
    menu.innerHTML = `
        <button class="folder-menu-item" data-action="rename">${t('重命名')}</button>
        <button class="folder-menu-item folder-menu-danger" data-action="delete">${t('删除文件夹')}</button>
    `;
    document.body.appendChild(menu);

    // Position
    const rect = els.folderSelect.getBoundingClientRect();
    menu.style.top = rect.bottom + 4 + 'px';
    menu.style.left = rect.left + 'px';

    function closeMenu() { menu.remove(); }
    menu.addEventListener('click', async (ev) => {
        const action = ev.target.dataset.action;
        closeMenu();
        if (action === 'rename') {
            const newName = prompt(t('请输入新名称：'), folder.name);
            if (!newName || newName === folder.name) return;
            try {
                const resp = await fetch(`/api/folders/${folderId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: newName }),
                });
                if (resp.ok) {
                    await loadFolders();
                    await loadTasks();
                    renderTaskList();
                }
            } catch (err) {
                console.error('Rename folder failed:', err);
            }
        } else if (action === 'delete') {
            if (!confirm(t('确定删除文件夹「{name}」？文件将移回根目录。', { name: folder.name }))) return;
            try {
                const resp = await fetch(`/api/folders/${folderId}`, { method: 'DELETE' });
                if (resp.ok) {
                    activeFolderId = null;
                    els.folderSelect.value = '';
                    await loadFolders();
                    await loadTasks();
                    renderTaskList();
                }
            } catch (err) {
                console.error('Delete folder failed:', err);
            }
        }
    });
    // Close on outside click
    setTimeout(() => {
        document.addEventListener('click', function handler(ev) {
            if (!menu.contains(ev.target)) {
                closeMenu();
                document.removeEventListener('click', handler);
            }
        });
    }, 0);
}

async function moveTaskToFolder(taskId, targetFolderId) {
    try {
        const resp = await fetch(`/api/tasks/${taskId}/folder`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folderId: targetFolderId || null }),
        });
        if (resp.ok) {
            await loadFolders();
            await loadTasks();
            renderTaskList();
        } else {
            const data = await resp.json();
            alert(data.detail || t('移动失败'));
        }
    } catch (err) {
        console.error('Move task failed:', err);
        alert(t('移动失败'));
    }
}

// ── Drag & Drop for tasks into folders ────────────────────────────

function setupTaskDragDrop(el, taskId) {
    el.addEventListener('dragstart', (e) => {
        draggedTaskId = taskId;
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', taskId);
        el.classList.add('dragging');
    });
    el.addEventListener('dragend', () => {
        draggedTaskId = null;
        el.classList.remove('dragging');
        // Remove drop highlight from sidebar
        els.sidebar?.classList.remove('folder-drop-active');
    });
}

// Set up the sidebar as a drop target — when a folder is selected in the
// dropdown, dropping a task on the sidebar moves it into that folder.
// When "全部文件" is selected, dropping moves the task OUT of any folder.
(function initFolderDropZone() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;

    sidebar.addEventListener('dragover', (e) => {
        if (!draggedTaskId) return;
        // Only activate if user has a folder selected (not "全部文件")
        const targetFolderId = els.folderSelect?.value;
        if (targetFolderId) {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            sidebar.classList.add('folder-drop-active');
        }
    });

    sidebar.addEventListener('dragleave', (e) => {
        // Only remove highlight if actually leaving the sidebar
        if (!sidebar.contains(e.relatedTarget)) {
            sidebar.classList.remove('folder-drop-active');
        }
    });

    sidebar.addEventListener('drop', async (e) => {
        e.preventDefault();
        sidebar.classList.remove('folder-drop-active');
        const taskId = e.dataTransfer.getData('text/plain') || draggedTaskId;
        if (!taskId) return;
        const targetFolderId = els.folderSelect?.value || null;
        if (targetFolderId) {
            await moveTaskToFolder(taskId, targetFolderId);
        }
        draggedTaskId = null;
    });
})();

// ── Right-click context menu for task folder assignment ─────────────

function showTaskFolderMenu(e, task) {
    // Remove any existing menu
    document.getElementById('task-folder-menu')?.remove();

    const menu = document.createElement('div');
    menu.id = 'task-folder-menu';
    menu.className = 'folder-context-menu';

    // Build folder options
    let html = '';
    if (folders.length === 0) {
        html = `<button class="folder-menu-item" data-action="create">${t('新建文件夹')}</button>`;
    } else {
        // "移出文件夹" option (only if task is currently in a folder)
        if (task.folderId) {
            html += `<button class="folder-menu-item" data-action="remove">${t('移出文件夹')}</button>`;
        }
        // List all folders as options
        for (const f of folders) {
            const isCurrent = f.id === task.folderId;
            const icon = isCurrent ? '✓ ' : '';
            html += `<button class="folder-menu-item${isCurrent ? ' current-folder' : ''}" data-folder-id="${f.id}">${icon}${f.name}</button>`;
        }
        html += `<div class="folder-menu-divider"></div>`;
        html += `<button class="folder-menu-item" data-action="create">${t('新建文件夹')}</button>`;
    }

    menu.innerHTML = html;
    document.body.appendChild(menu);

    // Position the menu near the click point
    menu.style.top = e.clientY + 'px';
    menu.style.left = e.clientX + 'px';
    // Adjust if menu would overflow viewport
    requestAnimationFrame(() => {
        const rect = menu.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            menu.style.left = (window.innerWidth - rect.width - 8) + 'px';
        }
        if (rect.bottom > window.innerHeight) {
            menu.style.top = (window.innerHeight - rect.height - 8) + 'px';
        }
    });

    function closeMenu() { menu.remove(); }

    menu.addEventListener('click', async (ev) => {
        const btn = ev.target.closest('.folder-menu-item');
        if (!btn) return;
        closeMenu();

        const action = btn.dataset.action;
        const folderId = btn.dataset.folderId;

        if (action === 'remove') {
            // Move task out of current folder
            await moveTaskToFolder(task.id, null);
        } else if (action === 'create') {
            // Create a new folder and move task into it
            const name = prompt(t('请输入文件夹名称：'));
            if (!name) return;
            try {
                const resp = await fetch('/api/folders', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name }),
                });
                if (!resp.ok) {
                    const data = await resp.json();
                    alert(data.detail || t('创建文件夹失败'));
                    return;
                }
                const newFolder = resp.json ? await resp.json() : null;
                if (newFolder?.id) {
                    await moveTaskToFolder(task.id, newFolder.id);
                }
            } catch (err) {
                console.error('Create folder failed:', err);
                alert(t('创建文件夹失败'));
            }
        } else if (folderId) {
            // Move task into the selected folder
            await moveTaskToFolder(task.id, folderId);
        }
    });

    // Close on outside click or scroll
    setTimeout(() => {
        const handler = (ev) => {
            if (!menu.contains(ev.target)) {
                closeMenu();
                document.removeEventListener('click', handler);
                document.removeEventListener('scroll', handler, true);
            }
        };
        document.addEventListener('click', handler);
        document.addEventListener('scroll', handler, true);
    }, 0);
}
