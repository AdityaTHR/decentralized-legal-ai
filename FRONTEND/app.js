// Enhanced Akash Network Legal AI Assistant JavaScript

// Configuration
const API_BASE_URL = 'http://localhost:8000'; // Backend API URL
const AKASH_API_URL = 'https://api.akash.network'; // Akash Network API (demo)

// Global state
let currentFiles = {
    summarize: null,
    extract: null
};

let processingStats = {
    currentNode: 'akash-gpu-' + Math.floor(Math.random() * 20 + 1).toString().padStart(2, '0'),
    gpuUtilization: Math.floor(Math.random() * 30 + 70),
    activeNodes: Math.floor(Math.random() * 10 + 20),
    processingTime: 0
};

// Processing messages for Akash Network
const PROCESSING_MESSAGES = [
    "Connecting to Akash GPU network...",
    "Processing on distributed nodes...",
    "Analyzing document with AI...",
    "Extracting legal entities...",
    "Generating summary on GPU cluster...",
    "Searching legal database...",
    "Optimizing results across nodes...",
    "Finalizing privacy-preserving processing..."
];

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('Akash Legal AI Assistant initializing...');
    initializeAkashStatus();
    initializeNavigation();
    initializeFileUploads();
    initializeEventListeners();
    startStatusUpdates();
    checkBackendConnection();
});

// Akash Network Status Management
function initializeAkashStatus() {
    console.log('Initializing Akash Network status...');
    updateGPUCount();
    updateNetworkStatus();
}

function updateGPUCount() {
    const gpuCountElement = document.getElementById('gpu-count');
    if (gpuCountElement) {
        gpuCountElement.textContent = processingStats.activeNodes;
    }
}

function updateNetworkStatus() {
    // Simulate network health check
    const statusItems = document.querySelectorAll('.status-item .status-icon');
    statusItems.forEach((icon, index) => {
        const isOnline = Math.random() > 0.1; // 90% uptime simulation
        icon.textContent = isOnline ? '🟢' : '🟡';
    });
}

function startStatusUpdates() {
    // Update GPU stats every 5 seconds
    setInterval(() => {
        processingStats.activeNodes = Math.max(15, processingStats.activeNodes + Math.floor(Math.random() * 6 - 3));
        processingStats.gpuUtilization = Math.max(40, Math.min(95, processingStats.gpuUtilization + Math.floor(Math.random() * 10 - 5)));
        updateGPUCount();
        updateProcessingStats();
    }, 5000);
}

function updateProcessingStats() {
    const nodeElement = document.getElementById('processing-node');
    const utilizationElement = document.getElementById('gpu-utilization');
    const currentNodeElement = document.getElementById('current-node');
    const currentUtilizationElement = document.getElementById('current-utilization');
    
    if (nodeElement) nodeElement.textContent = processingStats.currentNode;
    if (utilizationElement) utilizationElement.textContent = processingStats.gpuUtilization + '%';
    if (currentNodeElement) currentNodeElement.textContent = processingStats.currentNode;
    if (currentUtilizationElement) currentUtilizationElement.textContent = processingStats.gpuUtilization + '%';
}

// Navigation Management - Fixed
function initializeNavigation() {
    console.log('Initializing enhanced navigation...');
    
    // Navigation link clicks - Fixed event handling
    const navLinks = document.querySelectorAll('.nav-link');
    console.log('Found nav links:', navLinks.length);
    navLinks.forEach((link, index) => {
        const targetPage = link.getAttribute('data-page');
        console.log(`Nav link ${index}: ${targetPage}`);
        if (targetPage) {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Nav link clicked:', targetPage);
                showPage(targetPage);
            });
        }
    });

    // Hero action buttons - Fixed
    const heroButtons = document.querySelectorAll('.hero-actions .btn');
    console.log('Found hero buttons:', heroButtons.length);
    heroButtons.forEach((btn, index) => {
        const targetPage = btn.getAttribute('data-page');
        console.log(`Hero button ${index}: ${targetPage}`);
        if (targetPage) {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Hero button clicked:', targetPage);
                showPage(targetPage);
            });
        }
    });

    // Feature card buttons - Fixed
    const featureButtons = document.querySelectorAll('.feature-card .btn');
    console.log('Found feature buttons:', featureButtons.length);
    featureButtons.forEach((btn, index) => {
        const card = btn.closest('.feature-card');
        const targetPage = card ? card.getAttribute('data-page') : null;
        console.log(`Feature button ${index}: ${targetPage}`);
        if (targetPage) {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log(`Feature button ${index} clicked:`, targetPage);
                showPage(targetPage);
            });
        }
    });

    // Feature cards themselves - Fixed
    const featureCards = document.querySelectorAll('.feature-card');
    console.log('Found feature cards:', featureCards.length);
    featureCards.forEach((card, index) => {
        const targetPage = card.getAttribute('data-page');
        console.log(`Feature card ${index}: ${targetPage}`);
        if (targetPage) {
            card.addEventListener('click', function(e) {
                // Only trigger if not clicking on a button
                if (!e.target.classList.contains('btn') && !e.target.closest('.btn')) {
                    e.preventDefault();
                    e.stopPropagation();
                    console.log('Feature card clicked:', targetPage);
                    showPage(targetPage);
                }
            });
        }
    });

    // Brand navigation - Fixed
    const brandClickables = [
        '.nav-brand', 
        '.brand-text', 
        '.brand-icon', 
        '.brand-text-container'
    ];
    
    brandClickables.forEach(selector => {
        const elements = document.querySelectorAll(selector);
        elements.forEach(element => {
            element.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                console.log('Brand clicked, going home');
                showPage('home');
            });
            element.style.cursor = 'pointer';
        });
    });
}

function showPage(pageId) {
    console.log('Attempting to switch to page:', pageId);
    
    // Hide all pages
    const pages = document.querySelectorAll('.page');
    console.log('Found pages:', pages.length);
    pages.forEach(page => {
        page.classList.remove('active');
        console.log('Hiding page:', page.id);
    });
    
    // Show target page
    const targetPageId = pageId + '-page';
    const targetPage = document.getElementById(targetPageId);
    console.log('Looking for page with ID:', targetPageId);
    
    if (targetPage) {
        targetPage.classList.add('active');
        console.log('Successfully activated page:', pageId);
        
        // Update navigation active state
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            link.classList.remove('active');
            const linkPage = link.getAttribute('data-page');
            if (linkPage === pageId) {
                link.classList.add('active');
                console.log('Updated active nav link for:', pageId);
            }
        });
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        // Page-specific initialization
        if (pageId === 'home') {
            // Re-initialize home page animations if needed
            console.log('Home page activated');
        }
        
    } else {
        console.error('Page not found:', targetPageId);
        // List all available pages for debugging
        const allPages = document.querySelectorAll('.page');
        console.log('Available pages:');
        allPages.forEach(page => console.log('  -', page.id));
        
        // Fallback to home page
        console.log('Falling back to home page');
        const homePage = document.getElementById('home-page');
        if (homePage) {
            homePage.classList.add('active');
        }
    }
}

// File Upload Management
function initializeFileUploads() {
    console.log('Initializing file uploads...');
    setupFileUpload('summarize');
    setupFileUpload('extract');
}

function setupFileUpload(type) {
    const uploadArea = document.getElementById(`upload-area-${type}`);
    const fileInput = document.getElementById(`file-input-${type}`);
    const browseLink = document.getElementById(`browse-${type}`);

    if (!uploadArea || !fileInput || !browseLink) {
        console.error('Upload elements not found for type:', type);
        return;
    }

    console.log(`Setting up file upload for: ${type}`);

    // Click handlers
    browseLink.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });
    
    uploadArea.addEventListener('click', (e) => {
        if (e.target === browseLink || e.target.closest('#browse-' + type)) return;
        fileInput.click();
    });

    // File selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelection(e.target.files[0], type);
        }
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelection(e.dataTransfer.files[0], type);
        }
    });
}

function handleFileSelection(file, type) {
    console.log('Processing file selection:', file.name, 'for type:', type);
    
    if (!file) return;

    // Validate file type
    const validTypes = ['pdf', 'docx', 'txt'];
    const fileExt = file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(fileExt)) {
        showError('Please select a valid file type (PDF, DOCX, or TXT)');
        return;
    }

    // Store file
    currentFiles[type] = file;

    // Update UI
    const uploadArea = document.getElementById(`upload-area-${type}`);
    const uploadContent = uploadArea.querySelector('.upload-content');
    
    uploadArea.classList.add('has-file');
    uploadContent.innerHTML = `
        <div class="upload-icon">✅</div>
        <div class="file-info">
            <span class="file-name">${file.name}</span>
            <span class="file-size">(${formatFileSize(file.size)})</span>
        </div>
        <p class="upload-subtext">Click to change file</p>
    `;

    // Enable submit button
    const submitBtn = document.getElementById(`${type}-btn`);
    if (submitBtn) {
        submitBtn.disabled = false;
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Event Listeners
function initializeEventListeners() {
    console.log('Initializing event listeners...');
    
    // Document Summarization
    const summarizeBtn = document.getElementById('summarize-btn');
    if (summarizeBtn) {
        summarizeBtn.addEventListener('click', handleSummarize);
    }

    // Semantic Search
    const searchBtn = document.getElementById('search-btn');
    if (searchBtn) {
        searchBtn.addEventListener('click', handleSearch);
    }

    // Entity Extraction
    const extractBtn = document.getElementById('extract-btn');
    if (extractBtn) {
        extractBtn.addEventListener('click', handleExtract);
    }

    // Petition Generation
    const petitionForm = document.getElementById('petition-form');
    if (petitionForm) {
        petitionForm.addEventListener('submit', handlePetitionSubmit);
    }

    // Copy buttons
    const copySummaryBtn = document.getElementById('copy-summary');
    if (copySummaryBtn) {
        copySummaryBtn.addEventListener('click', () => copyToClipboard('summary-result'));
    }
    
    const copyPetitionBtn = document.getElementById('copy-petition');
    if (copyPetitionBtn) {
        copyPetitionBtn.addEventListener('click', () => copyToClipboard('petition-preview'));
    }

    // Download button
    const downloadBtn = document.getElementById('download-petition');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadPetition);
    }

    initializeModals();
}

function initializeModals() {
    // Close modals when clicking outside
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('modal')) {
            closeModal(e.target.id);
        }
    });

    // Escape key to close modals
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const openModals = document.querySelectorAll('.modal:not(.hidden)');
            openModals.forEach(modal => closeModal(modal.id));
        }
    });
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('hidden');
    }
}

// Enhanced Processing Modal
function showProcessingModal(title, message) {
    const modal = document.getElementById('processing-modal');
    const titleElement = document.getElementById('processing-title');
    const messageElement = document.getElementById('processing-message');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    if (titleElement) titleElement.textContent = title;
    if (messageElement) messageElement.textContent = message;
    
    // Update processing stats
    processingStats.currentNode = 'akash-gpu-' + Math.floor(Math.random() * 20 + 1).toString().padStart(2, '0');
    updateProcessingStats();
    
    if (modal) {
        modal.classList.remove('hidden');
        
        // Animate progress bar
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 100) progress = 100;
            
            if (progressFill) {
                progressFill.style.width = progress + '%';
            }
            if (progressText) {
                progressText.textContent = `${Math.floor(progress)}% complete`;
            }
            
            if (progress >= 100) {
                clearInterval(progressInterval);
            }
        }, 300);
    }
    
    return modal;
}

// API Functions with Enhanced Akash Integration
async function handleSummarize() {
    console.log('Starting Akash GPU summarization...');
    
    if (!currentFiles.summarize) {
        showError('Please select a file first');
        return;
    }

    const btn = document.getElementById('summarize-btn');
    const languageSelect = document.getElementById('language-select');
    const language = languageSelect ? languageSelect.value : 'en';
    const processingStats = document.getElementById('processing-stats');
    
    setButtonLoading(btn, true);
    showProcessingModal('Processing on Akash GPU Node...', 'Analyzing document with distributed AI processing');

    try {
        const formData = new FormData();
        formData.append('file', currentFiles.summarize);
        formData.append('language', language);

        // Show processing stats
        if (processingStats) {
            processingStats.style.display = 'block';
            updateProcessingStats();
        }

        // Simulate processing time for demo
        await new Promise(resolve => setTimeout(resolve, 3000));

        const response = await fetch(`${API_BASE_URL}/upload_summarize/`, {
            method: 'POST',
            body: formData
        });

        let summary;
        if (response.ok) {
            const data = await response.json();
            summary = data.summary;
        } else {
            throw new Error('Backend not available');
        }

        displaySummary(summary);
        closeModal('processing-modal');
        showSuccess('Document successfully processed on Akash Network!');
        
    } catch (error) {
        console.log('Using demo mode for summarization');
        
        // Generate demo summary based on file type/name
        const fileName = currentFiles.summarize.name.toLowerCase();
        let demoSummary = "This is an AI-generated summary processed on Akash Network's distributed GPU infrastructure. ";
        
        if (fileName.includes('contract')) {
            demoSummary += "The document appears to be a legal contract containing terms and conditions, obligations of parties, and dispute resolution mechanisms. Key provisions include payment terms, liability clauses, and termination conditions.";
        } else if (fileName.includes('petition')) {
            demoSummary += "The document is a legal petition filed before the court containing grounds for relief, factual allegations, and prayers sought by the petitioner. The matter involves legal issues requiring judicial intervention.";
        } else {
            demoSummary += "The legal document contains important provisions, clauses, and legal language relevant to the matter at hand. The document outlines rights, obligations, and legal procedures applicable to the parties involved.";
        }
        
        if (language === 'or') {
            demoSummary = "ଏହା ଏକ AI ଦ୍ୱାରା ପ୍ରସ୍ତୁତ ସାରାଂଶ ଯାହା Akash Network ର ବିତରଣ GPU infrastructure ରେ ପ୍ରକ୍ରିୟାକୃତ। ଏହି ଦଲିଲରେ ଗୁରୁତ୍ୱପୂର୍ଣ୍ଣ ଆଇନଗତ ବ୍ୟବସ୍ଥା ଓ ସର୍ତ୍ତାବଳୀ ରହିଛି।";
        } else if (language === 'hi') {
            demoSummary = "यह Akash Network के वितरित GPU infrastructure पर प्रोसेसिंग के साथ AI द्वారा जनरेट किया गया सारांश है। दस्तावेज़ में महत्वपूर्ण कानूनी प्रावधान और शर्तें शामिल हैं।";
        }
        
        displaySummary(demoSummary);
        closeModal('processing-modal');
        showSuccess('Demo summary generated using Akash Network simulation!');
        
    } finally {
        setButtonLoading(btn, false);
    }
}

async function handleSearch() {
    const queryInput = document.getElementById('search-query');
    const topKSelect = document.getElementById('search-top-k');
    
    if (!queryInput) {
        showError('Search form not found');
        return;
    }

    const query = queryInput.value.trim();
    const topK = topKSelect ? parseInt(topKSelect.value) : 3;

    if (!query) {
        showError('Please enter a search query');
        return;
    }

    const btn = document.getElementById('search-btn');
    setButtonLoading(btn, true);
    showProcessingModal('Searching on Akash Network...', 'Performing semantic search across distributed legal corpus');

    try {
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 2000));

        const response = await fetch(`${API_BASE_URL}/semantic_search/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                top_k: topK
            })
        });

        let results;
        if (response.ok) {
            const data = await response.json();
            results = data.results;
        } else {
            throw new Error('Backend not available');
        }

        displaySearchResults(results);
        closeModal('processing-modal');
        showSuccess('Search completed on Akash Network!');
        
    } catch (error) {
        console.log('Using demo mode for search');
        
        // Generate contextual demo results
        const queryLower = query.toLowerCase();
        let demoResults = [];
        
        if (queryLower.includes('contract') || queryLower.includes('agreement')) {
            demoResults = [
                `Contract enforcement case: The plaintiff filed a petition for enforcement of contract terms dated March 2023, seeking specific performance and damages for breach of agreement.`,
                `Agreement breach precedent: In similar cases, courts have held that material breach of contract allows the non-breaching party to seek termination and monetary compensation.`,
                `Contractual obligations: The defendant's failure to comply with settlement terms constituted a fundamental breach warranting judicial intervention and enforcement.`
            ];
        } else if (queryLower.includes('arbitration')) {
            demoResults = [
                `Arbitration award enforcement: The arbitration award dated March 15, 2023, was sought to be enforced under the Arbitration and Conciliation Act, with courts supporting enforcement.`,
                `Dispute resolution mechanism: Alternative dispute resolution through arbitration provides efficient resolution of commercial disputes outside traditional court systems.`,
                `Arbitral proceedings: The arbitrator's decision on commercial matters was upheld by the court, emphasizing the binding nature of arbitration awards.`
            ];
        } else {
            demoResults = [
                `Legal precedent relevant to "${query}": The court held in favor of the petitioner based on established legal principles and statutory provisions.`,
                `Case law interpretation: Similar matters have been decided by superior courts, providing guidance on the application of relevant legal provisions.`,
                `Judicial opinion: The court's reasoning in comparable cases emphasizes the importance of procedural compliance and substantive justice.`
            ];
        }
        
        displaySearchResults(demoResults.slice(0, topK));
        closeModal('processing-modal');
        showSuccess('Demo search results generated on Akash Network simulation!');
        
    } finally {
        setButtonLoading(btn, false);
    }
}

async function handleExtract() {
    if (!currentFiles.extract) {
        showError('Please select a file first');
        return;
    }

    const btn = document.getElementById('extract-btn');
    setButtonLoading(btn, true);
    showProcessingModal('Extracting Entities on Akash GPU...', 'Analyzing document for legal entities and citations using NLP models');

    try {
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 2500));

        const formData = new FormData();
        formData.append('file', currentFiles.extract);

        const response = await fetch(`${API_BASE_URL}/extract_entities/`, {
            method: 'POST',
            body: formData
        });

        let data;
        if (response.ok) {
            data = await response.json();
        } else {
            throw new Error('Backend not available');
        }

        displayExtractionResults(data);
        closeModal('processing-modal');
        showSuccess('Entity extraction completed on Akash Network!');
        
    } catch (error) {
        console.log('Using demo mode for extraction');
        
        // Generate contextual demo data based on filename
        const fileName = currentFiles.extract.name.toLowerCase();
        let demoData = {
            entities: [
                { text: 'Supreme Court of India', label: 'ORG' },
                { text: 'Delhi High Court', label: 'ORG' },
                { text: 'New Delhi', label: 'GPE' },
                { text: 'Justice Sharma', label: 'PERSON' }
            ],
            citations: [
                'Kesavananda Bharati v. State of Kerala, 1973',
                'Maneka Gandhi v. Union of India, 1978'
            ]
        };
        
        if (fileName.includes('contract')) {
            demoData.entities = [
                { text: 'ABC Corporation Ltd.', label: 'ORG' },
                { text: 'Mumbai', label: 'GPE' },
                { text: 'John Smith', label: 'PERSON' },
                { text: 'Commercial Court', label: 'ORG' },
                { text: 'March 15, 2024', label: 'DATE' }
            ];
            demoData.citations = [
                'Indian Contract Act, 1872',
                'Hadley v. Baxendale, 1854'
            ];
        }
        
        displayExtractionResults(demoData);
        closeModal('processing-modal');
        showSuccess('Demo extraction results generated on Akash Network simulation!');
        
    } finally {
        setButtonLoading(btn, false);
    }
}

async function handlePetitionSubmit(e) {
    e.preventDefault();

    const petitionerInput = document.getElementById('petitioner');
    const respondentInput = document.getElementById('respondent');
    const groundsInput = document.getElementById('grounds');
    const prayerInput = document.getElementById('prayer');

    if (!petitionerInput || !respondentInput || !groundsInput || !prayerInput) {
        showError('Form fields not found');
        return;
    }

    const petitioner = petitionerInput.value.trim();
    const respondent = respondentInput.value.trim();
    const grounds = groundsInput.value.trim();
    const prayer = prayerInput.value.trim();

    if (!petitioner || !respondent || !grounds || !prayer) {
        showError('Please fill in all fields');
        return;
    }

    const btn = e.target.querySelector('button[type="submit"]');
    setButtonLoading(btn, true);
    showProcessingModal('Generating Legal Petition...', 'Formatting petition according to court standards using AI assistance');

    try {
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 1500));

        const response = await fetch(`${API_BASE_URL}/format_petition/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                petitioner,
                respondent,
                grounds,
                prayer
            })
        });

        let petition;
        if (response.ok) {
            const data = await response.json();
            petition = data.petition;
        } else {
            throw new Error('Backend not available');
        }

        displayPetition(petition);
        closeModal('processing-modal');
        showSuccess('Legal petition generated successfully!');
        
    } catch (error) {
        console.log('Using demo mode for petition generation');
        
        const currentDate = new Date().toLocaleDateString('en-IN');
        const demoPetition = `IN THE HIGH COURT OF DELHI
AT NEW DELHI

CIVIL WRIT PETITION NO. _____ OF ${new Date().getFullYear()}

IN THE MATTER OF:

${petitioner.toUpperCase()}                    ...PETITIONER

VERSUS

${respondent.toUpperCase()}                    ...RESPONDENT

PETITION UNDER ARTICLE 226 OF THE CONSTITUTION OF INDIA

TO,
THE HON'BLE CHIEF JUSTICE AND HIS COMPANION JUSTICES
OF THE HIGH COURT OF DELHI AT NEW DELHI

THE HUMBLE PETITION OF THE PETITIONER ABOVE-NAMED

MOST RESPECTFULLY SHOWETH:

GROUNDS:
${grounds}

PRAYER:
In view of the facts and circumstances stated above, it is most respectfully prayed that this Hon'ble Court may be pleased to:

${prayer}

AND FOR SUCH OTHER AND FURTHER RELIEF AS THIS HON'BLE COURT MAY DEEM FIT AND PROPER IN THE CIRCUMSTANCES OF THE CASE.

AND FOR THIS ACT OF KINDNESS, THE PETITIONER AS IN DUTY BOUND, SHALL EVER PRAY.

PETITIONER

THROUGH: ________________
        ADVOCATE
        
Place: New Delhi
Date: ${currentDate}`;

        displayPetition(demoPetition);
        closeModal('processing-modal');
        showSuccess('Demo petition generated with proper court formatting!');
        
    } finally {
        setButtonLoading(btn, false);
    }
}

// Display Functions
function displaySummary(summary) {
    const resultDiv = document.getElementById('summary-result');
    const copyBtn = document.getElementById('copy-summary');
    
    if (resultDiv) {
        resultDiv.textContent = summary;
        resultDiv.style.fontStyle = 'normal';
    }
    if (copyBtn) {
        copyBtn.style.display = 'inline-block';
    }
}

function displaySearchResults(results) {
    const resultsSection = document.getElementById('search-results');
    const resultsContent = document.getElementById('search-results-content');
    const searchTimeElement = document.getElementById('search-time');
    
    if (!resultsContent) return;
    
    resultsContent.innerHTML = '';
    
    results.forEach((result, index) => {
        const resultItem = document.createElement('div');
        resultItem.className = 'search-result-item';
        resultItem.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <strong>Result ${index + 1}</strong>
                <span class="tech-badge">Akash GPU</span>
            </div>
            <p>${result}</p>
        `;
        resultsContent.appendChild(resultItem);
    });
    
    if (searchTimeElement) {
        const processingTime = (Math.random() * 1.5 + 0.5).toFixed(1);
        searchTimeElement.textContent = `Processed in ${processingTime}s on distributed nodes`;
    }
    
    if (resultsSection) {
        resultsSection.style.display = 'block';
    }
}

function displayExtractionResults(data) {
    const resultDiv = document.getElementById('extract-result');
    if (!resultDiv) return;
    
    let html = '';
    
    if (data.entities && data.entities.length > 0) {
        html += '<div class="entity-section"><h4>🏷️ Extracted Entities</h4><div class="entity-list">';
        data.entities.forEach(entity => {
            const entityClass = entity.label.toLowerCase().replace('_', '');
            html += `<span class="entity-tag ${entityClass}">${entity.text} (${entity.label})</span>`;
        });
        html += '</div></div>';
    }
    
    if (data.citations && data.citations.length > 0) {
        html += '<div class="entity-section"><h4>📚 Legal Citations</h4>';
        data.citations.forEach(citation => {
            html += `<div class="citation-item">${citation}</div>`;
        });
        html += '</div>';
    }
    
    if (!html) {
        html = '<p class="placeholder-text">No entities or citations found in the document.</p>';
    }
    
    resultDiv.innerHTML = html;
}

function displayPetition(petition) {
    const previewDiv = document.getElementById('petition-preview');
    const actionsDiv = document.getElementById('petition-actions');
    
    if (previewDiv) {
        previewDiv.textContent = petition;
        previewDiv.style.fontStyle = 'normal';
    }
    if (actionsDiv) {
        actionsDiv.style.display = 'flex';
    }
}

// Utility Functions
function setButtonLoading(button, isLoading) {
    if (!button) return;
    
    const spinner = button.querySelector('.loading-spinner');
    const text = button.querySelector('.btn-text');
    
    if (isLoading) {
        button.disabled = true;
        if (spinner) spinner.classList.remove('hidden');
        if (text) text.style.display = 'none';
    } else {
        button.disabled = false;
        if (spinner) spinner.classList.add('hidden');
        if (text) text.style.display = 'inline';
    }
}

function showError(message) {
    const modal = document.getElementById('error-modal');
    const messageEl = document.getElementById('error-message');
    
    if (messageEl) messageEl.textContent = message;
    if (modal) modal.classList.remove('hidden');
    
    console.error('Error:', message);
}

function showSuccess(message) {
    const modal = document.getElementById('success-modal');
    const messageEl = document.getElementById('success-message');
    
    if (messageEl) messageEl.textContent = message;
    if (modal) modal.classList.remove('hidden');
    
    console.log('Success:', message);
}

// Copy to clipboard functionality
async function copyToClipboard(elementId) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const text = element.textContent;
    
    try {
        await navigator.clipboard.writeText(text);
        showSuccess('Content copied to clipboard!');
    } catch (error) {
        console.error('Copy failed:', error);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        showSuccess('Content copied to clipboard!');
    }
}

// Download petition as text file
function downloadPetition() {
    const petitionText = document.getElementById('petition-preview');
    if (!petitionText) return;
    
    const text = petitionText.textContent;
    const petitionerInput = document.getElementById('petitioner');
    const petitioner = petitionerInput ? petitionerInput.value || 'Petition' : 'Petition';
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `${petitioner.replace(/\s+/g, '_')}_Petition_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    window.URL.revokeObjectURL(url);
    showSuccess('Petition downloaded successfully!');
}

// Backend connection check
async function checkBackendConnection() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);
        
        const response = await fetch(`${API_BASE_URL}/`, {
            method: 'GET',
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
            console.log('Backend connection established');
            showBackendStatus(true);
        } else {
            console.warn('Backend responded with status:', response.status);
            showBackendStatus(false);
        }
    } catch (error) {
        console.warn('Backend connection failed, running in demo mode');
        showBackendStatus(false);
    }
}

function showBackendStatus(isConnected) {
    const notification = document.createElement('div');
    notification.id = 'backend-status';
    notification.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: ${isConnected ? 'var(--color-success)' : 'var(--color-info)'};
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 14px;
        z-index: 1001;
        opacity: 0.95;
        box-shadow: var(--shadow-lg);
        max-width: 300px;
        border: 2px solid ${isConnected ? 'var(--color-success)' : 'var(--color-primary)'};
    `;
    
    if (isConnected) {
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <span>✅</span>
                <div>
                    <strong>Akash Network Connected</strong><br>
                    <small>GPU processing active</small>
                </div>
            </div>
        `;
    } else {
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <span>🔄</span>
                <div>
                    <strong>Demo Mode Active</strong><br>
                    <small>Simulating Akash Network processing</small>
                </div>
            </div>
        `;
    }
    
    document.body.appendChild(notification);
    
    // Remove notification after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

console.log('Akash Network Legal AI Assistant loaded successfully');