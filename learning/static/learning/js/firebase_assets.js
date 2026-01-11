// Client module to load letter images from Firestore + Firebase Storage
// Usage: import and call loadLettersFromFirestore(containerId, options)

if (!window._firebaseConfig) {
    window._firebaseConfig = {
        apiKey: "AIzaSyB8_Fr_gN5KpQ99fMlJOYb_wsvpCFqPA3M",
        authDomain: "asl-learning-app-b53e6.firebaseapp.com",
        projectId: "asl-learning-app-b53e6",
        storageBucket: "asl-learning-app-b53e6.appspot.com",
        messagingSenderId: "1039064561840",
        appId: "1:1039064561840:web:66265fa70866634e1e6375",
        measurementId: "G-YXD6RGBCD6"
    };
}

async function initFirebase() {
    if (window._firebaseInitialized) return window._firebaseApp;
    const { initializeApp } = await import('https://www.gstatic.com/firebasejs/9.22.2/firebase-app.js');
    const { getFirestore, collection, getDocs } = await import('https://www.gstatic.com/firebasejs/9.22.2/firebase-firestore.js');
    const { getStorage, ref, getDownloadURL } = await import('https://www.gstatic.com/firebasejs/9.22.2/firebase-storage.js');

    const app = initializeApp(window._firebaseConfig);
    const db = getFirestore(app);
    const storage = getStorage(app);
    window._firebaseInitialized = true;
    window._firebaseApp = { app, db, storage, ref, getDownloadURL, collection, getDocs };
    return window._firebaseApp;
}

function el(tag, cls) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
}

export async function loadLettersFromFirestore(containerId = 'letters-container', options = {}) {
    // options: { collectionPath: 'Vocabulary', imageField: 'image', nameField: 'name' }
    const { collectionPath = 'Vocabulary', imageField = 'image', nameField = 'name' } = options;
    const fb = await initFirebase();
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';


    try {
        const collRef = fb.collection(fb.db, collectionPath);
        const snapshot = await fb.getDocs(collRef);
        if (!snapshot || snapshot.empty) {
            container.innerHTML = '<div class="text-muted small">No letters found</div>';
            return;
        }

        // Collect docs with actual data objects
        const docs = [];
        snapshot.forEach(d => {
            try {
                docs.push({ id: d.id, data: d.data() });
            } catch (e) {
                // skip
            }
        });

        // For deterministic order, sort by nameField if available
        docs.sort((a,b) => {
            const an = (a.data && a.data[nameField] ? a.data[nameField] : a.id || '').toString().toLowerCase();
            const bn = (b.data && b.data[nameField] ? b.data[nameField] : b.id || '').toString().toLowerCase();
            return an < bn ? -1 : an > bn ? 1 : 0;
        });

        for (const doc of docs) {
            const data = doc.data || {};
            const displayName = data[nameField] || doc.id || '';
            // image path may be stored in different fields; try imageField then common names
            let imgPath = data[imageField] || data.imagePath || data.storagePath || data.path || data.url || null;

            // If imgPath is an array (e.g., multiple URLs), pick the first string entry
            if (Array.isArray(imgPath)) {
                imgPath = imgPath.find(x => typeof x === 'string') || null;
            }

            // If imgPath is an object with url or downloadURL, extract
            if (imgPath && typeof imgPath === 'object') {
                imgPath = imgPath.url || imgPath.downloadURL || imgPath.fullPath || null;
            }

            const card = el('div', 'd-inline-block m-1 text-center');
            card.style.width = '72px';

            const img = el('img', 'rounded shadow');
            img.alt = displayName;
            img.style.width = '64px';
            img.style.height = '64px';
            img.style.objectFit = 'cover';

            const caption = el('div', 'small mt-1');
            caption.textContent = displayName;

            if (imgPath) {
                try {
                    // Normalize path: handle gs://, full URLs, or storage fullPath
                    if (typeof imgPath === 'string' && (imgPath.startsWith('https://') || imgPath.startsWith('http://'))) {
                        img.src = imgPath;
                    } else {
                        let storagePath = imgPath.toString();
                        if (storagePath.startsWith('gs://')) {
                            const parts = storagePath.split('/');
                            storagePath = parts.slice(3).join('/');
                        }
                        // remove leading slash
                        if (storagePath.startsWith('/')) storagePath = storagePath.substring(1);
                        const refObj = fb.ref(fb.storage, storagePath);
                        const url = await fb.getDownloadURL(refObj);
                        img.src = url;
                    }
                } catch (e) {
                    console.warn('Failed to fetch image for', doc.id, e);
                    img.src = 'https://via.placeholder.com/64?text=' + encodeURIComponent(displayName);
                }
            } else {
                img.src = 'https://via.placeholder.com/64?text=' + encodeURIComponent(displayName);
            }

            card.appendChild(img);
            card.appendChild(caption);
            container.appendChild(card);
        }

    } catch (e) {
        console.error('Failed to load letters from Firestore', e);
        container.innerHTML = '<div class="text-danger small">Lỗi tải dữ liệu chữ cái.</div>';
    }
}

// Backwards compat: expose global
window.FirebaseAssets = window.FirebaseAssets || {};
window.FirebaseAssets.loadLettersFromFirestore = loadLettersFromFirestore;
// Module to load letter images and word videos from Firebase Storage
// This file uses the modular Firebase web SDK via CDN imports.

const firebaseConfig = {
    apiKey: "AIzaSyB8_Fr_gN5KpQ99fMlJOYb_wsvpCFqPA3M",
    authDomain: "asl-learning-app-b53e6.firebaseapp.com",
    projectId: "asl-learning-app-b53e6",
    storageBucket: "asl-learning-app-b53e6.appspot.com",
    messagingSenderId: "1039064561840",
    appId: "1:1039064561840:web:66265fa70866634e1e6375",
    measurementId: "G-YXD6RGBCD6"
};

// Helper: dynamically import firebase modules from CDN
async function initFirebase() {
    if (window._firebaseInitialized) return window._firebaseApp;
    const { initializeApp } = await import('https://www.gstatic.com/firebasejs/9.22.2/firebase-app.js');
    const { getStorage, ref, getDownloadURL, listAll } = await import('https://www.gstatic.com/firebasejs/9.22.2/firebase-storage.js');

    const app = initializeApp(firebaseConfig);
    const storage = getStorage(app);
    window._firebaseInitialized = true;
    window._firebaseApp = { app, storage, ref, getDownloadURL, listAll };
    return window._firebaseApp;
}

export async function loadAllLetters(containerId = 'letters-container') {
    const fb = await initFirebase();
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    try {
        const lettersRef = fb.ref(fb.storage, 'letters/');
        const listing = await fb.listAll(lettersRef);
        // listing.items contains file refs for each file under letters/
        for (const itemRef of listing.items) {
            try {
                const url = await fb.getDownloadURL(itemRef);
                const name = itemRef.name; // e.g., "A.jpg"
                const base = name.split('.')[0];

                const card = document.createElement('div');
                card.className = 'd-inline-block m-1 text-center';
                card.style.width = '72px';

                const img = document.createElement('img');
                img.alt = base;
                img.className = 'rounded shadow';
                img.style.width = '64px';
                img.style.height = '64px';
                img.style.objectFit = 'cover';
                img.src = url;

                const caption = document.createElement('div');
                caption.className = 'small mt-1';
                caption.textContent = base;

                card.appendChild(img);
                card.appendChild(caption);
                container.appendChild(card);
            } catch (e) {
                // skip individual failures
                console.warn('Failed to load letter item', itemRef.name, e);
            }
        }
    } catch (e) {
        console.warn('Failed to list letters/', e);
        // fallback: show A-Z placeholders
        const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
        for (const L of letters) {
            const card = document.createElement('div');
            card.className = 'd-inline-block m-1 text-center';
            card.style.width = '72px';

            const img = document.createElement('img');
            img.alt = L;
            img.className = 'rounded shadow';
            img.style.width = '64px';
            img.style.height = '64px';
            img.style.objectFit = 'cover';
            img.src = 'https://via.placeholder.com/64?text=' + L;

            const caption = document.createElement('div');
            caption.className = 'small mt-1';
            caption.textContent = L;

            card.appendChild(img);
            card.appendChild(caption);
            container.appendChild(card);
        }
    }
}

export async function loadWordsList(containerId = 'words-container') {
    const fb = await initFirebase();
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    try {
        const wordsRef = fb.ref(fb.storage, 'words/');
        const listing = await fb.listAll(wordsRef);

        // Group by base filename (without extension)
        const groups = {};
        for (const itemRef of listing.items) {
            const name = itemRef.name; // e.g., "hello.mp4" or "hello.jpg"
            const dot = name.lastIndexOf('.');
            const base = dot > 0 ? name.substring(0, dot) : name;
            const ext = dot > 0 ? name.substring(dot).toLowerCase() : '';
            if (!groups[base]) groups[base] = { thumbs: [], videos: [] };
            if (['.mp4', '.webm'].includes(ext)) groups[base].videos.push(itemRef);
            else groups[base].thumbs.push(itemRef);
        }

        for (const base of Object.keys(groups).sort()) {
            const item = groups[base];
            const card = document.createElement('div');
            card.className = 'card mb-2';
            card.style.maxWidth = '100%';

            const body = document.createElement('div');
            body.className = 'card-body d-flex align-items-center gap-2';

            const thumb = document.createElement('img');
            thumb.alt = base;
            thumb.className = 'rounded';
            thumb.style.width = '96px';
            thumb.style.height = '72px';
            thumb.style.objectFit = 'cover';

            const title = document.createElement('div');
            title.innerHTML = `<strong>${base}</strong>`;

            const btn = document.createElement('button');
            btn.className = 'btn btn-sm btn-outline-primary ms-auto';
            btn.textContent = 'Xem video';

            // set thumbnail if available
            if (item.thumbs.length > 0) {
                try {
                    const url = await fb.getDownloadURL(item.thumbs[0]);
                    thumb.src = url;
                } catch (e) {
                    thumb.src = 'https://via.placeholder.com/96x72?text=' + encodeURIComponent(base);
                }
            } else {
                thumb.src = 'https://via.placeholder.com/96x72?text=' + encodeURIComponent(base);
            }

            btn.addEventListener('click', async () => {
                // pick first available video
                if (item.videos.length === 0) {
                    alert('Không tìm thấy video cho từ: ' + base);
                    return;
                }
                try {
                    const url = await fb.getDownloadURL(item.videos[0]);
                    showWordModal(base, url);
                } catch (e) {
                    alert('Lỗi tải video: ' + e.message);
                }
            });

            body.appendChild(thumb);
            body.appendChild(title);
            body.appendChild(btn);
            card.appendChild(body);
            container.appendChild(card);
        }

    } catch (e) {
        console.warn('Failed to list words/', e);
    }
}

function showWordModal(word, videoUrl) {
    // simple modal using bootstrap modal markup
    let modal = document.getElementById('wordModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'wordModal';
        modal.className = 'modal fade';
        modal.tabIndex = -1;
        modal.innerHTML = `
<div class="modal-dialog modal-lg modal-dialog-centered">
  <div class="modal-content">
    <div class="modal-header">
      <h5 class="modal-title">${word}</h5>
      <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
    </div>
    <div class="modal-body text-center">
      <video id="wordVideoPlayer" controls style="width:100%; height:auto;">
        <source src="" type="video/mp4">
        Your browser does not support the video tag.
      </video>
    </div>
  </div>
</div>
`;
        document.body.appendChild(modal);
    }
    const player = modal.querySelector('#wordVideoPlayer');
    if (player) {
        player.pause();
        player.querySelector('source').src = videoUrl;
        player.load();
        player.play().catch(()=>{});
    }
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

// auto-export for non-module usage
window.FirebaseAssets = {
    loadAllLetters,
    loadWordsList
};
