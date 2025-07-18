const CACHE_NAME = 'agribot-v1';
const urlsToCache = [
    '/',
    '/static/fontawesome/css/all.min.css',
    '/static/fontawesome/webfonts/fa-solid-900.woff2',
    '/static/styles.css'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME).then(cache => {
            return cache.addAll(urlsToCache);
        })
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request).then(response => {
            return response || fetch(event.request);
        })
    );
});
