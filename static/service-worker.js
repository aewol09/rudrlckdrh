const CACHE_NAME = 'turtle-neck-cache-v1';
const urlsToCache = [
  '/',
  '/static/style.css',
  '/static/logo.png',
  // 필요한 정적 파일 추가
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(urlsToCache))
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request).then(response => response || fetch(event.request))
  );
});