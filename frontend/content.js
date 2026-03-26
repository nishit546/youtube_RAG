/**
 * content.js
 * Injected into YouTube pages.
 * Responds to messages from the popup asking for the current video ID.
 */

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "GET_VIDEO_ID") {
    const params = new URLSearchParams(window.location.search);
    const videoId = params.get("v") || null;
    sendResponse({ videoId });
  }
});
