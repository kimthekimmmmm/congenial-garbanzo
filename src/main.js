import './style.css';

const $ = selector => document.querySelector(selector);
const dropZone = $('#drop-zone');
const input = $('#file-input');
const browse = $('#browse-button');
const uploadPrompt = $('#upload-prompt');
const fileInfo = $('#file-info');
const fileName = $('#file-name');
const fileMeta = $('#file-meta');
const remove = $('#remove-file');
const separate = $('#separate-button');
const helper = $('#helper-text');
const results = $('#results');
const tracks = $('#tracks');
const toast = $('#toast');
let selectedFile;
let activeAudio;

const stemDetails = {
  vocals: { label: 'Sång', detail: 'AI-isolerad sång', color: 'coral', icon: '♩' },
  drums: { label: 'Trummor', detail: 'AI-isolerade trummor', color: 'lime', icon: '◒' },
  bass: { label: 'Bas', detail: 'AI-isolerad bas', color: 'lilac', icon: '⌁' },
  other: { label: 'Övrigt', detail: 'Övriga instrument', color: 'sky', icon: '✦' },
};

const formatSize = bytes => bytes < 1024 ** 2 ? `${Math.ceil(bytes / 1024)} KB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`;
function showToast(message) { toast.textContent = message; toast.classList.add('visible'); window.setTimeout(() => toast.classList.remove('visible'), 2600); }
function stopPlayback() { if (activeAudio) { activeAudio.pause(); activeAudio = null; } document.querySelectorAll('.play.active').forEach(button => { button.classList.remove('active'); button.textContent = '▶'; }); }
function clearFile() { stopPlayback(); selectedFile = null; input.value = ''; uploadPrompt.hidden = false; fileInfo.hidden = true; separate.disabled = true; results.hidden = true; helper.textContent = 'Välj en ljudfil för att komma igång'; }
function setFile(file) {
  if (!file || (!file.type.startsWith('audio/') && !/\.(mp3|wav|flac|m4a|ogg)$/i.test(file.name))) { showToast('Välj en ljudfil i ett av de stödda formaten.'); return; }
  if (file.size > 500 * 1024 * 1024) { showToast('Filen är större än 500 MB.'); return; }
  selectedFile = file; fileName.textContent = file.name; fileMeta.textContent = `${formatSize(file.size)} · Redo för AI-separering`;
  uploadPrompt.hidden = true; fileInfo.hidden = false; separate.disabled = false; results.hidden = true; helper.textContent = 'Filen är klar. Separeringen körs lokalt med Demucs.';
}
function makeTrack(stem) {
  const details = stemDetails[stem.name];
  const item = document.createElement('article'); item.className = `track ${details.color}`;
  item.innerHTML = `<span class="track-icon">${details.icon}</span><div class="track-copy"><strong>${details.label}</strong><small>${details.detail} · WAV</small></div><button class="play" aria-label="Spela ${details.label}">▶</button><a class="download" href="${stem.url}" download aria-label="Hämta ${details.label}">↓</a>`;
  item.querySelector('.play').addEventListener('click', event => playStem(stem.url, event.currentTarget)); tracks.append(item);
}
function playStem(url, button) {
  if (button.classList.contains('active')) { stopPlayback(); return; }
  stopPlayback(); activeAudio = new Audio(url); activeAudio.play(); button.classList.add('active'); button.textContent = '❚❚';
  activeAudio.addEventListener('ended', stopPlayback, { once: true }); activeAudio.addEventListener('error', () => { stopPlayback(); showToast('Spåret kunde inte spelas upp.'); }, { once: true });
}

browse.addEventListener('click', event => { event.stopPropagation(); input.click(); }); dropZone.addEventListener('click', () => input.click());
dropZone.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') input.click(); }); input.addEventListener('change', () => setFile(input.files[0]));
['dragenter', 'dragover'].forEach(type => dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.add('dragging'); })); ['dragleave', 'drop'].forEach(type => dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.remove('dragging'); })); dropZone.addEventListener('drop', event => setFile(event.dataTransfer.files[0])); remove.addEventListener('click', event => { event.stopPropagation(); clearFile(); });
separate.addEventListener('click', async () => {
  if (!selectedFile) return;
  const data = new FormData(); data.append('file', selectedFile); separate.classList.add('loading'); separate.disabled = true; separate.querySelector('span').textContent = 'AI-separerar…'; helper.textContent = 'Demucs analyserar låten. Detta kan ta ett par minuter.';
  try {
    const response = await fetch('/api/separate', { method: 'POST', body: data }); const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Separeringen misslyckades.');
    tracks.replaceChildren(); payload.stems.forEach(makeTrack); results.hidden = false; helper.textContent = 'Färdigt — lyssna på eller hämta dina AI-separerade spår.'; results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (error) { showToast(error.message); helper.textContent = 'Kunde inte separera filen. Kontrollera att Demucs är installerat.'; }
  finally { separate.classList.remove('loading'); separate.disabled = false; separate.querySelector('span').textContent = 'Separera igen'; }
});
$('#download-all').addEventListener('click', () => document.querySelectorAll('.download').forEach((link, index) => window.setTimeout(() => link.click(), index * 300)));
