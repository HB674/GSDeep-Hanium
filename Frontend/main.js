// ==========================
// main.js  (FULL REPLACEMENT)
// ==========================

// API 베이스 (config.js가 먼저 로드되면 window.API_BASE 사용)
const BASE = (window.API_BASE || "").replace(/\/+$/, "") || "";

// --------------------------
// 페이지 로드 애니메이션 등
// --------------------------
jQuery(window).on('load', function() {
  "use strict";

  // HIDE PRELOADER
  $(".preloader").addClass("hide-preloader");

  // SHOW/ANIMATE ANIMATION CONTAINER
  setTimeout(function(){
    $("#intro .animation-container").each(function() {
      var e = $(this);
      setTimeout(function(){
        e.addClass("run-animation");
      }, e.data("animation-delay") );
    });
  }, 800);
});

jQuery(document).ready(function($) {
  "use strict";

  // ONE PAGE NAVIGATION
  $(".navigation-main .navigation-items").onePageNav({
    currentClass: "current",
    changeHash: false,
    scrollSpeed: 750,
    scrollThreshold: 0.5,
    filter: ":not(.external)",
    easing: "swing"
  });

  // SMOOTH SCROLL FOR SAME PAGE LINKS
  $(document).on('click', 'a.smooth-scroll', function(event) {
    event.preventDefault();
    $('html, body').animate({
      scrollTop: $( $.attr(this, 'href') ).offset().top
    }, 500);
  });

  // INIT PARALLAX PLUGIN
  $(".background-content.parallax-on").parallax({
    scalarX: 24,
    scalarY: 15,
    frictionX: 0.1,
    frictionY: 0.1,
  });

  // SCROLL REVEAL SETUP
  window.sr = ScrollReveal();
  sr.reveal(".scroll-animated-from-bottom", {
    duration: 600,
    delay: 0,
    origin: "bottom",
    rotate: { x: 0, y: 0, z: 0 },
    opacity: 0,
    distance: "20vh",
    viewFactor: 0.4,
    scale: 1,
  });

  // WORK CAROUSEL
  $('.work-carousel').owlCarousel({
    center: true,
    items: 1,
    loop: true,
    margin: 30,
    autoplay: true,
    responsive:{
      800:{ items: 3 },
    }
  });


  // ------------------------------------------
  // === Service Upload UI (UPLOAD→JOB→POLL) ===
  // ------------------------------------------
  const $form     = $('#service-upload');
  if(!$form.length) return;

  const $dropImg  = $('#drop-zone');          // 이미지 드롭존
  const $imgInput = $('#file-input');         // 이미지 input
  const $preview  = $('#upload-preview');
  const $prompt   = $dropImg.find('.upload-prompt');
  const $status   = $('#upload-status');
  const $message    = $('#upload-message');    // 사용자가 적는 텍스트

  // 오디오/보이스 옵션
  const $audioInput = $('#audio-input');      // accept=".wav,.mp3"
  const $audioDrop  = $('#audio-drop');
  const $audioName  = $('#audio-filename');
  const $audioPrev  = $('#audio-preview');
  if ($audioInput && $audioInput[0]) $audioInput[0].required = false;

  const $selGender  = $('#voice-gender');     // male|female
  const $selAge     = $('#voice-age');        // young|middle|old
  const $rangePitch = $('#voice-level');      // -20 ~ 20

  // 텍스트 입력/파일 (파일은 있으면 우선)
  const $textInput  = $('#text-input').length ? $('#text-input') : $('#upload-message');
  const $textFile   = $('#text-file');        // .txt 파일(선택, 없으면 무시)

  let selectedImage = null;  // 이미지 파일
  let selectedAudio = null;  // 오디오 파일

  // ---------- 프리뷰/UX ----------
  function resetImgPreview(){
    $preview.hide().attr('src','');
    $prompt.show();

    // 결과 비디오도 숨김(백워드 호환)
    var $vid = $('#result-video');
    if ($vid.length) $vid.hide().attr('src','');
  }
  function showImgPreview(file){
    if(!file){ resetImgPreview(); return; }
    if(file.type && file.type.indexOf('image') === 0){
      const url = URL.createObjectURL(file);
      $preview.attr('src', url).show();
      $prompt.hide();
    } else {
      resetImgPreview();
    }
  }
  function showAudioPreview(file){
    if(!file) return;
    if ($audioName) $audioName.text(file.name);
    if ($audioPrev && $audioPrev.length){
      $audioPrev[0].src = URL.createObjectURL(file);
      $audioPrev.show();
      $audioPrev[0].onloadeddata = () => URL.revokeObjectURL($audioPrev[0].src);
    }
  }

  // ---------- 이미지 드래그/선택 ----------
  $dropImg.on('click', () => $imgInput.trigger('click'));
  $dropImg.on('keydown', (e) => {
    if(e.key === 'Enter' || e.key === ' '){
      e.preventDefault(); $imgInput.trigger('click');
    }
  });
  $dropImg.on('drag dragstart dragend dragover dragenter dragleave drop', function(e){
    e.preventDefault(); e.stopPropagation();
  }).on('dragover dragenter', () => $dropImg.addClass('dragover')
  ).on('dragleave dragend drop', () => $dropImg.removeClass('dragover')
  ).on('drop', function(e){
    const dt = e.originalEvent.dataTransfer;
    if(dt && dt.files && dt.files.length){
      selectedImage = dt.files[0];
      try{
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(selectedImage);
        $imgInput[0].files = dataTransfer.files;
      }catch(_){}
      showImgPreview(selectedImage);
    }
  });

  $imgInput.on('change', function(){
    selectedImage = this.files && this.files[0] ? this.files[0] : null;
    showImgPreview(selectedImage);
  });

  // ---------- 오디오 드래그/선택 ----------
  if ($audioDrop && $audioDrop.length){
    ['dragenter','dragover'].forEach(ev =>
      $audioDrop.on(ev, e => { e.preventDefault(); e.stopPropagation(); $audioDrop.addClass('dragover'); })
    );
    ['dragleave','drop'].forEach(ev =>
      $audioDrop.on(ev, e => { e.preventDefault(); e.stopPropagation(); $audioDrop.removeClass('dragover'); })
    );
    $audioDrop.on('drop', e => {
      const files = e.originalEvent.dataTransfer?.files;
      if (files && files.length) {
        const f = files[0];
        const isWav = f.type === 'audio/wav' || /\.wav$/i.test(f.name);
        const isMp3 = f.type === 'audio/mpeg' || /\.mp3$/i.test(f.name);
        if (isWav || isMp3) {
          selectedAudio = f;
          try {
            const dt = new DataTransfer();
            dt.items.add(f);
            $audioInput[0].files = dt.files;
          } catch (err) {
            console.warn('DataTransfer 주입 실패, selectedAudio로 진행:', err);
          }
          showAudioPreview(f);
        } else {
          alert('WAV 또는 MP3 파일만 업로드할 수 있습니다.');
        }
      }
    });
  }

  $audioInput.on('change', function(){
    const f = this.files && this.files[0] ? this.files[0] : null;
    if (f){
      selectedAudio = f;
      showAudioPreview(f);
    }
  });

  // ---------- 보이스 옵션 ----------
  function getVoiceProfile(){
    const gender = ($selGender && $selGender.val()) || 'female'; // male|female
    let   age    = ($selAge && $selAge.val())       || 'young';  // young|middle|old
    if (age === 'middle' || age === 'old') age = 'adult';
    return `${gender}_${age}`; // 예: female_young, male_adult
  }
  function getPitch(){
    const raw = ($rangePitch && $rangePitch.val()) || '0';
    const n = Number(raw);
    return Number.isFinite(n) ? n : 0;
  }

  // ---------- 텍스트 값(파일 우선) ----------
  async function getTextValue() {
    const file = ($textFile && $textFile[0] && $textFile[0].files && $textFile[0].files[0])
      ? $textFile[0].files[0]
      : null;

    if (file) {
      try {
        // 현대 브라우저
        const txt = await file.text();
        return (txt || '').trim();
      } catch (_) {
        // 폴백 (거의 필요 없음)
        const txt = await new Promise((resolve) => {
          const r = new FileReader();
          r.onload = () => resolve(String(r.result || '').trim());
          r.onerror = () => resolve('');
          r.readAsText(file);
        });
        return txt;
      }
    }
    const t = ($textInput && $textInput.val()) ? $textInput.val().trim() : '';
    return t;
  }

  // ---------- API 유틸 ----------
  async function uploadImageAndAudio(imgFile, msg, audioFile){
    const fd = new FormData();
    fd.append('file', imgFile, imgFile.name || 'upload.png');
    if (audioFile) fd.append('audio', audioFile, audioFile.name || 'voice.wav');
    if (msg) fd.append('message', msg);

    const res = await fetch(`${BASE}/api/upload`, { method:'POST', body: fd });
    if(!res.ok){
      const t = await res.text();
      throw new Error(`/api/upload 실패: ${res.status} ${t.slice(0,200)}`);
    }
    const json = await res.json();
    if(!json || !json.image_path){
      throw new Error('업로드 응답에 image_path가 없습니다.');
    }
    return json; // { ok, image_path, audio_path|null, message }
  }

  // 이미지 + 텍스트 업로드(신규)
  async function uploadImageAndText(imgFile, text, textBasename, msg){
    const fd = new FormData();
    fd.append('file', imgFile, imgFile.name || 'upload.png');
    fd.append('text', text);
    if (textBasename) fd.append('text_basename', textBasename);
    if (msg) fd.append('message', msg);

    const res = await fetch(`${BASE}/api/upload_with_text`, { method:'POST', body: fd });
    if(!res.ok){
      const t = await res.text();
      throw new Error(`/api/upload_with_text 실패: ${res.status} ${t.slice(0,200)}`);
    }
    const json = await res.json();
    if(!json || !json.image_path || !json.text_path){
      throw new Error('업로드 응답에 image_path 또는 text_path가 없습니다.');
    }
    return json; // { ok, image_path, text_path, message }
  }

  async function createAudioJob(image_path, audio_path){
    if (!audio_path) throw new Error('오디오가 업로드되지 않았습니다(audio_path 없음). WAV/MP3를 선택하세요.');
    const body = {
      image_path,
      audio_path,
      use_applio: true,
      pitch: getPitch(),
      voice_profile: getVoiceProfile()
    };
    const res = await fetch(`${BASE}/jobs/audio`, {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify(body)
    });
    if(!res.ok){
      const t = await res.text();
      throw new Error(`/jobs/audio 실패: ${res.status} ${t.slice(0,200)}`);
    }
    return res.json(); // { job_id, status }
  }

  // 텍스트(TTS) 파이프라인 잡 생성(신규)
  async function createTTSJob(image_path){
    const body = {
      image_path,
      // voice: 'onyx',
      response_format: 'mp3',
      use_applio: true,
      pitch: getPitch(),
      voice_profile: getVoiceProfile()
      // tts_text는 보내지 않습니다 → 서버가 input_text 최신 파일을 사용
    };
    const res = await fetch(`${BASE}/jobs/tts`, {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify(body)
    });
    if(!res.ok){
      const t = await res.text();
      throw new Error(`/jobs/tts 실패: ${res.status} ${t.slice(0,200)}`);
    }
    return res.json(); // { job_id, status }
  }

  async function pollJob(job_id, onProgress){
    while(true){
      const res = await fetch(`${BASE}/jobs/${job_id}`);
      if(!res.ok){
        const t = await res.text();
        throw new Error(`/jobs/${job_id} 실패: ${res.status} ${t.slice(0,200)}`);
      }
      const j = await res.json();
      if (onProgress && j.step) onProgress(j.step);

      if(j.status === 'done')  return j;
      if(j.status === 'failed') throw new Error(`잡 실패: ${j.error || '알 수 없는 오류'}`);

      await new Promise(r => setTimeout(r, 1500));
    }
  }

  // 결과 URL 빌더: 절대/상대 경로 모두 대응
  function buildFileUrl(p){
    if (!p) return '';
    if (p.startsWith('/')) {
      return `${BASE}/files?path=${encodeURIComponent(p)}`;   // 절대경로
    }
    return `${BASE}/files/${encodeURIComponent(p)}`;          // 상대경로
  }

  function setBusy(b){
    $form.find('button, input, select, textarea').prop('disabled', b);
  }

  // ---------- 제출 ----------
  $form.on('submit', async function(e){
    e.preventDefault();

    // 이미지 필수
    if(!selectedImage){
      $status.text('이미지 파일을 선택하세요.');
      return;
    }

    // 텍스트(파일 우선) 읽기
    const textVal = await getTextValue();

    // 오디오 후보 (오디오 모드일 때만 필요)
    let audioFile = selectedAudio
      || ($audioInput && $audioInput[0] && $audioInput[0].files && $audioInput[0].files[0])
      ? (selectedAudio || $audioInput[0].files[0])
      : null;

    // 모드 결정 규칙:
    // - 텍스트가 1자라도 있으면 => TTS (오디오가 있어도 무시)
    // - 텍스트가 없으면       => AUDIO
    const mode = textVal.length > 0 ? 'tts' : 'audio';

    // 최소 요구사항: 텍스트도 오디오도 없으면 막기
    if (mode === 'audio' && !audioFile) {
      $status.text('텍스트 또는 오디오 중 하나는 필요합니다. (텍스트 입력/파일 또는 WAV/MP3)');
      return;
    }

    try{
      setBusy(true);
      $status.text('업로드 중...');

      // 1) 업로드 → 경로 획득 (모드별)
      let up = null;
      if (mode === 'tts') {
        // 이미지 + 텍스트 업로드 (오디오는 무시)
        up = await uploadImageAndText(selectedImage, textVal, /*textBasename*/ null, $message.val());
      } else {
        // 이미지 + 오디오 업로드
        up = await uploadImageAndAudio(selectedImage, $message.val(), audioFile);
        if (!up.audio_path) {
          $status.text('오디오 업로드 실패: audio_path가 없습니다. WAV/MP3 파일을 다시 선택해 주세요.');
          return;
        }
      }

      // 2) 잡 생성
      $status.text('잡 생성 중...');
      let job;
      if (mode === 'tts') {
        job = await createTTSJob(up.image_path);
      } else {
        job = await createAudioJob(up.image_path, up.audio_path);
      }

      // 3) 폴링
      $status.text(mode === 'tts'
        ? '처리 중입니다... (tts → applio → sadtalker → wav2lip → gfpgan)'
        : '처리 중입니다... (applio → sadtalker → wav2lip → gfpgan)'
      );
      const result = await pollJob(job.job_id, step => $status.text(`진행 중: ${step}...`));

      // 4) 결과 재생
      const finalRel = result?.artifacts?.final;
      if(!finalRel) throw new Error('최종 산출물 경로가 없습니다.');
      const finalUrl = buildFileUrl(finalRel);
      $status.text('완료!');
      if (typeof playResult === 'function') {
        playResult(finalUrl, { subline: $message && $message.val ? $message.val() : '' });
      } else {
        console.log('FINAL:', finalUrl);
      }

    } catch(err){
      console.error(err);
      $status.text('실패: ' + (err && err.message ? err.message : '네트워크/서버 오류'));
    } finally {
      setBusy(false);
    }
  });

  // ---------- 전송 시 로딩 GIF ----------
  (function(){
    const LOADING_GIF = 'assets/img/loading_page.gif'; // 경로 확인
    const service = document.querySelector('#service');
    if (!service) return;
    const form    = service.querySelector('#service-upload');
    const drop    = service.querySelector('#drop-zone');
    const prompt  = service.querySelector('.upload-prompt');
    const preview = service.querySelector('#upload-preview');
    if (!form || !drop || !preview) return;
    function showLoading() {
      if (prompt) prompt.style.display = 'none';
      preview.src = LOADING_GIF;
      preview.style.display = 'block';
      preview.style.objectFit = 'cover';
      drop.classList.add('is-loading');
    }
    form.addEventListener('submit', function(){ showLoading(); }, { capture: true });
  })();

  // 초기화
  resetImgPreview();
});


// ==========================
// 결과 표시 (라이트박스)
// ==========================
function openVideoLightbox({ src, poster = '', title = '', subline = '' }) {
  if (!src) { alert('동영상 주소를 찾을 수 없습니다.'); return; }

  const safe = (s) => String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));

  const content = `
    <div class="work-lightbox">
      <video class="lightbox-video" src="${safe(src)}" controls preload="metadata" playsinline ${poster ? `poster="${safe(poster)}"` : ''}></video>
      <div class="description">
        ${title ? `<h3>${safe(title)}</h3>` : ''}
        ${subline ? `<p class="subline">${safe(subline)}</p>` : ''}
      </div>
    </div>
  `;

  if (window.jQuery && window.jQuery.featherlight) {
    // Featherlight 사용 시: afterClose 훅에서 리셋
    window.jQuery.featherlight(content, {
      persist: false,
      closeOnEsc: true,
      closeOnClick: 'anywhere',
      afterContent() {
        const v = this.$content.find('video').get(0);
        if (v) v.play().catch(()=>{});
        if (v) v.focus({ preventScroll: true });
      },
      afterClose() {
        if (typeof window.resetUploadUI === 'function') window.resetUploadUI();
      }
    });
  } else {
    // 간이 오버레이 버전: 닫을 때 리셋
    const overlay = document.createElement('div');
    overlay.className = 'featherlight featherlight-open';
    overlay.innerHTML = `
      <div class="featherlight-content">
        ${content}
        <button type="button" class="featherlight-close-icon" aria-label="닫기">✕</button>
      </div>
    `;
    Object.assign(overlay.style, { position:'fixed', inset:'0', background:'rgba(0,0,0,.7)', zIndex:9999, display:'flex', alignItems:'center', justifyContent:'center' });
    document.body.appendChild(overlay);

    const doClose = () => {
      overlay.remove();
      if (typeof window.resetUploadUI === 'function') window.resetUploadUI();
    };

    overlay.querySelector('.featherlight-close-icon').onclick = doClose;
    overlay.addEventListener('click', (e) => { if (e.target === overlay) doClose(); });

    const v = overlay.querySelector('video');
    if (v) v.play().catch(()=>{});
  }
}

function playResult(url, opts = {}){
  const {
    title = 'Result',
    subline = '',
    poster = ''
  } = opts;

  openVideoLightbox({
    src: url,
    poster,
    title,
    subline
  });
}


// 업로드 패널을 초기 상태로 되돌리는 전역 함수
window.resetUploadUI = function resetUploadUI() {
  try {
    const $form     = $('#service-upload');
    const $dropImg  = $('#drop-zone');
    const $prompt   = $dropImg.find('.upload-prompt');
    const $preview  = $('#upload-preview');
    const $status   = $('#upload-status');
    const $imgInput = $('#file-input');
    const $audioInp = $('#audio-input');
    const $audioNm  = $('#audio-filename');
    const $audioPrev= $('#audio-preview');

    // 파일 input 정리
    if ($imgInput && $imgInput.length)  $imgInput.val('');
    if ($audioInp && $audioInp.length)  $audioInp.val('');

    // 프리뷰/로딩 표시 정리
    if ($preview && $preview.length) { 
      $preview.hide().attr('src',''); 
      $preview.css({ objectFit: '' });
    }
    if ($prompt && $prompt.length)  $prompt.show();
    if ($dropImg && $dropImg.length) $dropImg.removeClass('is-loading dragover');

    // 상태문구/오디오 프리뷰 정리
    if ($status && $status.length) $status.text('');
    if ($audioNm && $audioNm.length) $audioNm.text('');
    if ($audioPrev && $audioPrev.length) { 
      $audioPrev.hide(); 
      $audioPrev[0].src = ''; 
    }

    // 내부 변수도 깔끔히 (선언된 경우에만)
    if (window.selectedImage !== undefined) window.selectedImage = null;
    if (window.selectedAudio !== undefined) window.selectedAudio = null;

    // 버튼 등 비활성화 해제(혹시 남아 있으면)
    if ($form && $form.length) $form.find('button, input, select, textarea').prop('disabled', false);
  } catch (e) {
    console.warn('resetUploadUI 실패:', e);
  }
};
