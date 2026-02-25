import { useState, useRef, useEffect, useCallback } from "react";

const BLUE = { 50: "#EBF8FF", 100: "#BEE3F8", 200: "#90CDF4", 300: "#63B3ED", 400: "#4299E1", 500: "#2B7AB8", 600: "#1A5D8F", 700: "#1A365D", 800: "#153E75", 900: "#0F2B46" };
const ACCENT = { red: "#FC8181", redLight: "#FFF5F5", green: "#68D391", greenLight: "#F0FFF4", orange: "#F6AD55", orangeLight: "#FFFAF0", purple: "#B794F4", purpleLight: "#FAF5FF" };
const MODEL_COLORS = [BLUE[300], ACCENT.red, ACCENT.green, ACCENT.orange];
const MODEL_LABELS = ["장소 / 배경", "사고 유형", "차량 A", "차량 B"];
const MODEL_ICONS = ["📍", "💥", "🚗", "🚙"];
const API_URL = "http://51.20.205.173:5002";

const GLOBAL_CSS = `
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html,body,#root{height:100%;font-family:'Noto Sans KR','Outfit',system-ui,sans-serif;background:#F8FAFD;color:#1A365D;overflow-x:hidden}
::-webkit-scrollbar{width:0;height:0}
@keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes scaleIn{from{opacity:0;transform:scale(.92)}to{opacity:1;transform:scale(1)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes spin{to{transform:rotate(360deg)}}
.fade-up{animation:fadeUp .6s cubic-bezier(.22,1,.36,1) both}
.fade-in{animation:fadeIn .5s ease both}
.scale-in{animation:scaleIn .5s cubic-bezier(.22,1,.36,1) both}
`;

/* ───────── shared components ───────── */
const Phone = ({ children }) => (
  <div style={{ maxWidth: 430, margin: "0 auto", minHeight: "100dvh", background: "#FFFFFF", position: "relative", overflow: "hidden", boxShadow: "0 0 80px rgba(26,54,93,.08)" }}>{children}</div>
);
const StepDots = ({ current, total = 4 }) => (
  <div style={{ display: "flex", gap: 6, justifyContent: "center", padding: "8px 0 4px" }}>
    {Array.from({ length: total }).map((_, i) => (
      <div key={i} style={{ width: i <= current ? 20 : 8, height: 8, borderRadius: 4, background: i <= current ? `linear-gradient(135deg,${BLUE[400]},${BLUE[300]})` : "#E2E8F0", transition: "all .4s cubic-bezier(.22,1,.36,1)" }} />
    ))}
  </div>
);
const NavBar = ({ title, onBack, step }) => (
  <div style={{ position: "sticky", top: 0, zIndex: 50, background: "rgba(255,255,255,.88)", backdropFilter: "blur(20px)", WebkitBackdropFilter: "blur(20px)", borderBottom: "1px solid rgba(226,232,240,.6)" }}>
    <div style={{ display: "flex", alignItems: "center", padding: "14px 20px 6px" }}>
      {onBack ? <button onClick={onBack} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 22, color: BLUE[500], width: 30, textAlign: "left", fontFamily: "inherit" }}>‹</button> : <div style={{ width: 30 }} />}
      <span style={{ flex: 1, textAlign: "center", fontSize: 17, fontWeight: 700, letterSpacing: -0.3 }}>{title}</span>
      <div style={{ width: 30 }} />
    </div>
    {step !== undefined && <StepDots current={step} />}
  </div>
);
const PrimaryBtn = ({ children, onClick, disabled, icon }) => (
  <button onClick={onClick} disabled={disabled} style={{ width: "100%", height: 56, borderRadius: 14, border: "none", cursor: disabled ? "default" : "pointer", background: disabled ? "#CBD5E0" : `linear-gradient(135deg,${BLUE[500]},${BLUE[300]})`, color: "#fff", fontSize: 17, fontWeight: 800, fontFamily: "inherit", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, boxShadow: disabled ? "none" : `0 4px 20px rgba(43,122,184,.3)`, transition: "all .3s ease", opacity: disabled ? 0.6 : 1 }}>
    {icon && <span style={{ fontSize: 20 }}>{icon}</span>}{children}
  </button>
);
const Badge = ({ children, color = BLUE[500], bg }) => (
  <span style={{ display: "inline-flex", alignItems: "center", gap: 4, padding: "5px 12px", borderRadius: 20, fontSize: 12, fontWeight: 600, color, background: bg || `${color}12`, border: `1px solid ${color}30`, whiteSpace: "nowrap" }}>{children}</span>
);
const SectionHeader = ({ icon, text, color = BLUE[300] }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "24px 0 14px", paddingBottom: 10, borderBottom: "1px solid #EDF2F7" }}>
    <div style={{ width: 36, height: 36, borderRadius: 10, background: `${color}18`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, flexShrink: 0 }}>{icon}</div>
    <span style={{ fontSize: 16, fontWeight: 700, color: "#4A5568" }}>{text}</span>
  </div>
);
const fmt = (s) => { const m = Math.floor(s / 60); const sec = Math.floor(s % 60); const ms = Math.floor((s % 1) * 10); return `${m}:${String(sec).padStart(2, "0")}.${ms}`; };

/* ───────── CustomVideoPlayer ───────── */
const CustomVideoPlayer = ({ src, trimStart = 0, trimEnd, isTrimmed = false }) => {
  const videoRef = useRef(null);
  const progressRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const clipDuration = isTrimmed ? (trimEnd - trimStart) : 0;
  const seekTo = useCallback((relTime) => { const v = videoRef.current; if (!v) return; const c = Math.max(0, Math.min(relTime, clipDuration)); v.currentTime = trimStart + c; setCurrentTime(c); }, [trimStart, clipDuration]);
  useEffect(() => { const v = videoRef.current; if (!v || !isTrimmed) return; const onL = () => { v.currentTime = trimStart; }; v.addEventListener("loadedmetadata", onL); if (v.readyState >= 1) v.currentTime = trimStart; return () => v.removeEventListener("loadedmetadata", onL); }, [src, trimStart, isTrimmed]);
  useEffect(() => { const v = videoRef.current; if (!v || !isTrimmed) return; const onT = () => { if (v.currentTime >= trimEnd - 0.05) { v.pause(); v.currentTime = trimStart; setPlaying(false); setCurrentTime(0); return; } if (v.currentTime < trimStart) v.currentTime = trimStart; setCurrentTime(Math.max(0, v.currentTime - trimStart)); }; v.addEventListener("timeupdate", onT); return () => v.removeEventListener("timeupdate", onT); }, [src, trimStart, trimEnd, isTrimmed]);
  const togglePlay = () => { const v = videoRef.current; if (!v) return; if (playing) { v.pause(); setPlaying(false); } else { if (v.currentTime < trimStart || v.currentTime >= trimEnd - 0.1) v.currentTime = trimStart; v.play(); setPlaying(true); } };
  const handleProgressClick = (e) => { if (!progressRef.current) return; const rect = progressRef.current.getBoundingClientRect(); seekTo(Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)) * clipDuration); };
  if (!isTrimmed) return (<div style={{ borderRadius: 14, overflow: "hidden", boxShadow: "0 4px 20px rgba(0,0,0,.08)", background: "#000" }}><video src={src} controls playsInline style={{ width: "100%", display: "block" }} /></div>);
  const pct = clipDuration > 0 ? (currentTime / clipDuration) * 100 : 0;
  return (
    <div style={{ borderRadius: 14, overflow: "hidden", boxShadow: "0 4px 20px rgba(0,0,0,.08)", background: "#000", position: "relative", userSelect: "none" }}>
      <video ref={videoRef} src={src} playsInline style={{ width: "100%", display: "block" }} onClick={togglePlay} />
      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, background: "linear-gradient(transparent, rgba(0,0,0,.75))", padding: "28px 14px 12px" }}>
        <div ref={progressRef} onClick={handleProgressClick} style={{ height: 20, display: "flex", alignItems: "center", cursor: "pointer", marginBottom: 6 }}>
          <div style={{ width: "100%", height: 5, borderRadius: 3, background: "rgba(255,255,255,.25)", position: "relative" }}>
            <div style={{ width: `${pct}%`, height: "100%", borderRadius: 3, background: `linear-gradient(90deg,${BLUE[300]},${BLUE[400]})`, transition: "width .1s linear" }} />
            <div style={{ position: "absolute", top: "50%", left: `${pct}%`, transform: "translate(-50%,-50%)", width: 14, height: 14, borderRadius: "50%", background: "#FFF", boxShadow: "0 1px 4px rgba(0,0,0,.4)" }} />
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <button onClick={togglePlay} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 22, color: "#FFF", padding: 0 }}>{playing ? "⏸" : "▶"}</button>
          <span style={{ fontSize: 13, fontWeight: 600, color: "#FFF", fontFamily: "'Outfit',monospace" }}>{fmt(currentTime)} / {fmt(clipDuration)}</span>
        </div>
      </div>
    </div>
  );
};

/* ───────── PAGE 1 ───────── */
const Page1 = ({ onNext }) => {
  const [show, setShow] = useState(false);
  useEffect(() => { setTimeout(() => setShow(true), 100); }, []);
  return (
    <Phone>
      <div style={{ minHeight: "100dvh", display: "flex", flexDirection: "column", justifyContent: "space-between", padding: "0 24px", background: "linear-gradient(180deg,#FFFFFF 0%,#F0F7FF 100%)", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: -80, right: -60, width: 260, height: 260, borderRadius: "50%", background: `radial-gradient(circle,${BLUE[100]}60,transparent 70%)`, animation: "float 6s ease-in-out infinite" }} />
        <div style={{ position: "absolute", bottom: 120, left: -80, width: 200, height: 200, borderRadius: "50%", background: `radial-gradient(circle,${BLUE[50]}80,transparent 70%)`, animation: "float 8s ease-in-out infinite 1s" }} />
        <div style={{ flex: 1 }} />
        <div style={{ textAlign: "center", opacity: show ? 1 : 0, transform: show ? "translateY(0)" : "translateY(30px)", transition: "all .8s cubic-bezier(.22,1,.36,1)" }}>
          <div style={{ width: 110, height: 110, margin: "0 auto 28px", borderRadius: 28, background: `linear-gradient(135deg,${BLUE[500]},${BLUE[300]})`, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: `0 12px 40px ${BLUE[300]}50`, position: "relative", overflow: "hidden" }}>
            <img src="/logo.png" alt="AI 문철 로고" style={{ width: 110, height: 110, objectFit: "contain", display: "block", borderRadius: "20%" }} />
          </div>
          <h1 style={{ fontSize: 42, fontWeight: 900, letterSpacing: -1.5, color: BLUE[700], fontFamily: "'Outfit','Noto Sans KR',sans-serif" }}>AI 문철</h1>


        </div>
        <div style={{ flex: 1.2 }} />
        <div style={{ paddingBottom: 48, opacity: show ? 1 : 0, transform: show ? "translateY(0)" : "translateY(20px)", transition: "all .8s cubic-bezier(.22,1,.36,1) .3s" }}>
          <PrimaryBtn onClick={onNext}>분석 시작하기</PrimaryBtn>
        </div>
      </div>
    </Phone>
  );
};

/* ───────── PAGE 2 ───────── */
const Page2 = ({ onNext, onBack, setVideoData }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [duration, setDuration] = useState(null);
  const [converting, setConverting] = useState(false);
  const [videoError, setVideoError] = useState(false);
  const inputRef = useRef();
  const videoRef = useRef();

  const handleFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setVideoError(false);
    setConverting(false);
    setDuration(null);
  };

  const handleLoadedMeta = () => {
    if (videoRef.current) {
      const d = videoRef.current.duration;
      setDuration(isFinite(d) ? d : null);
    }
  };

  const handleVideoError = async () => {
    setVideoError(true);
    if (!file || converting) return;
    setConverting(true);
    try {
      const formData = new FormData();
      formData.append("video", file);
      const res = await fetch(`${API_URL}/api/convert`, {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        setPreview(url);
        setVideoError(false);
      }
    } catch (err) {
      console.error("변환 실패:", err);
    } finally {
      setConverting(false);
    }
  };

  const handleNext = () => {
    if (!file || !preview) return;
    const d = duration || 10;
    const sig = `${file.name}_${file.size}_${Date.now()}`;
    setVideoData({ file, url: preview, duration: d, isTrimmed: false, trimStart: 0, trimEnd: d, sig });
    onNext(d <= 10);
  };

  return (
    <Phone>
      <NavBar title="영상 업로드" onBack={onBack} step={0} />
      <div style={{ padding: "20px 24px 40px", minHeight: "calc(100dvh - 100px)", display: "flex", flexDirection: "column" }}>
        <div className="fade-up">
          <h2 style={{ fontSize: 20, fontWeight: 800, color: BLUE[700] }}>분석할 블랙박스 영상을<br />업로드해 주세요</h2>
          <p style={{ fontSize: 14, color: "#8892B0", marginTop: 8 }}>MP4, AVI, MOV 형식을 지원합니다</p>
        </div>
        <div className="fade-up" style={{ animationDelay: ".1s", marginTop: 28, flex: 1 }}>
          {!file ? (
            <div onClick={() => inputRef.current?.click()} style={{ border: `2px dashed ${BLUE[300]}`, borderRadius: 20, padding: "52px 24px", background: "#F7FBFF", textAlign: "center", cursor: "pointer" }}>
              <div style={{ width: 64, height: 64, margin: "0 auto 16px", borderRadius: 20, background: `linear-gradient(135deg,${BLUE[50]},${BLUE[100]})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 32 }}>☁️</div>
              <p style={{ fontSize: 16, fontWeight: 700, color: BLUE[500] }}>터치하여 영상 선택</p>
              <p style={{ fontSize: 13, color: "#A0AEC0", marginTop: 6 }}>또는 파일 앱에서 가져오기</p>
            </div>
          ) : (
            <div className="scale-in" style={{ border: `2px solid ${BLUE[300]}`, borderRadius: 20, padding: 16, background: BLUE[50] }}>
              {converting ? (
                <div style={{ width: "100%", padding: "40px 20px", borderRadius: 12, background: "#1A202C", textAlign: "center" }}>
                  <div style={{ width: 40, height: 40, margin: "0 auto 12px", border: `3px solid ${BLUE[100]}`, borderTopColor: BLUE[500], borderRadius: "50%", animation: "spin .8s linear infinite" }} />
                  <p style={{ fontSize: 14, fontWeight: 600, color: "#FFF" }}>영상 변환 중...</p>
                  <p style={{ fontSize: 12, color: "#A0AEC0", marginTop: 6 }}>브라우저 미리보기를 위해 코덱을 변환하고 있습니다</p>
                </div>
              ) : (
                <video
                  ref={videoRef}
                  src={preview}
                  onLoadedMetadata={handleLoadedMeta}
                  onError={handleVideoError}
                  controls
                  playsInline
                  style={{ width: "100%", borderRadius: 12, background: "#000" }}
                />
              )}
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 12 }}>
                <div style={{ width: 32, height: 32, borderRadius: 8, background: "#C6F6D5", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>✓</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ fontSize: 14, fontWeight: 600, color: BLUE[700], overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</p>
                  {duration ? (
                    <p style={{ fontSize: 12, color: "#8892B0", marginTop: 2 }}>영상 길이: {duration.toFixed(1)}초</p>
                  ) : converting ? (
                    <p style={{ fontSize: 12, color: ACCENT.orange, marginTop: 2 }}>⚡ 코덱 변환 중...</p>
                  ) : videoError ? (
                    <p style={{ fontSize: 12, color: ACCENT.orange, marginTop: 2 }}>⚡ 변환 실패 — 분석은 서버에서 자동 변환됩니다</p>
                  ) : null}
                </div>
                <button onClick={() => { setFile(null); setPreview(null); setDuration(null); setVideoError(false); setConverting(false); }} style={{ background: "none", border: "none", fontSize: 13, color: BLUE[500], fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>변경</button>
              </div>
            </div>
          )}
          <input ref={inputRef} type="file" accept="video/*" onChange={handleFile} style={{ display: "none" }} />
        </div>
        <div style={{ paddingTop: 20 }}><PrimaryBtn onClick={handleNext} disabled={!file || converting}>다음</PrimaryBtn></div>
      </div>
    </Phone>
  );
};

/* ───────── PAGE 3 ───────── */
const Page3 = ({ onNext, onBack, videoData, setVideoData }) => {
  const dur = videoData?.duration || 30;
  const [accidentTime, setAccidentTime] = useState(Math.min(dur / 2, dur));
  const [trimming, setTrimming] = useState(false);
  const [trimDone, setTrimDone] = useState(false);
  const start = Math.max(0, accidentTime - 5);
  const end = Math.min(dur, accidentTime + 5);
  const handleTrim = () => { setTrimming(true); setTimeout(() => { setVideoData(prev => ({ ...prev, isTrimmed: true, trimStart: start, trimEnd: end })); setTrimming(false); setTrimDone(true); setTimeout(() => onNext(), 600); }, 1500); };
  return (
    <Phone>
      <NavBar title="사고 구간 설정" onBack={onBack} step={1} />
      <div style={{ padding: "16px 24px 40px", minHeight: "calc(100dvh - 100px)", display: "flex", flexDirection: "column" }}>
        <SectionHeader icon="🎬" text="원본 영상" />
        <div className="fade-up" style={{ borderRadius: 14, overflow: "hidden", boxShadow: "0 4px 20px rgba(0,0,0,.08)", background: "#000" }}>
          <video src={videoData?.url} controls playsInline style={{ width: "100%", display: "block" }} />
        </div>
        <div style={{ marginTop: 8 }}><Badge color={BLUE[500]}>전체 {dur.toFixed(1)}초</Badge></div>
        <SectionHeader icon="✂️" text="사고 시점 선택" color={ACCENT.orange} />
        <div className="fade-up" style={{ animationDelay: ".15s" }}>
          <p style={{ fontSize: 14, fontWeight: 600, color: BLUE[700], marginBottom: 16 }}>사고 발생 시점을 선택해 주세요</p>
          <div style={{ position: "relative", padding: "28px 0 12px" }}>
            <div style={{ position: "absolute", top: 0, left: `calc(${(accidentTime / dur) * 100}% - 28px)`, background: BLUE[500], color: "#fff", borderRadius: 8, padding: "3px 10px", fontSize: 12, fontWeight: 700, transition: "left .15s ease", whiteSpace: "nowrap", zIndex: 2 }}>
              {accidentTime.toFixed(1)}초
              <div style={{ position: "absolute", bottom: -4, left: "50%", transform: "translateX(-50%) rotate(45deg)", width: 8, height: 8, background: BLUE[500] }} />
            </div>
            <div style={{ position: "relative", height: 8, borderRadius: 4, background: "#E2E8F0" }}>
              <div style={{ position: "absolute", left: `${(start / dur) * 100}%`, width: `${((end - start) / dur) * 100}%`, height: "100%", borderRadius: 4, background: `linear-gradient(90deg,${BLUE[300]},${BLUE[400]})` }} />
            </div>
            <input type="range" min={0} max={dur} step={0.5} value={accidentTime} onChange={e => setAccidentTime(Number(e.target.value))} style={{ position: "absolute", top: 24, left: 0, width: "100%", height: 16, opacity: 0, cursor: "pointer" }} />
          </div>
        </div>

        {/* 상태 표시 영역: 버튼과 분리 */}
        {trimming && (
          <div className="fade-in" style={{ textAlign: "center", padding: "20px", marginTop: 20 }}>
            <div style={{ width: 40, height: 40, margin: "0 auto 12px", border: `3px solid ${BLUE[100]}`, borderTopColor: BLUE[500], borderRadius: "50%", animation: "spin .8s linear infinite" }} />
            <p style={{ fontSize: 15, fontWeight: 600, color: BLUE[600] }}>영상 자르는 중...</p>
          </div>
        )}

        {!trimming && trimDone && (
          <div className="scale-in" style={{ textAlign: "center", padding: "16px 0", marginTop: 20 }}>
            <div style={{ width: 48, height: 48, margin: "0 auto 8px", borderRadius: "50%", background: "#C6F6D5", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24 }}>✓</div>
            <p style={{ fontSize: 15, fontWeight: 700, color: ACCENT.green }}>자르기 완료!</p>
          </div>
        )}

        {/* 버튼 영역: 항상 표시 + 상태에 따라 disabled */}
        <div style={{ marginTop: "auto", paddingTop: 20 }}>
          <PrimaryBtn
            onClick={handleTrim}
            disabled={trimming || trimDone}
          >
            {trimming ? "자르는 중..." : trimDone ? "자르기 완료" : "영상 자르기"}
          </PrimaryBtn>
        </div>

      </div>
    </Phone>
  );
};

/* ───────── PAGE 4 ───────── */
const Page4 = ({ onNext, onBack, videoData }) => {
  const dur = videoData?.duration || 10;
  const isTrimmed = videoData?.isTrimmed || false;
  const trimStart = videoData?.trimStart || 0;
  const trimEnd = videoData?.trimEnd || dur;
  const clipDuration = isTrimmed ? (trimEnd - trimStart) : dur;
  return (
    <Phone>
      <NavBar title="영상 확인" onBack={onBack} step={2} />
      <div style={{ padding: "16px 24px 40px", minHeight: "calc(100dvh - 100px)", display: "flex", flexDirection: "column" }}>
        <SectionHeader icon="🎬" text={isTrimmed ? "편집된 영상" : "분석 대상 영상"} />

        <div className="fade-up"><CustomVideoPlayer src={videoData?.url} trimStart={trimStart} trimEnd={trimEnd} isTrimmed={isTrimmed} /></div>
        {!isTrimmed && <div style={{ marginTop: 10 }}><Badge color={BLUE[500]}>원본 영상 ({dur.toFixed(1)}초)</Badge></div>}

        <p style={{ fontSize: 13, color: "#8892B0", marginTop: 15 }}>{isTrimmed ? `원본 ${trimStart.toFixed(1)}초~${trimEnd.toFixed(1)}초 구간 (${clipDuration.toFixed(1)}초)` : "아래 영상으로 AI 분석이 진행됩니다"}</p>

        <div className="fade-up" style={{ animationDelay: ".15s", marginTop: 24, padding: "18px 20px", borderRadius: 14, background: "#F7FBFF", borderLeft: `4px solid ${BLUE[300]}` }}>
          <p style={{ fontSize: 14, fontWeight: 700, color: BLUE[700], marginBottom: 12 }}>AI가 4개 모델로 다음 항목을 분석합니다</p>
          {MODEL_LABELS.map((label, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 0", fontSize: 14, color: "#4A5568" }}>
              <span style={{ width: 24, height: 24, borderRadius: 7, background: `${MODEL_COLORS[i]}18`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 800, color: MODEL_COLORS[i] }}>{i + 1}</span>
              <span style={{ fontWeight: 500 }}>{label}</span>
            </div>
          ))}
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ paddingTop: 24 }}><PrimaryBtn onClick={onNext}>AI 정밀 분석 시작</PrimaryBtn></div>
      </div>
    </Phone>
  );
};

/* ═══════════════════════════════════════════════════════
   PAGE 5 : RESULTS — ✅ SSE 스트리밍으로 실시간 진행
   ═══════════════════════════════════════════════════════ */
const MODEL_KEYS = ["model1", "model2", "model3", "model4"];

const ResultCard = ({ data, index, visible }) => {
  const color = MODEL_COLORS[index];
  const icon = MODEL_ICONS[index];
  if (!data || !data.top) return null;
  return (
    <div style={{ background: "#FFF", border: "1px solid #EDF2F7", borderRadius: 16, padding: "16px 12px", boxShadow: "0 2px 12px rgba(0,0,0,.04)", minWidth: 0, overflow: "hidden", opacity: visible ? 1 : 0, transform: visible ? "translateY(0)" : "translateY(20px)", transition: `all .5s cubic-bezier(.22,1,.36,1) ${index * 0.1}s` }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "#8892B0", letterSpacing: 1.5, paddingBottom: 8, borderBottom: `2px solid ${color}`, marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}><span>{icon}</span>{data.label}</div>
      <p style={{ fontSize: 13, fontWeight: 800, color: BLUE[700], lineHeight: 1.4, marginBottom: 4, wordBreak: "keep-all", overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>{data.top[0].label.replace(/\s*\(\d+\)\s*$/, '')}</p>
      <p style={{ fontSize: 24, fontWeight: 900, color, marginBottom: 10, fontFamily: "'Outfit',sans-serif" }}>{(data.top[0].prob * 100).toFixed(1)}%</p>
    </div>
  );
};

const FaultBox = ({ label, pct, role, color, colorLight }) => (
  <div style={{ textAlign: "center", padding: "20px 12px", borderRadius: 14, background: `linear-gradient(135deg,${colorLight},#FFF)`, border: `1px solid ${color}30` }}>
    <p style={{ fontSize: 13, color: "#8892B0", fontWeight: 500, marginBottom: 8 }}>{label}</p>
    <p style={{ fontSize: 48, fontWeight: 900, color, lineHeight: 1, fontFamily: "'Outfit',sans-serif" }}>{pct}%</p>
    <p style={{ fontSize: 13, fontWeight: 700, color, marginTop: 8 }}>{role}</p>
  </div>
);

const Page5 = ({ onBack, onHome, videoData }) => {
  const [status, setStatus] = useState("analyzing");
  const [statusMsg, setStatusMsg] = useState("서버에 영상 전송 중...");
  const [progress, setProgress] = useState(0);
  const [apiResult, setApiResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");
  const [expandAlts, setExpandAlts] = useState(false);
  const [expandModels, setExpandModels] = useState(false);
  const [expandFault, setExpandFault] = useState(false);  // ✅ 모델별 완료 상태 (SSE로 개별 추적)
  const [modelDone, setModelDone] = useState([false, false, false, false]);
  const [vlmReport, setVlmReport] = useState(null);
  const [vlmLoading, setVlmLoading] = useState(false);

  const generateVlm = async () => {
    setVlmLoading(true);
    setVlmReport(null);
    try {
      // 모델 결과를 기반으로 VLM 리포트 생성 (더미)
      await new Promise(r => setTimeout(r, 2000));

      const place = apiResult?.models?.model1?.top?.[0]?.label || "알 수 없음";
      const type = apiResult?.models?.model2?.top?.[0]?.label || "알 수 없음";
      const carA = apiResult?.models?.model3?.top?.[0]?.label || "알 수 없음";
      const carB = apiResult?.models?.model4?.top?.[0]?.label || "알 수 없음";
      const fa = apiResult?.fault?.fa;
      const fb = apiResult?.fault?.fb;

      const vlm = apiResult?.vlm_report;

      const templates = [
        vlm
      ];

      const picked = templates[Math.floor(Math.random() * templates.length)];
      setVlmReport(picked);
    } catch (err) {
      console.error("VLM 생성 실패:", err);
    } finally {
      setVlmLoading(false);
    }
  };

  useEffect(() => {
    if (!videoData?.file) { setStatus("error"); setErrorMsg("영상 파일이 없습니다"); return; }

    setStatus("analyzing");
    setApiResult(null);
    setErrorMsg("");
    setExpandAlts(false);
    setProgress(0);
    setModelDone([false, false, false, false]);

    const controller = new AbortController();

    const callApi = async () => {
      try {
        setStatusMsg("서버에 영상 전송 중...");
        setProgress(0);

        const formData = new FormData();
        formData.append("video", videoData.file);

        const res = await fetch(`${API_URL}/api/analyze`, {
          method: "POST",
          body: formData,
          signal: controller.signal,
        });

        if (!res.ok) {
          const errText = await res.text();
          throw new Error(errText || `서버 오류 (${res.status})`);
        }

        // ✅ SSE 스트림 읽기
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            try {
              const evt = JSON.parse(jsonStr);

              if (evt.type === "converting") {
                setStatusMsg("영상 코덱 변환 중...");
              }

              if (evt.type === "model_start") {
                const idx = evt.model_index;
                const msgs = ["장소/배경 분석 중...", "사고유형 분석 중...", "차량 A 분석 중...", "차량 B 분석 중..."];
                // 첫 모델은 바로, 이후는 0.8초 뒤에 메시지 변경 (완료! 메시지 보이도록)
                if (idx === 0) {
                  setStatusMsg(`모델 ${idx + 1}/4: ${msgs[idx]}`);
                  setProgress(idx * 25);
                } else {
                  setTimeout(() => {
                    setStatusMsg(`모델 ${idx + 1}/4: ${msgs[idx]}`);
                    setProgress(idx * 25);
                  }, 800);
                }
              }

              if (evt.type === "model_done") {
                const idx = evt.model_index;
                const labels = ["장소/배경", "사고유형", "차량 A", "차량 B"];
                setModelDone(prev => {
                  const next = [...prev];
                  next[idx] = true;
                  return next;
                });
                setProgress((idx + 1) * 25);
                setStatusMsg(`모델 ${idx + 1}/4: ${labels[idx]} 분석 완료 ✓`);
              }

              if (evt.type === "complete") {
                setProgress(100);
                setStatusMsg("분석 완료!");
                setModelDone([true, true, true, true]);
                setApiResult({
                  models: evt.models,
                  fault: evt.fault,
                  alt_faults: evt.alt_faults,
                });
                setTimeout(() => setStatus("done"), 400);
              }

              if (evt.type === "error") {
                throw new Error(evt.error || "서버 오류");
              }
            } catch (parseErr) {
              if (parseErr.message && !parseErr.message.includes("JSON")) {
                throw parseErr;
              }
            }
          }
        }
      } catch (err) {
        if (err.name === "AbortError") return;
        console.error("API 호출 실패:", err);
        setStatus("error");
        setErrorMsg(err.message || "서버 연결 실패");
      }
    };

    callApi();
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoData?.sig]);

  const modelResults = apiResult
    ? MODEL_KEYS.map((k) => apiResult.models?.[k] || null)
    : [];
  const fault = apiResult?.fault;
  const altFaults = apiResult?.alt_faults || [];

  return (
    <Phone>
      <NavBar title="분석 결과" onBack={onBack} step={3} />
      <div style={{ padding: "16px 24px 60px" }}>

        {/* ═══ 분석 중 ═══ */}
        {status === "analyzing" && (
          <div className="fade-in" style={{ textAlign: "center", paddingTop: 80 }}>
            <div style={{ display: "flex", justifyContent: "center", gap: 16, marginBottom: 32 }}>
              {MODEL_ICONS.map((ic, i) => (
                <div key={i} style={{ width: 48, height: 48, borderRadius: 14, background: modelDone[i] ? `${MODEL_COLORS[i]}20` : "#F7FAFC", border: `2px solid ${modelDone[i] ? MODEL_COLORS[i] : "#E2E8F0"}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, transition: "all .4s ease", position: "relative" }}>
                  {ic}
                  {modelDone[i] && <div style={{ position: "absolute", top: -4, right: -4, width: 16, height: 16, borderRadius: "50%", background: ACCENT.green, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: "#FFF" }}>✓</div>}
                </div>
              ))}
            </div>
            <div style={{ height: 6, borderRadius: 3, background: "#EDF2F7", overflow: "hidden", maxWidth: 260, margin: "0 auto" }}>
              <div style={{ height: "100%", borderRadius: 3, background: `linear-gradient(90deg,${BLUE[400]},${BLUE[300]})`, width: `${progress}%`, transition: "width .5s ease" }} />
            </div>
            <p style={{ fontSize: 15, fontWeight: 600, color: BLUE[600], marginTop: 16 }}>{statusMsg}</p>
            <div style={{ width: 40, height: 40, margin: "20px auto 0", border: `3px solid ${BLUE[100]}`, borderTopColor: BLUE[500], borderRadius: "50%", animation: "spin .8s linear infinite" }} />
          </div>
        )}

        {/* ═══ 에러 ═══ */}
        {status === "error" && (
          <div className="fade-in" style={{ textAlign: "center", paddingTop: 80 }}>
            <div style={{ width: 64, height: 64, margin: "0 auto 16px", borderRadius: "50%", background: ACCENT.redLight, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 32 }}>❌</div>
            <p style={{ fontSize: 18, fontWeight: 700, color: BLUE[700] }}>분석 실패</p>
            <p style={{ fontSize: 14, color: "#8892B0", marginTop: 8, lineHeight: 1.6 }}>{errorMsg}</p>
            <div style={{ marginTop: 16, padding: "14px 20px", borderRadius: 12, background: "#FFF5F5", border: "1px solid #FED7D7", textAlign: "left" }}>
              <p style={{ fontSize: 13, fontWeight: 700, color: ACCENT.red, marginBottom: 8 }}>확인사항:</p>
              <p style={{ fontSize: 12, color: "#4A5568", lineHeight: 1.8 }}>
                1. 터미널에서 <code style={{ background: "#EDF2F7", padding: "2px 6px", borderRadius: 4 }}>python backend.py</code> 실행 중인지 확인<br />
                2. http://localhost:5002/api/health 접속 확인<br />
                3. 모델 파일 4개가 ~/Downloads/모델에 있는지 확인
              </p>
            </div>
            <div style={{ marginTop: 24 }}>
              <PrimaryBtn onClick={onHome} icon="🏠">처음으로</PrimaryBtn>
            </div>
          </div>
        )}

        {/* ═══ 결과 표시 ═══ */}
        {status === "done" && apiResult && (
          <>
            {/* ── 분석 영상 미리보기 ── */}
            <SectionHeader icon="🎬" text={videoData?.isTrimmed ? "분석 영상" : "분석 영상"} color={BLUE[300]} />
            <div className="fade-up">
              <CustomVideoPlayer
                src={videoData?.url}
                trimStart={videoData?.trimStart || 0}
                trimEnd={videoData?.trimEnd || videoData?.duration || 10}
                isTrimmed={videoData?.isTrimmed || false}
              />
              {videoData?.isTrimmed && (
                <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
                  <Badge color={ACCENT.orange} bg="#FFF8EB">✂️ {(videoData.trimEnd - videoData.trimStart).toFixed(1)}초 클립</Badge>
                  <Badge color={BLUE[500]}>원본 {videoData.trimStart.toFixed(1)}초 ~ {videoData.trimEnd.toFixed(1)}초</Badge>
                </div>
              )}
            </div>

            {fault && (
              <div className="fade-up">
                <SectionHeader icon="⚖️" text="과실비율 산정 결과" color={ACCENT.red} />
                <div style={{ borderRadius: 18, background: "#F7FBFF", border: "1px solid #E2E8F0", padding: "22px 18px", boxShadow: "0 2px 16px rgba(0,0,0,.04)" }}>

                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                    <FaultBox label="차량 A 과실" pct={fault.fa} role={fault.role_a} color={ACCENT.red} colorLight={ACCENT.redLight} />
                    <FaultBox label="차량 B 과실" pct={fault.fb} role={fault.role_b} color={BLUE[400]} colorLight={BLUE[50]} />
                  </div>

                  {altFaults.length > 0 && (
                    <div>
                      <button onClick={() => setExpandAlts(!expandAlts)} style={{ width: "100%", marginTop: 14, padding: "12px 16px", borderRadius: 12, border: "1px solid #E2E8F0", background: "#FFF", cursor: "pointer", fontFamily: "inherit", fontSize: 14, fontWeight: 600, color: BLUE[500], display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
                        🔎 다른 가능성 보기 ({altFaults.length}건) <span style={{ transform: expandAlts ? "rotate(180deg)" : "rotate(0)", transition: "transform .3s ease", display: "inline-block" }}>▾</span>
                      </button>
                      {expandAlts && (
                        <div className="fade-up" style={{ marginTop: 12 }}>
                          {altFaults.map((alt, i) => (
                            <div key={i} style={{ marginTop: i > 0 ? 12 : 0, padding: "14px 16px", borderRadius: 14, background: "#FAFCFF", border: "1px solid #EDF2F7" }}>
                              <p style={{ fontSize: 12, color: "#4A5568", margin: "8px 0 10px", lineHeight: 1.6 }}>{alt.desc}</p>
                              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                                <div style={{ textAlign: "center", padding: "10px 8px", borderRadius: 10, background: ACCENT.redLight }}>
                                  <p style={{ fontSize: 11, color: "#8892B0" }}>내 과실 (A)</p>
                                  <p style={{ fontSize: 26, fontWeight: 900, color: ACCENT.red, fontFamily: "'Outfit',sans-serif" }}>{alt.fa}%</p>
                                </div>
                                <div style={{ textAlign: "center", padding: "10px 8px", borderRadius: 10, background: BLUE[50] }}>
                                  <p style={{ fontSize: 11, color: "#8892B0" }}>상대 과실 (B)</p>
                                  <p style={{ fontSize: 26, fontWeight: 900, color: BLUE[400], fontFamily: "'Outfit',sans-serif" }}>{alt.fb}%</p>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}

            {!fault && (
              <div style={{ marginTop: 24, padding: "18px 20px", borderRadius: 14, background: ACCENT.orangeLight, border: `1px solid ${ACCENT.orange}30` }}>
                <p style={{ fontSize: 14, fontWeight: 700, color: "#C05621" }}>⚠️ 과실비율 매칭 실패</p>
                <p style={{ fontSize: 13, color: "#744210", marginTop: 6, lineHeight: 1.6 }}>DB에서 정확히 일치하는 조합을 찾지 못했습니다. CSV 파일이 ~/Downloads에 있는지 확인해주세요.</p>
              </div>
            )}

            {/* ── AI 분석 결과 (토글) ── */}
            <div style={{ marginTop: 12 }}>
              <button onClick={() => setExpandModels(!expandModels)} style={{ width: "100%", padding: "14px 18px", borderRadius: 14, border: "1px solid #E2E8F0", background: "#F7FBFF", cursor: "pointer", fontFamily: "inherit", fontSize: 15, fontWeight: 700, color: BLUE[600], display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
                📋 AI 모델별 분석 결과 <span style={{ transform: expandModels ? "rotate(180deg)" : "rotate(0)", transition: "transform .3s ease", display: "inline-block", fontSize: 14 }}>▾</span>
              </button>
              {expandModels && (
                <div className="fade-up" style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, width: "100%" }}>
                  {modelResults.map((d, i) => (
                    <ResultCard key={i} data={d} index={i} visible={true} />
                  ))}
                </div>
              )}
            </div>


            {/* ── VLM 리포트 ── */}
            <div style={{ marginTop: 24 }}>
              {!vlmReport && !vlmLoading && (
                <button onClick={generateVlm} style={{ width: "100%", padding: "14px 18px", borderRadius: 14, border: "none", background: `linear-gradient(135deg, ${ACCENT.purple}, ${BLUE[400]})`, cursor: "pointer", fontFamily: "inherit", fontSize: 15, fontWeight: 700, color: "#FFF", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, boxShadow: "0 4px 16px rgba(183,148,244,.3)" }}>
                  AI 영상 분석 리포트 생성하기
                </button>
              )}
              {vlmLoading && (
                <div style={{ textAlign: "center", padding: "20px", borderRadius: 14, background: "#FAF5FF", border: "1px solid #E9D8FD" }}>
                  <div style={{ width: 36, height: 36, margin: "0 auto 10px", border: `3px solid #E9D8FD`, borderTopColor: ACCENT.purple, borderRadius: "50%", animation: "spin .8s linear infinite" }} />
                  <p style={{ fontSize: 14, fontWeight: 600, color: ACCENT.purple }}>VLM 리포트 생성 중...</p>
                </div>
              )}
              {vlmReport && (
                <div className="fade-up" style={{ borderRadius: 16, background: "#FAF5FF", border: "1px solid #E9D8FD", padding: "18px 16px" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14, paddingBottom: 10, borderBottom: "1px solid #E9D8FD" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 18 }}>📝</span>
                      <span style={{ fontSize: 15, fontWeight: 800, color: BLUE[700] }}>AI 영상 분석 리포트</span>
                    </div>
                    <Badge color={ACCENT.purple} bg="#F3E8FF">VLM</Badge>
                  </div>
                  {vlmReport.map((sentence, i) => (
                    <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", padding: "10px 0", borderTop: i > 0 ? "1px solid #F3E8FF" : "none" }}>
                      <div style={{ width: 26, height: 26, borderRadius: 8, background: "#F3E8FF", border: "1px solid #E9D8FD", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 800, color: ACCENT.purple, flexShrink: 0 }}>{i + 1}</div>
                      <p style={{ flex: 1, fontSize: 14, color: "#4A5568", lineHeight: 1.7, wordBreak: "keep-all" }}>{sentence}</p>
                    </div>
                  ))}
                  <button onClick={generateVlm} style={{ width: "100%", marginTop: 14, padding: "12px 16px", borderRadius: 12, border: "1px solid #E9D8FD", background: "#FFF", cursor: "pointer", fontFamily: "inherit", fontSize: 14, fontWeight: 600, color: ACCENT.purple, display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
                    다른 결과 생성하기
                  </button>
                </div>
              )}
            </div>

            <div style={{ marginTop: 32 }}>
              <button onClick={onHome} style={{ width: "100%", height: 52, borderRadius: 14, border: `2px solid ${BLUE[300]}`, background: "#FFF", cursor: "pointer", fontFamily: "inherit", fontSize: 15, fontWeight: 700, color: BLUE[500], display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>🏠 처음으로</button>
            </div>

          </>
        )}


      </div>
    </Phone>
  );
};

/* ───────── APP ───────── */
export default function App() {
  const [page, setPage] = useState(1);
  const [videoData, setVideoData] = useState(null);
  useEffect(() => { if (!document.getElementById("ai-muncheol-css")) { const s = document.createElement("style"); s.id = "ai-muncheol-css"; s.textContent = GLOBAL_CSS; document.head.appendChild(s); } }, []);
  const goHome = () => { setPage(1); setVideoData(null); };
  const goToUpload = () => { setPage(2); setVideoData(null); };
  switch (page) {
    case 1: return <Page1 onNext={() => setPage(2)} />;
    case 2: return <Page2 onBack={() => setPage(1)} onNext={(skip) => setPage(skip ? 4 : 3)} setVideoData={setVideoData} />;
    case 3: return <Page3 onBack={goToUpload} onNext={() => setPage(4)} videoData={videoData} setVideoData={setVideoData} />;
    case 4: return <Page4 onBack={() => setPage(videoData?.duration > 10 ? 3 : 2)} onNext={() => setPage(5)} videoData={videoData} />;
    case 5: return <Page5 key={videoData?.sig || "no-sig"} onBack={() => setPage(4)} onHome={goHome} videoData={videoData} />;
    default: return <Page1 onNext={() => setPage(2)} />;
  }
}