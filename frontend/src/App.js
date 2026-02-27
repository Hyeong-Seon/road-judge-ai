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
  <div className="phone-container" style={{ maxWidth: 430, margin: "0 auto", minHeight: "100dvh", background: "#FFFFFF", position: "relative", overflow: "hidden", boxShadow: "0 0 80px rgba(26,54,93,.08)" }}>{children}</div>
);
const StepDots = ({ current, total = 5 }) => (
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
const Page1 = ({ onNext, bigFont, setBigFont }) => {
  const [show, setShow] = useState(false);
  useEffect(() => { setTimeout(() => setShow(true), 100); }, []);
  return (
    <Phone>
      <div style={{ position: "absolute", top: 20, right: 16, zIndex: 100 }}>
        <button onClick={() => setBigFont(!bigFont)} style={{ background: bigFont ? BLUE[500] : "#F7FBFF", color: bigFont ? "#FFF" : BLUE[500], border: `1.5px solid ${BLUE[300]}`, borderRadius: 10, padding: "8px 14px", fontSize: 13, fontWeight: 700, cursor: "pointer", fontFamily: "inherit", display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 400 }}>가</span>
          <span style={{ fontSize: 10, color: bigFont ? "#FFF" : "#A0AEC0" }}>→</span>
          <span style={{ fontSize: 20, fontWeight: 900 }}>가</span>
          <span style={{ fontSize: 10, marginLeft: 4, padding: "2px 6px", borderRadius: 6, background: bigFont ? "rgba(255,255,255,.25)" : `${BLUE[500]}15`, fontWeight: 800 }}>{bigFont ? "ON" : "OFF"}</span>
        </button>
      </div>
      <div style={{ minHeight: "100dvh", display: "flex", flexDirection: "column", justifyContent: "space-between", padding: "0 24px", background: "linear-gradient(180deg,#FFFFFF 0%,#F0F7FF 100%)", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: -80, right: -60, width: 260, height: 260, borderRadius: "50%", background: `radial-gradient(circle,${BLUE[100]}60,transparent 70%)`, animation: "float 6s ease-in-out infinite" }} />
        <div style={{ position: "absolute", bottom: 120, left: -80, width: 200, height: 200, borderRadius: "50%", background: `radial-gradient(circle,${BLUE[50]}80,transparent 70%)`, animation: "float 8s ease-in-out infinite 1s" }} />
        <div style={{ flex: 1 }} />
        <div style={{ textAlign: "center", opacity: show ? 1 : 0, transform: show ? "translateY(0)" : "translateY(30px)", transition: "all .8s cubic-bezier(.22,1,.36,1)" }}>
          <div style={{ width: 220, height: 220, margin: "0 auto 28px", borderRadius: 70, background: `linear-gradient(135deg,${BLUE[500]},${BLUE[300]})`, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: `0 12px 40px ${BLUE[300]}50`, position: "relative", overflow: "hidden" }}>
            <img src="/newlogo.png" alt="AI 문철 로고" style={{ width: 350, height: 350, objectFit: "contain", display: "block", borderRadius: "30%" }} />
          </div>
          <h1 style={{ fontSize: 42, fontWeight: 900, letterSpacing: -1.5, color: BLUE[700], fontFamily: "'Outfit','Noto Sans KR',sans-serif" }}>AI 문철</h1>
          <p style={{ fontSize: 15, fontWeight: 220, letterSpacing: -1.5, color: BLUE[500], fontFamily: "'Outfit','Noto Sans KR',sans-serif" }}>한문철의 후계자 3명에게 도움을 받아보세요</p>
        </div>
        <div style={{ flex: 1.2 }} />
        <div style={{ paddingBottom: 48, opacity: show ? 1 : 0, transform: show ? "translateY(0)" : "translateY(20px)", transition: "all .8s cubic-bezier(.22,1,.36,1) .3s" }}>
          <p style={{ fontSize: 11, color: "#A0AEC0", textAlign: "center", lineHeight: 1.6, marginBottom: 16 }}>본 서비스의 AI 분석 결과는 참고용으로만 제공되며, 법적 효력 및 <br />증거 능력이 없습니다. 정확한 과실비율 판정은 보험사·법원 등 <br />전문기관의 판단을 따르시기 바랍니다.</p>
          <PrimaryBtn onClick={onNext}>분석 시작하기</PrimaryBtn>
        </div>
      </div>
    </Phone>
  );
};

/* ───────── PAGE 2 ───────── */
const Page2 = ({ onNext, onBack, setVideoData }) => {
  const [file, setFile] = useState(null); const [preview, setPreview] = useState(null); const [duration, setDuration] = useState(null); const [converting, setConverting] = useState(false); const [videoError, setVideoError] = useState(false); const inputRef = useRef(); const videoRef = useRef();
  const handleFile = (e) => { const f = e.target.files?.[0]; if (!f) return; setFile(f); setPreview(URL.createObjectURL(f)); setVideoError(false); setConverting(false); setDuration(null); };
  const handleLoadedMeta = () => { if (videoRef.current) { const d = videoRef.current.duration; setDuration(isFinite(d) ? d : null); } };
  const handleVideoError = async () => { setVideoError(true); if (!file || converting) return; setConverting(true); try { const fd = new FormData(); fd.append("video", file); const res = await fetch(`${API_URL}/api/convert`, { method: "POST", body: fd }); if (res.ok) { const blob = await res.blob(); setPreview(URL.createObjectURL(blob)); setVideoError(false); } } catch (err) { console.error("변환 실패:", err); } finally { setConverting(false); } };
  const handleNext = () => { if (!file || !preview) return; const d = duration || 10; const sig = `${file.name}_${file.size}_${Date.now()}`; setVideoData({ file, url: preview, duration: d, isTrimmed: false, trimStart: 0, trimEnd: d, sig }); onNext(d <= 10); };
  return (
    <Phone>
      <NavBar title="영상 업로드" onBack={onBack} step={0} />
      <div style={{ padding: "20px 24px 40px", minHeight: "calc(100dvh - 100px)", display: "flex", flexDirection: "column" }}>
        <div className="fade-up"><h2 style={{ fontSize: 20, fontWeight: 800, color: BLUE[700] }}>분석할 블랙박스 영상을<br />업로드해 주세요</h2><p style={{ fontSize: 14, color: "#8892B0", marginTop: 8 }}>MP4, AVI, MOV 형식을 지원합니다</p></div>
        <div className="fade-up" style={{ animationDelay: ".1s", marginTop: 28, flex: 1 }}>
          {!file ? (
            <div onClick={() => inputRef.current?.click()} style={{ border: `2px dashed ${BLUE[300]}`, borderRadius: 20, padding: "52px 24px", background: "#F7FBFF", textAlign: "center", cursor: "pointer" }}>
              <div style={{ width: 64, height: 64, margin: "0 auto 16px", borderRadius: 20, background: `linear-gradient(135deg,${BLUE[50]},${BLUE[100]})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 32 }}>☁️</div>
              <p style={{ fontSize: 16, fontWeight: 700, color: BLUE[500] }}>터치하여 영상 선택</p><p style={{ fontSize: 13, color: "#A0AEC0", marginTop: 6 }}>또는 파일 앱에서 가져오기</p>
            </div>
          ) : (
            <div className="scale-in" style={{ border: `2px solid ${BLUE[300]}`, borderRadius: 20, padding: 16, background: BLUE[50] }}>
              {converting ? (<div style={{ width: "100%", padding: "40px 20px", borderRadius: 12, background: "#1A202C", textAlign: "center" }}><div style={{ width: 40, height: 40, margin: "0 auto 12px", border: `3px solid ${BLUE[100]}`, borderTopColor: BLUE[500], borderRadius: "50%", animation: "spin .8s linear infinite" }} /><p style={{ fontSize: 14, fontWeight: 600, color: "#FFF" }}>영상 변환 중...</p><p style={{ fontSize: 12, color: "#A0AEC0", marginTop: 6 }}>브라우저 미리보기를 위해 코덱을 변환하고 있습니다</p></div>
              ) : (<video ref={videoRef} src={preview} onLoadedMetadata={handleLoadedMeta} onError={handleVideoError} controls playsInline style={{ width: "100%", borderRadius: 12, background: "#000" }} />)}
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 12 }}>
                <div style={{ width: 32, height: 32, borderRadius: 8, background: "#C6F6D5", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16 }}>✓</div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p className="filename-text" style={{ fontSize: 14, fontWeight: 600, color: BLUE[700], overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</p>
                  {duration ? <p style={{ fontSize: 12, color: "#8892B0", marginTop: 2 }}>영상 길이: {duration.toFixed(1)}초</p> : converting ? <p style={{ fontSize: 12, color: ACCENT.orange, marginTop: 2 }}>⚡ 코덱 변환 중...</p> : videoError ? <p style={{ fontSize: 12, color: ACCENT.orange, marginTop: 2 }}>⚡ 변환 실패 — 분석은 서버에서 자동 변환됩니다</p> : null}
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
  const dur = videoData?.duration || 30; const [accidentTime, setAccidentTime] = useState(Math.min(dur / 2, dur)); const [trimming, setTrimming] = useState(false); const [trimDone, setTrimDone] = useState(false);
  const start = Math.max(0, accidentTime - 5); const end = Math.min(dur, accidentTime + 5);
  const handleTrim = () => { setTrimming(true); setTimeout(() => { setVideoData(prev => ({ ...prev, isTrimmed: true, trimStart: start, trimEnd: end })); setTrimming(false); setTrimDone(true); setTimeout(() => onNext(), 600); }, 1500); };
  return (
    <Phone>
      <NavBar title="사고 구간 설정" onBack={onBack} step={1} />
      <div style={{ padding: "16px 24px 40px", minHeight: "calc(100dvh - 100px)", display: "flex", flexDirection: "column" }}>
        <SectionHeader icon="🎬" text="원본 영상" />
        <div className="fade-up" style={{ borderRadius: 14, overflow: "hidden", boxShadow: "0 4px 20px rgba(0,0,0,.08)", background: "#000" }}><video src={videoData?.url} controls playsInline style={{ width: "100%", display: "block" }} /></div>
        <div style={{ marginTop: 8 }}><Badge color={BLUE[500]}>전체 {dur.toFixed(1)}초</Badge></div>
        <SectionHeader icon="✂️" text="사고 시점 선택" color={ACCENT.orange} />
        <div className="fade-up" style={{ animationDelay: ".15s" }}>
          <p style={{ fontSize: 14, fontWeight: 600, color: BLUE[700], marginBottom: 16 }}>사고 발생 시점을 선택해 주세요</p>
          <div style={{ position: "relative", padding: "28px 0 12px" }}>
            <div style={{ position: "absolute", top: 0, left: `calc(${(accidentTime / dur) * 100}% - 28px)`, background: BLUE[500], color: "#fff", borderRadius: 8, padding: "3px 10px", fontSize: 12, fontWeight: 700, transition: "left .15s ease", whiteSpace: "nowrap", zIndex: 2 }}>{accidentTime.toFixed(1)}초<div style={{ position: "absolute", bottom: -4, left: "50%", transform: "translateX(-50%) rotate(45deg)", width: 8, height: 8, background: BLUE[500] }} /></div>
            <div style={{ position: "relative", height: 8, borderRadius: 4, background: "#E2E8F0" }}><div style={{ position: "absolute", left: `${(start / dur) * 100}%`, width: `${((end - start) / dur) * 100}%`, height: "100%", borderRadius: 4, background: `linear-gradient(90deg,${BLUE[300]},${BLUE[400]})` }} /></div>
            <input type="range" min={0} max={dur} step={0.5} value={accidentTime} onChange={e => setAccidentTime(Number(e.target.value))} style={{ position: "absolute", top: 24, left: 0, width: "100%", height: 16, opacity: 0, cursor: "pointer" }} />
          </div>
        </div>
        {trimming && (<div className="fade-in" style={{ textAlign: "center", padding: "20px", marginTop: 20 }}><div style={{ width: 40, height: 40, margin: "0 auto 12px", border: `3px solid ${BLUE[100]}`, borderTopColor: BLUE[500], borderRadius: "50%", animation: "spin .8s linear infinite" }} /><p style={{ fontSize: 15, fontWeight: 600, color: BLUE[600] }}>영상 자르는 중...</p></div>)}
        {!trimming && trimDone && (<div className="scale-in" style={{ textAlign: "center", padding: "16px 0", marginTop: 20 }}><div style={{ width: 48, height: 48, margin: "0 auto 8px", borderRadius: "50%", background: "#C6F6D5", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24 }}>✓</div><p style={{ fontSize: 15, fontWeight: 700, color: ACCENT.green }}>자르기 완료!</p></div>)}
        <div style={{ marginTop: "auto", paddingTop: 20 }}><PrimaryBtn onClick={handleTrim} disabled={trimming || trimDone}>{trimming ? "자르는 중..." : trimDone ? "자르기 완료" : "영상 자르기"}</PrimaryBtn></div>
      </div>
    </Phone>
  );
};

/* ───────── PAGE 4 ───────── */
const Page4 = ({ onNext, onBack, videoData }) => {
  const dur = videoData?.duration || 10; const isTrimmed = videoData?.isTrimmed || false; const trimStart = videoData?.trimStart || 0; const trimEnd = videoData?.trimEnd || dur; const clipDuration = isTrimmed ? (trimEnd - trimStart) : dur;
  return (
    <Phone>
      <NavBar title="영상 확인" onBack={onBack} step={2} />
      <div style={{ padding: "16px 24px 40px", minHeight: "calc(100dvh - 100px)", display: "flex", flexDirection: "column" }}>
        <SectionHeader icon="🎬" text={isTrimmed ? "편집된 영상" : "분석 대상 영상"} />
        <div className="fade-up"><CustomVideoPlayer src={videoData?.url} trimStart={trimStart} trimEnd={trimEnd} isTrimmed={isTrimmed} /></div>
        {!isTrimmed && <div style={{ marginTop: 10 }}><Badge color={BLUE[500]}>원본 영상 ({dur.toFixed(1)}초)</Badge></div>}
        <p style={{ fontSize: 13, color: "#8892B0", marginTop: 15 }}>{isTrimmed ? `원본 ${trimStart.toFixed(1)}초~${trimEnd.toFixed(1)}초 구간 (${clipDuration.toFixed(1)}초)` : "아래 영상으로 AI 분석이 진행됩니다"}</p>
        <div className="fade-up" style={{ animationDelay: ".15s", marginTop: 24, padding: "18px 20px", borderRadius: 14, background: "#F7FBFF", borderLeft: `4px solid ${BLUE[300]}` }}>
          <p style={{ fontSize: 14, fontWeight: 700, color: BLUE[700], marginBottom: 12 }}>3개의 AI모델로 다음 4개의 항목을 분석합니다</p>
          {MODEL_LABELS.map((label, i) => (<div key={i} style={{ display: "flex", alignItems: "center", gap: 10, padding: "7px 0", fontSize: 14, color: "#4A5568" }}><span style={{ width: 24, height: 24, borderRadius: 7, background: `${MODEL_COLORS[i]}18`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, fontWeight: 800, color: MODEL_COLORS[i] }}>{i + 1}</span><span style={{ fontWeight: 500 }}>{label}</span></div>))}
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ paddingTop: 24 }}><PrimaryBtn onClick={onNext}>AI 정밀 분석 시작</PrimaryBtn></div>
      </div>
    </Phone>
  );
};

/* ═══════════════════════════════════════════════════════
   SHARED DATA / HELPERS (Page5 & Page6)
   ═══════════════════════════════════════════════════════ */

const LABEL_MAP_PLACE = { 0: "직선 도로", 1: "신호 없는 교차로", 2: "신호 있는 교차로", 3: "T자형 도로", 4: "기타 도로", 5: "주차장", 6: "회전 교차로", 13: "고속도로" };

const ANALYSTS = [
  { name: "엄도식", icon: "🤖", key: "형선", faultKey: "형선", vlmKey: "형선" },
  { name: "민다정", icon: "💗", key: "은석", faultKey: "은석", vlmKey: "은석" },
  { name: "윤 슬", icon: "💫", key: "c3d", faultKey: "c3d", vlmKey: "수민", short: "C3D" },
];

const MODEL_FIELD_MAP = [
  { codeKey: "accident_place", probKey: "probability", labelMap: LABEL_MAP_PLACE, labelMapsKey: "place" },
  { codeKey: "accident_place_feature_code", probKey: "probability", labelMap: null, labelMapsKey: "type" },
  { codeKey: "vehicle_a_code", probKey: "prob", labelMap: null, labelMapsKey: "action" },
  { codeKey: "vehicle_b_code", probKey: "prob", labelMap: null, labelMapsKey: "action" },
];

const transformGroupData = (groupArr, labelMaps) => {
  if (!groupArr || groupArr.length < 4) return {};
  const result = {};
  for (let i = 0; i < 4; i++) {
    const raw = groupArr[i]; const field = MODEL_FIELD_MAP[i];
    if (!raw || raw.length === 0) { result[`model${i + 1}`] = null; continue; }
    const top = raw.map(item => {
      const code = item[field.codeKey]; const prob = item[field.probKey] || 0;
      let label = null;
      if (field.labelMap && field.labelMap[code] !== undefined) label = field.labelMap[code];
      if (!label && labelMaps && field.labelMapsKey) {
        const bMap = labelMaps[field.labelMapsKey];
        if (bMap) label = bMap[String(code)] || bMap[code] || null;
      }
      return { label: label || `코드 ${code}`, prob };
    });
    result[`model${i + 1}`] = { label: MODEL_LABELS[i], top };
  }
  return result;
};

const MODEL_KEYS = ["model1", "model2", "model3", "model4"];

const ResultCard = ({ data, index, visible }) => {
  const color = MODEL_COLORS[index]; const icon = MODEL_ICONS[index];
  if (!data || !data.top || data.top.length === 0) return null;
  return (
    <div style={{ background: "#FFF", border: "1px solid #EDF2F7", borderRadius: 16, padding: "16px 12px", boxShadow: "0 2px 12px rgba(0,0,0,.04)", minWidth: 0, overflow: "hidden", opacity: visible ? 1 : 0, transform: visible ? "translateY(0)" : "translateY(20px)", transition: `all .5s cubic-bezier(.22,1,.36,1) ${index * 0.1}s` }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: "#8892B0", letterSpacing: 1.5, paddingBottom: 8, borderBottom: `2px solid ${color}`, marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}><span>{icon}</span>{data.label}</div>
      <p className="result-card-label" style={{ fontSize: 13, fontWeight: 800, color: BLUE[700], lineHeight: 1.4, marginBottom: 4, wordBreak: "keep-all", overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>{data.top[0].label.replace(/\s*\(\d+\)\s*$/, '')}</p>
    </div>
  );
};

const FaultBox = ({ label, pct, role, color, colorLight }) => (
  <div style={{ textAlign: "center", padding: "20px 12px", borderRadius: 14, background: `linear-gradient(135deg,${colorLight},#FFF)`, border: `1px solid ${color}30` }}>
    <p style={{ fontSize: 13, color: "#8892B0", fontWeight: 500, marginBottom: 8 }}>{label}</p>
    <p className="fault-pct" style={{ fontSize: 48, fontWeight: 900, color, lineHeight: 1, fontFamily: "'Outfit',sans-serif" }}>{pct}%</p>
    <p style={{ fontSize: 13, fontWeight: 700, color, marginTop: 8 }}>{role}</p>
  </div>
);

/* ── 분석 중 개별 진행바 row ── */
const SUB_LABELS = ["장소", "유형", "차A", "차B"];
const ProgressRow = ({ icon, name, percent, msg, done, active, color, currentSub = -1, showSubs = true }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderRadius: 14, background: done ? `${color}08` : active ? "#FAFCFF" : "#F7FAFC", border: `1.5px solid ${done ? color : active ? BLUE[300] : "#E2E8F0"}`, transition: "all .4s ease" }}>
    <div style={{ width: 40, height: 40, borderRadius: 12, background: done ? `${color}15` : "#F0F4F8", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, position: "relative", flexShrink: 0 }}>
      {icon}
      {done && <div style={{ position: "absolute", top: -3, right: -3, width: 16, height: 16, borderRadius: "50%", background: ACCENT.green, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: "#FFF" }}>✓</div>}
    </div>
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontSize: 14, fontWeight: 700, color: done ? color : BLUE[700] }}>{name}</span>
        <span style={{ fontSize: 12, fontWeight: 700, color: done ? ACCENT.green : active ? BLUE[500] : "#A0AEC0", fontFamily: "'Outfit',sans-serif" }}>{done ? "완료" : active ? `${percent}%` : "대기"}</span>
      </div>
      {showSubs && (active || done) && (
        <div style={{ display: "flex", gap: 3, marginBottom: 5 }}>
          {MODEL_ICONS.map((mi, idx) => {
            const isSubDone = done || idx < currentSub;
            const isSubCurrent = !done && idx === currentSub;
            return (
              <div key={idx} style={{
                display: "flex", alignItems: "center", gap: 2,
                padding: "2px 6px", borderRadius: 8,
                background: isSubDone ? "#F0FFF4" : isSubCurrent ? `${BLUE[400]}12` : "transparent",
                border: `1.5px solid ${isSubDone ? ACCENT.green : isSubCurrent ? BLUE[400] : "#E2E8F0"}`,
                transition: "all .3s ease",
              }}>
                <span style={{ fontSize: 11 }}>{mi}</span>
                <span style={{ fontSize: 10, fontWeight: 800, color: isSubDone ? ACCENT.green : isSubCurrent ? BLUE[500] : "#CBD5E0", minWidth: isSubCurrent ? 16 : 0 }}>
                  {isSubDone ? "✓" : isSubCurrent ? SUB_LABELS[idx] : ""}
                </span>
              </div>
            );
          })}
        </div>
      )}
      <div style={{ height: 5, borderRadius: 3, background: "#EDF2F7", overflow: "hidden" }}>
        <div style={{ height: "100%", borderRadius: 3, background: done ? ACCENT.green : `linear-gradient(90deg,${BLUE[400]},${BLUE[300]})`, width: `${percent}%`, transition: "width .5s ease" }} />
      </div>
      <p className="progress-msg" style={{ fontSize: 11, marginTop: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        {msg.includes("→") ? (
          <>
            <span style={{ color: ACCENT.green, fontWeight: 700 }}>{msg.split("→")[0]}→ </span>
            <span style={{ color: BLUE[400], fontWeight: 600 }}>{msg.split("→")[1]}</span>
          </>
        ) : (
          <span style={{ color: msg.includes("완료") ? ACCENT.green : BLUE[400], fontWeight: msg.includes("완료") ? 700 : 600 }}>{msg}</span>
        )}
      </p>
    </div>
    {active && !done && <div style={{ width: 24, height: 24, border: `2.5px solid ${BLUE[100]}`, borderTopColor: BLUE[500], borderRadius: "50%", animation: "spin .8s linear infinite", flexShrink: 0 }} />}
  </div>
);

/* ── 모델 탭 버튼 (공용) ── */
const ANALYST_COLORS = [ACCENT.orange, BLUE[800], ACCENT.purple]; // Lite, Heavy, C3D

const AnalystTabs = ({ selected, onSelect }) => (
  <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
    {ANALYSTS.map((a, i) => {
      const c = ANALYST_COLORS[i];
      return (
        <button key={i} onClick={() => onSelect(i)} style={{ flex: 1, padding: "10px 6px", borderRadius: 12, border: selected === i ? `2px solid ${c}` : "2px solid #E2E8F0", background: selected === i ? `${c}10` : "#FFF", cursor: "pointer", fontFamily: "inherit", fontSize: 13, fontWeight: selected === i ? 800 : 600, color: selected === i ? c : "#8892B0", display: "flex", flexDirection: "column", alignItems: "center", gap: 4, transition: "all .2s ease" }}>
          <span style={{ fontSize: 20 }}>{a.icon}</span><span>{a.name}</span>
        </button>
      );
    })}
  </div>
);


/* ═══════════════════════════════════════════════════════
   PAGE 5 : AI 분석 진행 중
   ═══════════════════════════════════════════════════════ */
/* 🆕 모듈 레벨 중복 호출 방지 */
let _analyzeAbort = null;

const Page5 = ({ onBack, onHome, onComplete, onVlmReady, videoData }) => {
  const [status, setStatus] = useState("analyzing");
  const [statusMsg, setStatusMsg] = useState("서버에 영상 전송 중...");
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const [uploadPhase, setUploadPhase] = useState("uploading");

  const [groupProgress, setGroupProgress] = useState({
    heavy: { percent: 0, msg: "대기 중", done: false, active: false, currentSub: -1 },
    lite: { percent: 0, msg: "대기 중", done: false, active: false, currentSub: -1 },
    c3d: { percent: 0, msg: "대기 중", done: false, active: false, currentSub: -1 },
  });

  // 🆕 콜백을 ref로 추적 → Page5 unmount 후에도 최신 콜백 호출 가능
  const onCompleteRef = useRef(onComplete);
  const onVlmReadyRef = useRef(onVlmReady);
  useEffect(() => { onCompleteRef.current = onComplete; }, [onComplete]);
  useEffect(() => { onVlmReadyRef.current = onVlmReady; }, [onVlmReady]);

  useEffect(() => {
    if (!videoData?.file) { setStatus("error"); setErrorMsg("영상 파일이 없습니다"); return; }

    // 🆕 이전 SSE가 살아있으면 abort 후 새로 시작
    if (_analyzeAbort) { _analyzeAbort.abort(); _analyzeAbort = null; }

    setStatus("analyzing"); setErrorMsg(""); setProgress(0);
    setGroupProgress({
      heavy: { percent: 0, msg: "대기 중", done: false, active: false, currentSub: -1 },
      lite: { percent: 0, msg: "대기 중", done: false, active: false, currentSub: -1 },
      c3d: { percent: 0, msg: "대기 중", done: false, active: false, currentSub: -1 },
    });
    const controller = new AbortController();
    _analyzeAbort = controller; // 🆕 모듈 레벨에 저장

    const SUB_NAMES = ["장소 / 배경", "사고 유형", "차량 A", "차량 B"];
    const detectSub = (m) => {
      if (m.includes("차량B")) return 3;
      if (m.includes("차량A")) return 2;
      if (m.includes("사고유형")) return 1;
      if (m.includes("장소")) return 0;
      return -1;
    };

    const callApi = async () => {
      try {
        setStatusMsg("서버에 영상 전송 중..."); setProgress(0); setUploadPhase("uploading");
        const fd = new FormData(); fd.append("video", videoData.file);
        const res = await fetch(`${API_URL}/api/analyze`, { method: "POST", body: fd, signal: controller.signal });
        if (!res.ok) { const t = await res.text(); throw new Error(t || `서버 오류 (${res.status})`); }
        const reader = res.body.getReader(); const decoder = new TextDecoder(); let buffer = "";

        while (true) {
          const { done, value } = await reader.read(); if (done) break;
          buffer += decoder.decode(value, { stream: true }); const lines = buffer.split("\n"); buffer = lines.pop() || "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue; const jsonStr = line.slice(6).trim(); if (!jsonStr) continue;
            try {
              const evt = JSON.parse(jsonStr);
              if (evt.type === "converting") { setStatusMsg("영상 코덱 변환 중..."); setUploadPhase("converting"); }

              if (evt.type === "progress") {
                const pct = evt.percent || 0; const msg = evt.message || "";
                setUploadPhase(prev => prev !== "analyzing" ? "analyzing" : prev);
                const subIdx = detectSub(msg);
                setProgress(pct); setStatusMsg(msg);

                setGroupProgress(prev => {
                  const next = { ...prev };

                  if (msg.includes("은석")) {
                    const prevSub = prev.heavy.currentSub;
                    let displayMsg = msg.replace("은석 ", "");
                    if (subIdx > prevSub && prevSub >= 0) {
                      displayMsg = `✓ ${SUB_NAMES[prevSub]} 완료 → ${SUB_NAMES[subIdx]} 분석 중...`;
                    } else if (subIdx >= 0) {
                      displayMsg = `${SUB_NAMES[subIdx]} 분석 중...`;
                    }
                    next.heavy = {
                      percent: Math.min(100, Math.round((pct / 45) * 100)),
                      msg: displayMsg, done: false, active: true,
                      currentSub: subIdx >= 0 ? subIdx : prevSub,
                    };
                    next.lite = { ...prev.lite, active: false };

                  } else if (msg.includes("형선")) {
                    if (!prev.heavy.done) {
                      next.heavy = { percent: 100, msg: "4개 모델 분석 완료 ✓", done: true, active: false, currentSub: 4 };
                    }
                    const prevSub = prev.lite.currentSub;
                    let displayMsg = msg.replace("형선 ", "");
                    if (subIdx > prevSub && prevSub >= 0) {
                      displayMsg = `✓ ${SUB_NAMES[prevSub]} 완료 → ${SUB_NAMES[subIdx]} 분석 중...`;
                    } else if (subIdx >= 0) {
                      displayMsg = `${SUB_NAMES[subIdx]} 분석 중...`;
                    }
                    next.lite = {
                      percent: Math.max(0, Math.min(100, Math.round(((pct - 45) / 45) * 100))),
                      msg: displayMsg, done: false, active: true,
                      currentSub: subIdx >= 0 ? subIdx : prevSub,
                    };

                  } else if (msg.includes("수민")) {
                    if (!prev.heavy.done) {
                      next.heavy = { percent: 100, msg: "4개 모델 분석 완료 ✓", done: true, active: false, currentSub: 4 };
                    }
                    if (!prev.lite.done) {
                      next.lite = { percent: 100, msg: "4개 모델 분석 완료 ✓", done: true, active: false, currentSub: 4 };
                    }
                    next.c3d = {
                      percent: Math.max(10, Math.min(90, Math.round(((pct - 80) / 10) * 100))),
                      msg: "3D CNN 분석 중...", done: false, active: true, currentSub: -1,
                    };
                  }
                  return next;
                });
              }

              // 🌟 1차: 모델 분석 완료 → 6페이지로 이동 (ref 콜백 사용)
              if (evt.type === "partial_complete") {
                setProgress(80);
                setStatusMsg("AI 모델 분석 완료! VLM 분석 준비 중...");
                setGroupProgress({
                  heavy: { percent: 100, msg: "4개 모델 분석 완료 ✓", done: true, active: false, currentSub: 4 },
                  lite: { percent: 100, msg: "4개 모델 분석 완료 ✓", done: true, active: false, currentSub: 4 },
                  c3d: { percent: 100, msg: "분석 완료 ✓", done: true, active: false, currentSub: -1 },
                });

                const partialResult = {
                  input_data: evt.input_data || null,
                  c3d_data: evt.c3d_data || null,
                  fault_results: evt.fault_results || null,
                  fault: evt.fault,
                  alt_faults: evt.alt_faults,
                  vlm_report: null,
                  label_maps: evt.label_maps || null,
                };

                setTimeout(() => onCompleteRef.current(partialResult), 600);
              }

              // 🌟 2차: VLM 스코어링 완료 → 세션 정보 전달 (ref 콜백 사용)
              if (evt.type === "vlm_ready") {
                setProgress(100);
                setStatusMsg("VLM 분석 완료!");
                if (onVlmReadyRef.current) {
                  onVlmReadyRef.current({
                    session_id: evt.session_id,
                    best_model: evt.best_model,
                    best_code: evt.best_code,
                    pred_codes: evt.pred_codes,
                    error: evt.error || null,
                  });
                }
                // 🆕 VLM까지 끝났으면 모듈 레벨 참조 정리
                _analyzeAbort = null;
              }
              if (evt.type === "error") throw new Error(evt.error || "서버 오류");
            } catch (pe) { if (pe.message && !pe.message.includes("JSON")) throw pe; }
          }
        }
      } catch (err) { if (err.name === "AbortError") return; console.error("API 호출 실패:", err); setStatus("error"); setErrorMsg(err.message || "서버 연결 실패"); }
    };
    callApi();
    // 🆕 Page5 unmount 시 abort하지 않음 (vlm_ready 수신을 위해)
    // 대신 모듈 레벨 _analyzeAbort로 중복 방지
    return () => { /* SSE는 계속 유지 */ };
  }, [videoData?.sig]);

  return (
    <Phone>
      <NavBar title="AI 분석" onBack={onBack} step={3} />
      <div style={{ padding: "16px 24px 60px" }}>

        {status === "analyzing" && (
          <div className="fade-in" style={{ paddingTop: 40 }}>
            <p style={{ textAlign: "center", fontSize: 18, fontWeight: 800, color: BLUE[700], marginBottom: 24 }}>AI 분석 진행 중</p>

            <div className="fade-in" style={{ marginBottom: 16, padding: "16px 18px", borderRadius: 14, background: uploadPhase === "analyzing" ? `${ACCENT.green}08` : "linear-gradient(135deg, #F7FBFF, #EBF8FF)", border: `1.5px solid ${uploadPhase === "analyzing" ? ACCENT.green : BLUE[300]}`, display: "flex", alignItems: "center", gap: 14, transition: "all .4s ease" }}>
              <div style={{ width: 40, height: 40, borderRadius: 12, background: uploadPhase === "analyzing" ? "#F0FFF4" : "#FFF", border: `1.5px solid ${uploadPhase === "analyzing" ? ACCENT.green : BLUE[200]}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, flexShrink: 0 }}>
                {uploadPhase === "uploading" ? "☁️" : uploadPhase === "converting" ? "🔄" : "☁️"}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontSize: 14, fontWeight: 700, color: uploadPhase === "analyzing" ? ACCENT.green : BLUE[700] }}>
                  {uploadPhase === "uploading" ? "서버에 영상 전송 중..." : uploadPhase === "converting" ? "영상 코덱 변환 중..." : "영상 전송 완료"}
                </p>
                <p style={{ fontSize: 11, color: "#8892B0", marginTop: 3 }}>
                  {uploadPhase === "uploading" ? "블랙박스 영상을 분석 서버로 업로드하고 있습니다" : uploadPhase === "converting" ? "서버에서 영상 포맷을 변환하고 있습니다" : "AI 모델 분석이 진행 중입니다"}
                </p>
              </div>
              {uploadPhase !== "analyzing" && <div style={{ width: 24, height: 24, border: `2.5px solid ${BLUE[100]}`, borderTopColor: BLUE[500], borderRadius: "50%", animation: "spin .8s linear infinite", flexShrink: 0 }} />}
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <ProgressRow icon="💗" name="민다정" color={BLUE[800]} percent={groupProgress.heavy.percent} msg={groupProgress.heavy.msg} done={groupProgress.heavy.done} active={groupProgress.heavy.active} currentSub={groupProgress.heavy.currentSub} />
              <ProgressRow icon="🤖" name="엄도식" color={ACCENT.orange} percent={groupProgress.lite.percent} msg={groupProgress.lite.msg} done={groupProgress.lite.done} active={groupProgress.lite.active} currentSub={groupProgress.lite.currentSub} />
              <ProgressRow icon="💫" name="윤 슬" color={ACCENT.purple} percent={groupProgress.c3d.percent} msg={groupProgress.c3d.msg} done={groupProgress.c3d.done} active={groupProgress.c3d.active} currentSub={groupProgress.c3d.currentSub} showSubs={false} />
            </div>
            <div style={{ marginTop: 28 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: "#8892B0" }}>전체 진행률</span>
                <span style={{ fontSize: 13, fontWeight: 800, color: BLUE[500], fontFamily: "'Outfit',sans-serif" }}>{progress}%</span>
              </div>
              <div style={{ height: 8, borderRadius: 4, background: "#EDF2F7", overflow: "hidden" }}>
                <div style={{ height: "100%", borderRadius: 4, background: `linear-gradient(90deg,${BLUE[400]},${BLUE[300]})`, width: `${progress}%`, transition: "width .5s ease" }} />
              </div>
            </div>
          </div>
        )}

        {status === "error" && (
          <div className="fade-in" style={{ textAlign: "center", paddingTop: 80 }}>
            <div style={{ width: 64, height: 64, margin: "0 auto 16px", borderRadius: "50%", background: ACCENT.redLight, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 32 }}>❌</div>
            <p style={{ fontSize: 18, fontWeight: 700, color: BLUE[700] }}>분석 실패</p>
            <p style={{ fontSize: 14, color: "#8892B0", marginTop: 8, lineHeight: 1.6 }}>{errorMsg}</p>
            <div style={{ marginTop: 16, padding: "14px 20px", borderRadius: 12, background: "#FFF5F5", border: "1px solid #FED7D7", textAlign: "left" }}>
              <p style={{ fontSize: 13, fontWeight: 700, color: ACCENT.red, marginBottom: 8 }}>확인사항:</p>
              <p style={{ fontSize: 12, color: "#4A5568", lineHeight: 1.8 }}>1. 터미널에서 <code style={{ background: "#EDF2F7", padding: "2px 6px", borderRadius: 4 }}>python backend.py</code> 실행 중인지 확인<br />2. http://localhost:5002/api/health 접속 확인<br />3. 모델 파일이 서버에 있는지 확인</p>
            </div>
            <div style={{ marginTop: 24 }}><PrimaryBtn onClick={onHome} icon="🏠">처음으로</PrimaryBtn></div>
          </div>
        )}

      </div>
    </Phone>
  );
};

/* ═══════════════════════════════════════════════════════
   PAGE 6 : 분석 결과
   ═══════════════════════════════════════════════════════ */
const Page6 = ({ onBack, onHome, videoData, apiResult, vlmSession }) => {
  const shareRef = useRef(null);
  const [selectedFaultTab, setSelectedFaultTab] = useState(0);
  const [expandAlts, setExpandAlts] = useState(false);
  const [vlmLoadings, setVlmLoadings] = useState([false, false, false]);
  const [vlmReports, setVlmReports] = useState([null, null, null]);

  // 🆕 vlmSession을 ref로 추적 → generateVlm 내부 polling에서 최신값 접근
  const vlmSessionRef = useRef(vlmSession);
  useEffect(() => { vlmSessionRef.current = vlmSession; }, [vlmSession]);

  const generateVlm = async () => {
    const currentTab = selectedFaultTab;
    const vlmKey = ANALYSTS[currentTab].vlmKey;
    const displayName = ANALYSTS[currentTab].name;

    // 1. 로딩 켜기
    setVlmLoadings(prev => { const a = [...prev]; a[currentTab] = true; return a; });

    try {
      // 2. vlmSession이 아직 없으면 준비될 때까지 대기 (최대 120초)
      let session = vlmSessionRef.current;
      if (!session || !session.session_id) {
        for (let i = 0; i < 120; i++) {
          await new Promise(r => setTimeout(r, 1000));
          session = vlmSessionRef.current;
          if (session?.session_id || session?.error) break;
        }
      }

      // 3. 세션 에러 또는 타임아웃
      if (!session?.session_id) {
        const errMsg = session?.error || "VLM 분석 시간 초과";
        setVlmReports(prev => { const a = [...prev]; a[currentTab] = `⚠️ ${errMsg}`; return a; });
        return;
      }

      // 4. 해당 모델의 예측코드 없음
      if (!session.pred_codes?.[vlmKey]) {
        setVlmReports(prev => { const a = [...prev]; a[currentTab] = `⚠️ ${displayName} 모델의 예측 코드가 없어 리포트를 생성할 수 없습니다.`; return a; });
        return;
      }

      // 5. 백엔드 API 호출
      const res = await fetch(`${API_URL}/api/vlm_report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: session.session_id,
          model_name: vlmKey,
        }),
      });
      const data = await res.json();

      if (data.status === "success" && data.report) {
        setVlmReports(prev => { const a = [...prev]; a[currentTab] = data.report; return a; });
      } else {
        setVlmReports(prev => { const a = [...prev]; a[currentTab] = `⚠️ ${data.message || data.error || "리포트 생성 실패"}`; return a; });
      }
    } catch (err) {
      console.error("VLM 리포트 생성 실패:", err);
      setVlmReports(prev => { const a = [...prev]; a[currentTab] = `⚠️ 네트워크 오류: ${err.message}`; return a; });
    } finally {
      setVlmLoadings(prev => { const a = [...prev]; a[currentTab] = false; return a; });
    }
  };

  const handleShare = async () => {
    if (!shareRef.current) return;
    try {
      if (!window.html2canvas) { const s = document.createElement("script"); s.src = "https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"; document.head.appendChild(s); await new Promise((r, j) => { s.onload = r; s.onerror = j; }); }
      const style = document.createElement("style"); style.id = "share-fix"; style.textContent = `.fade-up,.fade-in,.scale-in{animation:none!important;opacity:1!important;transform:none!important}`; document.head.appendChild(style);
      const canvas = await window.html2canvas(shareRef.current, { backgroundColor: "#FFFFFF", scale: 2, useCORS: true, logging: false }); document.getElementById("share-fix")?.remove();
      canvas.toBlob(async (blob) => { if (!blob) return; const file = new File([blob], "AI문철_분석결과.png", { type: "image/png" }); if (navigator.canShare && navigator.canShare({ files: [file] })) { try { await navigator.share({ title: "AI 문철 분석 결과", files: [file] }); return; } catch (e) { } } const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = "AI문철_분석결과.png"; a.click(); URL.revokeObjectURL(url); }, "image/png");
    } catch (err) { console.error("공유 실패:", err); document.getElementById("share-fix")?.remove(); alert("이미지 생성에 실패했습니다."); }
  };

  const getModelResults = () => {
    if (!apiResult) return [];
    const analyst = ANALYSTS[selectedFaultTab];
    const lm = apiResult.label_maps;
    if (analyst.key === "c3d") {
      if (apiResult.c3d_data)
        return MODEL_KEYS.map(k => transformGroupData(apiResult.c3d_data, lm)[k] || null);
      return [];
    }
    if (apiResult.input_data?.[analyst.key])
      return MODEL_KEYS.map(k => transformGroupData(apiResult.input_data[analyst.key], lm)[k] || null);
    return [];
  };
  const modelResults = getModelResults();

  const getFaultForTab = (tabIdx) => {
    const analyst = ANALYSTS[tabIdx];
    if (!apiResult) return { fault: null, alts: [] };
    if (analyst.faultKey && apiResult.fault_results?.[analyst.faultKey]) {
      const fr = apiResult.fault_results[analyst.faultKey];
      return { fault: fr.best, alts: fr.alts || [] };
    }
    return { fault: apiResult.fault, alts: apiResult.alt_faults || [] };
  };
  const currentFault = getFaultForTab(selectedFaultTab);

  if (!apiResult) return null;

  return (
    <Phone>
      <NavBar title="분석 결과" onBack={onBack} step={4} />
      <div style={{ padding: "16px 24px 60px" }}>
        <div ref={shareRef} style={{ background: "#FFF" }}>

          <SectionHeader icon="🎬" text="분석 영상" color={BLUE[300]} />
          <div className="fade-up">
            <CustomVideoPlayer src={videoData?.url} trimStart={videoData?.trimStart || 0} trimEnd={videoData?.trimEnd || videoData?.duration || 10} isTrimmed={videoData?.isTrimmed || false} />
            {videoData?.isTrimmed && (<div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}><Badge color={ACCENT.orange} bg="#FFF8EB">✂️ {(videoData.trimEnd - videoData.trimStart).toFixed(1)}초 클립</Badge><Badge color={BLUE[500]}>원본 {videoData.trimStart.toFixed(1)}초 ~ {videoData.trimEnd.toFixed(1)}초</Badge></div>)}
          </div>

          {/* ========================================== */}
          {/* 🆕 VLM 분석 상태 + 최적 모델 블록 */}
          {/* ========================================== */}
          {!vlmSession ? (
            // ⏳ VLM 스코어링 진행 중 (Page6 진입 직후)
            <div className="fade-up" style={{ marginTop: 20, padding: "16px 18px", borderRadius: 14, background: "#F8FAFC", border: "1px dashed #CBD5E0", display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ width: 28, height: 28, border: "3px solid #E2E8F0", borderTopColor: BLUE[500], borderRadius: "50%", animation: "spin 1s linear infinite", flexShrink: 0 }} />
              <div>
                <p style={{ fontSize: 14, fontWeight: 700, color: "#4A5568" }}>VLM 영상 분석 중...</p>
                <p style={{ fontSize: 12, color: "#718096", marginTop: 2 }}>Gemini가 영상과 3개 모델 결과를 종합 분석하고 있습니다. (약 30초 소요)</p>
              </div>
            </div>
          ) : vlmSession.error ? (
            // 🚨 VLM 에러
            <div className="fade-up" style={{ marginTop: 20, padding: "16px 18px", borderRadius: 14, background: "#FFF5F5", border: "1px solid #FED7D7", display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ fontSize: 28, flexShrink: 0 }}>🚨</div>
              <div>
                <p style={{ fontSize: 14, fontWeight: 800, color: "#C53030" }}>VLM 평가 실패</p>
                <p style={{ fontSize: 12, color: "#E53E3E", marginTop: 2 }}>{vlmSession.error}</p>
              </div>
            </div>
          ) : (
            // 🏆 VLM 완료 + 최적 모델 표시
            <div className="fade-up" style={{ marginTop: 20, padding: "16px 18px", borderRadius: 14, background: "linear-gradient(135deg, #F0E6FF, #F3E8FF)", border: "1px solid #D6BCFA", display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{ fontSize: 28, flexShrink: 0 }}>🏆</div>
              <div>
                <p style={{ fontSize: 14, fontWeight: 800, color: "#553C9A" }}>VLM 선정 최적 모델: {vlmSession.best_model}</p>
                <p style={{ fontSize: 12, color: "#805AD5", marginTop: 2 }}>각 모델 탭에서 상세 리포트를 생성할 수 있습니다.</p>
              </div>
            </div>
          )}
          {/* ========================================== */}

          {/* ── 과실비율 (모델별 탭) ── */}
          <div className="fade-up">
            <SectionHeader icon="⚖️" text="모델별 사고 분석 결과" color={ACCENT.red} />
            <AnalystTabs selected={selectedFaultTab} onSelect={(i) => { setSelectedFaultTab(i); setExpandAlts(false); }} />

            {currentFault.fault ? (
              <div style={{ borderRadius: 18, background: "#F7FBFF", border: "1px solid #E2E8F0", padding: "22px 18px", boxShadow: "0 2px 16px rgba(0,0,0,.04)" }}>

                {/* ── 탭 내부 VLM 리포트 표시 영역 ── */}
                <div style={{ marginBottom: 16 }}>

                  {/* 1. 로딩 표시 */}
                  {vlmLoadings[selectedFaultTab] && (
                    <div style={{ textAlign: "center", padding: "16px", borderRadius: 12, background: "#FAF5FF", border: "1px solid #E9D8FD" }}>
                      <div style={{ width: 32, height: 32, margin: "0 auto 8px", border: "3px solid #E9D8FD", borderTopColor: ACCENT.purple, borderRadius: "50%", animation: "spin .8s linear infinite" }} />
                      <p style={{ fontSize: 13, fontWeight: 600, color: ACCENT.purple }}>{ANALYSTS[selectedFaultTab].name} VLM 리포트 생성 중...</p>
                      <p style={{ fontSize: 11, color: "#A0AEC0", marginTop: 4 }}>
                        {!vlmSession?.session_id
                          ? "VLM 영상 분석 완료 대기 후 리포트를 작성합니다"
                          : "Gemini가 영상을 분석하여 리포트를 작성하고 있습니다"}
                      </p>
                    </div>
                  )}

                  {/* 2. 리포트 텍스트 표시 (생성 완료 후) */}
                  {vlmReports[selectedFaultTab] && !vlmLoadings[selectedFaultTab] && (
                    <div className="fade-up" style={{ borderRadius: 14, background: "#FAF5FF", border: "1px solid #E9D8FD", padding: "14px 14px" }}>
                      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10, paddingBottom: 8, borderBottom: "1px solid #E9D8FD" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <span style={{ fontSize: 16 }}>📝</span>
                          <span style={{ fontSize: 13, fontWeight: 800, color: BLUE[700] }}>{ANALYSTS[selectedFaultTab].name} AI 상세 리포트</span>
                        </div>
                        <Badge color={ACCENT.purple} bg="#F3E8FF">VLM</Badge>
                      </div>
                      <p style={{ fontSize: 13, color: "#4A5568", lineHeight: 1.7, wordBreak: "keep-all", whiteSpace: "pre-wrap", margin: 0 }}>
                        {vlmReports[selectedFaultTab]}
                      </p>
                    </div>
                  )}

                  {/* 3. 리포트 생성 버튼 (항상 표시 — 클릭 시 VLM 대기 후 생성) */}
                  {!vlmReports[selectedFaultTab] && !vlmLoadings[selectedFaultTab] && (
                    <button
                      onClick={() => generateVlm()}
                      style={{ width: "100%", height: 48, borderRadius: 12, border: "none", background: `linear-gradient(135deg, ${ACCENT.purple}, ${BLUE[400]})`, cursor: "pointer", fontFamily: "inherit", fontSize: 14, fontWeight: 700, color: "#FFF", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, boxShadow: "0 4px 16px rgba(183,148,244,.3)" }}
                    >
                      📝 {ANALYSTS[selectedFaultTab].name} 리포트 생성
                    </button>
                  )}
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                  <FaultBox label="차량 A 과실" pct={currentFault.fault.fa} role={currentFault.fault.role_a} color={ACCENT.red} colorLight={ACCENT.redLight} />
                  <FaultBox label="차량 B 과실" pct={currentFault.fault.fb} role={currentFault.fault.role_b} color={BLUE[400]} colorLight={BLUE[50]} />
                </div>
                {/* ── AI 모델별 분석 결과 (3개 모델 모두 동일 로직) ── */}
                <div style={{ marginTop: 16 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, width: "100%" }}>
                    {modelResults.map((d, i) => <ResultCard key={`${selectedFaultTab}-${i}`} data={d} index={i} visible={true} />)}
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ padding: "18px 20px", borderRadius: 14, background: ACCENT.orangeLight, border: `1px solid ${ACCENT.orange}30` }}>
                <p style={{ fontSize: 14, fontWeight: 700, color: "#C05621" }}>⚠️ 과실비율 매칭 실패</p>
                <p style={{ fontSize: 13, color: "#744210", marginTop: 6, lineHeight: 1.6 }}>{ANALYSTS[selectedFaultTab].name} 모델에서 매칭되는 조합을 찾지 못했습니다.</p>
              </div>
            )}
          </div>

          <div style={{ marginTop: 12 }}>
            <button onClick={handleShare} style={{ width: "100%", height: 52, borderRadius: 14, border: "none", background: `linear-gradient(135deg,${BLUE[500]},${BLUE[300]})`, cursor: "pointer", fontFamily: "inherit", fontSize: 15, fontWeight: 700, color: "#FFF", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, boxShadow: "0 4px 20px rgba(43,122,184,.3)" }}>📤 결과 공유하기</button>
          </div>
          <div style={{ marginTop: 12 }}>
            <button onClick={onHome} style={{ width: "100%", height: 52, borderRadius: 14, border: `2px solid ${BLUE[300]}`, background: "#FFF", cursor: "pointer", fontFamily: "inherit", fontSize: 15, fontWeight: 700, color: BLUE[500], display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>🏠 처음으로</button>
          </div>
        </div>
      </div>
    </Phone>
  );
};

/* ───────── APP ───────── */
export default function App() {
  const [page, setPage] = useState(1); const [videoData, setVideoData] = useState(null); const [bigFont, setBigFont] = useState(false);
  const [apiResult, setApiResult] = useState(null);
  const [vlmSession, setVlmSession] = useState(null); // 🆕 {session_id, best_model, pred_codes}
  useEffect(() => {
    if (!document.getElementById("ai-muncheol-css")) { const s = document.createElement("style"); s.id = "ai-muncheol-css"; s.textContent = GLOBAL_CSS; document.head.appendChild(s); }
    let el = document.getElementById("ai-muncheol-bigfont");
    if (!el) { el = document.createElement("style"); el.id = "ai-muncheol-bigfont"; document.head.appendChild(el); }
    // 변경점: .result-card-label 과 .progress-msg 의 word-break 속성을 keep-all에서 break-all로 변경
    el.textContent = bigFont ? `.phone-container p,.phone-container span,.phone-container button,.phone-container h2,.phone-container code{font-size:calc(1em + 4px)!important}.phone-container .fault-pct{font-size:48px!important}.phone-container h1{font-size:42px!important}.phone-container button{height:auto!important;min-height:48px!important;padding-top:12px!important;padding-bottom:12px!important;white-space:normal!important;word-break:keep-all!important}.phone-container .filename-text{white-space:normal!important;overflow:visible!important;text-overflow:unset!important;word-break:break-all!important;line-height:1.4!important}.phone-container .result-card-label{-webkit-line-clamp:unset!important;-webkit-box-orient:unset!important;display:block!important;overflow:visible!important;white-space:normal!important;word-break:break-all!important}.phone-container .progress-msg{white-space:normal!important;overflow:visible!important;text-overflow:unset!important;word-break:break-all!important;line-height:1.4!important}` : "";
  }, [bigFont]);
  const goHome = () => { setPage(1); setVideoData(null); setApiResult(null); setVlmSession(null); };
  const goToUpload = () => { setPage(2); setVideoData(null); setApiResult(null); setVlmSession(null); };
  const handleAnalysisComplete = (result) => { setApiResult(result); setPage(6); };
  const handleVlmReady = (data) => { setVlmSession(data); }; // 🆕
  switch (page) {
    case 1: return <Page1 onNext={() => setPage(2)} bigFont={bigFont} setBigFont={setBigFont} />;
    case 2: return <Page2 onBack={() => setPage(1)} onNext={(skip) => setPage(skip ? 4 : 3)} setVideoData={setVideoData} />;
    case 3: return <Page3 onBack={goToUpload} onNext={() => setPage(4)} videoData={videoData} setVideoData={setVideoData} />;
    case 4: return <Page4 onBack={() => setPage(videoData?.duration > 10 ? 3 : 2)} onNext={() => setPage(5)} videoData={videoData} />;
    case 5: return <Page5 key={videoData?.sig || "no-sig"} onBack={() => setPage(4)} onHome={goHome} onComplete={handleAnalysisComplete} onVlmReady={handleVlmReady} videoData={videoData} />;
    case 6: return <Page6 onBack={() => setPage(4)} onHome={goHome} videoData={videoData} apiResult={apiResult} vlmSession={vlmSession} />;
    default: return <Page1 onNext={() => setPage(2)} />;
  }
}