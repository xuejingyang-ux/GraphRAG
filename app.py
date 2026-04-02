
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from src.config import get_settings
from src.kg_builder import MedicalKnowledgeGraphBuilder
from src.qa_chain import GraphRAGQAChain
from src.retriever import HybridRetriever


st.set_page_config(page_title="医疗 GraphRAG 指挥台", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")
settings = get_settings()

PAGE_CSS = """
<style>
:root{--bg:#0F172A;--panel:rgba(15,23,42,.74);--text:#E2E8F0;--muted:#94A3B8;--accent:#38BDF8;--accent2:#34D399;--danger:#F97316;--line:rgba(255,255,255,.08);--shadow:0 20px 60px rgba(2,8,23,.42);--sidebar-collapsed:5.8rem;--sidebar-expanded:17.5rem;--sidebar-icon:2.7rem;}
html,body,[class*='css']{font-family:"Aptos","Segoe UI Variable Text","Microsoft YaHei UI","PingFang SC",sans-serif;}
.stApp{color:var(--text);background:radial-gradient(circle at 18% 14%,rgba(56,189,248,.16),transparent 24%),radial-gradient(circle at 84% 18%,rgba(52,211,153,.12),transparent 22%),linear-gradient(180deg,#09111F 0%,#0F172A 55%,#0A1324 100%);}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(rgba(148,163,184,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.04) 1px,transparent 1px);background-size:64px 64px;mask-image:radial-gradient(circle at center,rgba(255,255,255,.85),transparent 85%);opacity:.55;}
.block-container{padding-top:1.15rem;padding-bottom:2rem;max-width:none;}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;}
section[data-testid="stSidebar"]{width:var(--sidebar-collapsed)!important;min-width:var(--sidebar-collapsed)!important;background:linear-gradient(180deg,rgba(15,23,42,.58),rgba(8,15,30,.88))!important;border-right:1px solid rgba(255,255,255,.06)!important;box-shadow:16px 0 44px rgba(2,8,23,.34);backdrop-filter:blur(24px);overflow-x:hidden!important;transition:width .22s ease,min-width .22s ease;}
section[data-testid="stSidebar"]:hover{width:var(--sidebar-expanded)!important;min-width:var(--sidebar-expanded)!important;}
section[data-testid="stSidebar"] .block-container{padding:1rem .9rem;}
.sidebar-copy{opacity:0;max-width:0;overflow:hidden;transition:opacity .18s ease,max-width .18s ease,transform .18s ease;white-space:nowrap;transform:translateX(-.18rem);}
section[data-testid="stSidebar"]:hover .sidebar-copy{opacity:1;max-width:220px;transform:translateX(0);}
.sidebar-box,.sidebar-kpi,.hero,.drawer,.card,.signal,.notice,.dialog-card,.evidence{background:linear-gradient(180deg,rgba(15,23,42,.82),rgba(10,18,34,.72));border:1px solid var(--line);box-shadow:var(--shadow);backdrop-filter:blur(18px);}
.sidebar-box,.sidebar-kpi{width:100%;box-sizing:border-box;transition:border-color .18s ease,box-shadow .18s ease,background .18s ease;}
.sidebar-item{display:grid;grid-template-columns:minmax(0,1fr);justify-items:center;align-items:center;gap:0;padding:.72rem .4rem;box-sizing:border-box;}
.sidebar-box.sidebar-item{border-radius:22px;margin-bottom:.75rem;min-height:4.9rem;}
.sidebar-kpi.sidebar-item{border-radius:18px;margin-bottom:.65rem;min-height:4.2rem;}
.orb{width:2.62rem;height:2.62rem;border-radius:999px;display:grid;place-items:center;background:linear-gradient(135deg,rgba(56,189,248,.95),rgba(14,165,233,.18));box-shadow:0 0 22px rgba(56,189,248,.26);font-size:1.02rem;line-height:1;transform:none!important;justify-self:center;align-self:center;}
.kpi-icon{width:2.48rem;height:2.48rem;display:grid;place-items:center;border-radius:18px;background:linear-gradient(135deg,rgba(56,189,248,.16),rgba(52,211,153,.16));font-size:1.04rem;line-height:1;transform:none!important;justify-self:center;align-self:center;overflow:hidden;}
.kpi-value{font-size:1.08rem;font-weight:800;color:#F8FAFC;text-shadow:0 0 16px rgba(56,189,248,.24);}
.kpi-label,.meta,.drawer-sub,.notice{color:var(--muted);font-size:.8rem;}
.meta-card{display:none;padding:.8rem .9rem;border-radius:18px;margin-bottom:.8rem;}
.meta-row{display:flex;justify-content:space-between;gap:.5rem;margin:.35rem 0;font-size:.76rem;color:var(--muted);}
.meta-badge{padding:.14rem .48rem;border-radius:999px;background:rgba(56,189,248,.12);border:1px solid rgba(56,189,248,.16);color:#E0F2FE;}
.admin-panel-head{position:relative;padding:1rem 1rem 1.05rem;border-radius:24px;background:linear-gradient(135deg,rgba(10,24,42,.98),rgba(9,19,35,.9));border:1px solid rgba(56,189,248,.14);box-shadow:0 22px 48px rgba(2,8,23,.34),inset 0 1px 0 rgba(255,255,255,.05);overflow:hidden;margin-bottom:.9rem;}
.admin-panel-head::before{content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(180deg,rgba(255,255,255,.03),transparent 42%),radial-gradient(circle at 82% 18%,rgba(56,189,248,.2),transparent 34%);}
.admin-panel-kicker{position:relative;z-index:1;display:inline-flex;align-items:center;gap:.45rem;color:#9ADFFF;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;}
.admin-panel-kicker::before{content:"";width:.42rem;height:.42rem;border-radius:999px;background:linear-gradient(135deg,#38BDF8,#34D399);box-shadow:0 0 12px rgba(56,189,248,.34);}
.admin-panel-title{position:relative;z-index:1;margin-top:.6rem;font-size:1.15rem;font-weight:800;color:#F8FAFC;}
.admin-panel-sub{position:relative;z-index:1;margin-top:.45rem;color:#9FB1C5;line-height:1.7;font-size:.84rem;}
div[data-testid="stPopover"] > button,.stButton > button{border-radius:18px;border:1px solid rgba(56,189,248,.18);background:linear-gradient(180deg,rgba(12,21,40,.94),rgba(15,23,42,.84));color:#E0F2FE;}
.stButton > button[kind="primary"]{background:linear-gradient(135deg,rgba(14,165,233,.94),rgba(56,189,248,.72));color:#07101E;font-weight:700;}
div[data-testid="stChatMessage"]{background:transparent!important;border:none!important;box-shadow:none!important;padding:0 0 .75rem!important;gap:.9rem;align-items:flex-start!important;}
div[data-testid="stChatMessageContent"],div[data-testid="stChatMessageContent"] > div,div[data-testid="stChatMessageContent"] .stMarkdown{background:transparent!important;}
div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"],div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"],div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarCustom"],div[data-testid="stChatMessage"] img{width:2.9rem!important;height:2.9rem!important;border-radius:20px!important;box-shadow:0 18px 36px rgba(2,8,23,.26),inset 0 1px 0 rgba(255,255,255,.04)!important;border:1px solid rgba(255,255,255,.08)!important;overflow:hidden;}
div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"]{background:linear-gradient(135deg,rgba(56,189,248,.22),rgba(14,165,233,.84))!important;color:#F0F9FF!important;border-color:rgba(125,211,252,.16)!important;}
div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"]{background:linear-gradient(135deg,rgba(7,40,64,.96),rgba(8,72,102,.86))!important;color:#E0F2FE!important;border-color:rgba(56,189,248,.16)!important;}
div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"] svg,div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] svg,div[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarCustom"] svg{color:inherit!important;}
div[data-testid="stBottom"],div[data-testid="stBottomBlockContainer"]{background:transparent!important;}
div[data-testid="stChatInput"],.stChatInput{background:linear-gradient(180deg,rgba(11,19,35,.92),rgba(9,16,31,.88))!important;border:1px solid rgba(56,189,248,.16)!important;border-radius:28px!important;box-shadow:0 20px 56px rgba(2,8,23,.42),0 0 0 1px rgba(56,189,248,.04)!important;backdrop-filter:blur(20px);padding:.24rem .4rem .18rem!important;}
div[data-testid="stChatInput"] *:not(button):not(svg):not(path),.stChatInput *:not(button):not(svg):not(path){background:transparent!important;background-color:transparent!important;background-image:none!important;box-shadow:none!important;}
div[data-testid="stChatInput"] form,div[data-testid="stChatInput"] > div,div[data-testid="stChatInput"] > div > div,.stChatInput form,.stChatInput > div,.stChatInput > div > div,div[data-testid="stChatInput"] [data-baseweb="base-input"],div[data-testid="stChatInput"] [data-baseweb="input"],div[data-testid="stChatInput"] [data-baseweb="textarea"],div[data-testid="stChatInput"] [data-baseweb="textarea"] > div,div[data-testid="stChatInput"] [data-baseweb="textarea"] > div > div,.stChatInput [data-baseweb="base-input"],.stChatInput [data-baseweb="input"],.stChatInput [data-baseweb="textarea"],.stChatInput [data-testid="stChatInputTextArea"],.stChatInput [data-testid="stChatInputTextArea"] > div,.stChatInput [data-testid="stChatInputTextArea"] > div > textarea{background:transparent!important;background-color:transparent!important;border:none!important;box-shadow:none!important;}
div[data-testid="stChatInput"] [data-testid="stChatInputTextArea"],.stChatInput [data-testid="stChatInputTextArea"]{background:transparent!important;background-color:transparent!important;border:none!important;box-shadow:none!important;}
div[data-testid="stChatInput"] textarea,div[data-testid="stChatInput"] input,div[data-testid="stChatInput"] [role="textbox"],.stChatInput textarea,.stChatInput input,.stChatInput [role="textbox"]{background:transparent!important;background-color:transparent!important;color:#E6F0FA!important;-webkit-text-fill-color:#E6F0FA!important;caret-color:#7DD3FC!important;border:none!important;box-shadow:none!important;}
div[data-testid="stChatInput"] textarea::placeholder,div[data-testid="stChatInput"] input::placeholder,.stChatInput textarea::placeholder,.stChatInput input::placeholder{color:#7E93AB!important;}
div[data-testid="stChatInput"] button,.stChatInput [data-testid="stChatInputSubmitButton"]{border-radius:20px!important;background:linear-gradient(135deg,rgba(14,165,233,.92),rgba(52,211,153,.78))!important;color:#08111F!important;border:1px solid rgba(125,211,252,.2)!important;box-shadow:0 14px 28px rgba(14,165,233,.2)!important;transition:transform .18s ease,box-shadow .18s ease!important;}
div[data-testid="stChatInput"] button:hover,.stChatInput [data-testid="stChatInputSubmitButton"]:hover{transform:translateY(-1px);box-shadow:0 18px 36px rgba(14,165,233,.28)!important;}
.card{position:relative;overflow:hidden;padding:1.08rem 1.18rem 1.04rem;max-width:92%;background:linear-gradient(180deg,rgba(10,18,33,.96),rgba(8,15,29,.88));border:1px solid rgba(148,163,184,.12);box-shadow:0 24px 60px rgba(2,8,23,.38),inset 0 1px 0 rgba(255,255,255,.04);}
.card::after{content:"";position:absolute;inset:0;pointer-events:none;background:radial-gradient(circle at top right,rgba(56,189,248,.12),transparent 36%),linear-gradient(180deg,rgba(255,255,255,.02),transparent 40%);}
.user-card{margin-left:auto;border-radius:28px 10px 28px 28px;background:linear-gradient(180deg,rgba(8,43,72,.98),rgba(8,24,45,.9));border-color:rgba(56,189,248,.2);}
.assistant-card{margin-right:auto;border-radius:10px 32px 32px 28px;border-color:rgba(125,211,252,.12);}
.welcome-card{background:linear-gradient(135deg,rgba(15,23,42,.95),rgba(11,28,48,.84));}
.scanline{position:absolute;inset:0;pointer-events:none;background:linear-gradient(180deg,transparent,rgba(56,189,248,.07),transparent),repeating-linear-gradient(180deg,rgba(255,255,255,.012) 0px,rgba(255,255,255,.012) 1px,transparent 1px,transparent 4px);mix-blend-mode:screen;animation:scan 5s linear infinite;opacity:.72;}
@keyframes scan{0%{transform:translateY(-12%);}100%{transform:translateY(112%);}}
.kicker{display:inline-flex;align-items:center;gap:.45rem;margin-bottom:.65rem;color:#BFEAFF;font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;}
.kicker::before{content:"";width:.42rem;height:.42rem;border-radius:999px;background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 0 14px rgba(56,189,248,.46);}
.copy{position:relative;z-index:1;color:#E5EEF8;line-height:1.75;font-size:.98rem;}
.meter{display:flex;align-items:center;gap:.8rem;margin-top:.85rem;color:var(--muted);font-size:.77rem;}
.track{flex:1;height:.38rem;border-radius:999px;overflow:hidden;background:rgba(255,255,255,.08);}
.fill{height:100%;border-radius:999px;background:linear-gradient(90deg,rgba(52,211,153,.95),rgba(56,189,248,.95));box-shadow:0 0 18px rgba(56,189,248,.32);}
.cursor{display:inline-block;width:.5ch;margin-left:.08rem;color:#9ADFFF;animation:blink .9s steps(2,start) infinite;}
@keyframes blink{50%{opacity:0;}}
.hero{padding:1.5rem 1.6rem;border-radius:30px;position:relative;overflow:hidden;margin-bottom:1.2rem;box-shadow:0 24px 70px rgba(2,8,23,.42),inset 0 1px 0 rgba(255,255,255,.04);}
.hero::after{content:"";position:absolute;right:-8%;bottom:-40%;width:220px;height:220px;border-radius:999px;background:radial-gradient(circle,rgba(56,189,248,.22),transparent 70%);filter:blur(16px);}
.hero-title{margin-top:.95rem;font-size:2.2rem;font-weight:800;line-height:1.06;color:#F8FAFC;}
.hero-title span{background:linear-gradient(90deg,#7DD3FC,#34D399);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero-sub{margin-top:.75rem;max-width:52rem;color:#B6C7DA;line-height:1.8;font-size:.97rem;}
.hero-chips{display:flex;flex-wrap:wrap;gap:.65rem;margin-top:1.05rem;}
.chip{display:inline-flex;align-items:center;gap:.45rem;padding:.55rem .8rem;border-radius:999px;background:rgba(12,23,41,.76);border:1px solid var(--line);font-size:.82rem;transition:border-color .18s ease,transform .18s ease,box-shadow .18s ease;}
.chip:hover{transform:translateY(-1px);border-color:rgba(56,189,248,.18);box-shadow:0 12px 24px rgba(2,8,23,.24);}
.signal{padding:.88rem 1rem;border-radius:20px;margin-bottom:.8rem;}
.signal-row{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;}
.signal-chip{padding:.4rem .66rem;border-radius:999px;font-family:"Cascadia Code","JetBrains Mono",monospace;font-size:.78rem;color:#E0F2FE;background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.16);animation:pulse 2.4s ease-in-out infinite;}
.signal-link{position:relative;flex:0 0 30px;height:1px;background:linear-gradient(90deg,rgba(52,211,153,0),rgba(52,211,153,.62),rgba(56,189,248,0));overflow:hidden;}
.signal-link::after{content:"";position:absolute;left:-20px;width:20px;top:-1px;bottom:-1px;background:linear-gradient(90deg,transparent,rgba(125,211,252,.95),transparent);animation:run 1.6s linear infinite;}
@keyframes pulse{50%{box-shadow:0 0 16px rgba(56,189,248,.22);transform:translateY(-1px);}}
@keyframes run{100%{transform:translateX(50px);opacity:0;}}
.drawer{padding:1.1rem 1rem .9rem;border-radius:28px;position:sticky;top:1rem;}
.trace-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.65rem;margin-bottom:.85rem;}
.trace-card{padding:.8rem .85rem;border-radius:18px;background:rgba(12,21,40,.72);border:1px solid var(--line);transition:border-color .18s ease,transform .18s ease;}
.trace-card:hover{transform:translateY(-1px);border-color:rgba(56,189,248,.16);}
.trace-label{color:var(--muted);font-size:.74rem;}.trace-value{margin-top:.3rem;font-size:1.14rem;font-weight:800;color:#F8FAFC;}
.evidence{padding:.86rem .92rem;border-radius:18px;margin-bottom:.7rem;}.evidence-head{display:flex;justify-content:space-between;gap:.5rem;margin-bottom:.45rem;}.evidence-title{font-weight:700;color:#E0F2FE;font-size:.84rem;}.evidence-score{color:#A7F3D0;font-size:.74rem;}.dialog-card{padding:1rem 1.05rem;border-radius:24px;background:linear-gradient(180deg,rgba(33,15,10,.84),rgba(15,23,42,.84));border-color:rgba(249,115,22,.16);}.dialog-copy{margin-top:.45rem;color:#FDBA74;line-height:1.7;font-size:.9rem;}.dialog-risk{display:inline-block;margin-top:.7rem;padding:.28rem .6rem;border-radius:999px;background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.18);color:#FED7AA;font-size:.76rem;}
.stTabs [data-baseweb="tab-list"]{gap:.45rem;background:rgba(12,21,40,.55);padding:.35rem;border-radius:18px;}.stTabs [data-baseweb="tab"]{border-radius:14px;padding:.35rem .85rem;color:#94A3B8;}.stTabs [aria-selected="true"]{background:rgba(56,189,248,.12)!important;color:#E0F2FE!important;}
section[data-testid="stSidebar"]:hover .meta-card{display:block;}
section[data-testid="stSidebar"]:hover .sidebar-item{grid-template-columns:var(--sidebar-icon) minmax(0,1fr);justify-items:stretch;gap:.82rem;padding-left:.72rem;padding-right:.78rem;border-color:rgba(255,255,255,.1);}
section[data-testid="stSidebar"]:hover .sidebar-item .orb,section[data-testid="stSidebar"]:hover .sidebar-item .kpi-icon{justify-self:start;}
section[data-testid="stSidebar"]:hover .sidebar-kpi:hover,section[data-testid="stSidebar"]:hover .sidebar-box:hover{border-color:rgba(56,189,248,.18);box-shadow:0 16px 30px rgba(2,8,23,.26),0 0 0 1px rgba(56,189,248,.06);}
section[data-testid="stSidebar"] .sidebar-kpi .sidebar-copy{display:block;}
section[data-testid="stSidebar"] div[data-testid="stPopover"]{position:relative;display:grid!important;place-items:center;width:100%;min-height:4.8rem;padding:.48rem .42rem 0!important;border-radius:22px;background:linear-gradient(180deg,rgba(15,23,42,.84),rgba(10,18,34,.72));border:1px solid rgba(56,189,248,.14);box-shadow:0 22px 52px rgba(2,8,23,.34),inset 0 1px 0 rgba(255,255,255,.04);overflow:hidden;box-sizing:border-box;transition:border-color .18s ease,box-shadow .18s ease,padding .18s ease,background .18s ease;margin-top:.2rem;}
section[data-testid="stSidebar"] div[data-testid="stPopover"]::before{content:"";position:absolute;inset:0;pointer-events:none;background:linear-gradient(180deg,rgba(255,255,255,.03),transparent 42%),radial-gradient(circle at 50% 46%,rgba(56,189,248,.14),transparent 58%);}
section[data-testid="stSidebar"] div[data-testid="stPopover"]::after{content:"管理员维护";position:absolute;left:4.9rem;right:.8rem;top:50%;transform:translate(-.18rem,-50%);opacity:0;max-width:0;overflow:hidden;clip-path:inset(0 100% 0 0);transition:opacity .18s ease,transform .18s ease,max-width .18s ease,clip-path .18s ease;color:#E0F2FE;font-weight:800;letter-spacing:.02em;white-space:nowrap;pointer-events:none;text-shadow:0 0 18px rgba(56,189,248,.16);}
section[data-testid="stSidebar"]:hover div[data-testid="stPopover"]{place-items:center start;padding:.48rem .82rem 0!important;border-color:rgba(125,211,252,.2);background:linear-gradient(180deg,rgba(10,27,46,.86),rgba(10,18,34,.78));}
section[data-testid="stSidebar"]:hover div[data-testid="stPopover"]::after{opacity:1;max-width:10rem;clip-path:inset(0 0 0 0);transform:translate(0,-50%);}
section[data-testid="stSidebar"] div[data-testid="stPopover"]:hover{border-color:rgba(56,189,248,.26);box-shadow:0 20px 38px rgba(2,8,23,.3),0 0 0 1px rgba(56,189,248,.1);}
section[data-testid="stSidebar"] [data-testid="stPopoverButton"],section[data-testid="stSidebar"] [data-testid="stPopoverButton"] > div{background:transparent!important;background-color:transparent!important;box-shadow:none!important;}
section[data-testid="stSidebar"] [data-testid="stPopoverButton"] button{width:3.45rem!important;height:3.45rem!important;padding:0!important;margin:0!important;border-radius:999px!important;background:linear-gradient(135deg,rgba(56,189,248,.96),rgba(14,165,233,.74))!important;border:1px solid rgba(125,211,252,.3)!important;box-shadow:0 0 0 6px rgba(56,189,248,.08),0 0 34px rgba(56,189,248,.24)!important;color:#E0F2FE!important;display:grid!important;place-items:center!important;position:relative;z-index:1;font-size:1.16rem!important;line-height:1!important;}
section[data-testid="stSidebar"] [data-testid="stPopoverButton"] button *,section[data-testid="stSidebar"] [data-testid="stPopoverButton"] button > div{background:transparent!important;background-color:transparent!important;}
section[data-testid="stSidebar"] [data-testid="stPopoverButton"] button svg{display:none!important;}
.stPopover > div[data-testid="stPopoverContent"],.stPopover > div[data-testid="stPopoverContent"] > div,body [data-baseweb="popover"],body [data-baseweb="popover"] > div{background:linear-gradient(180deg,rgba(11,20,37,.98),rgba(8,16,31,.94))!important;background-color:#0B1425!important;border:1px solid rgba(56,189,248,.12)!important;border-radius:28px!important;box-shadow:0 26px 70px rgba(2,8,23,.5),0 0 0 1px rgba(56,189,248,.04)!important;backdrop-filter:blur(24px)!important;}
.stPopover > div[data-testid="stPopoverContent"] [data-testid="stVerticalBlock"],.stPopover > div[data-testid="stPopoverContent"] [data-testid="stVerticalBlockBorderWrapper"],.stPopover > div[data-testid="stPopoverContent"] .block-container,.stPopover > div[data-testid="stPopoverContent"] [data-testid="stMarkdownContainer"],body [data-baseweb="popover"] [data-testid="stVerticalBlock"],body [data-baseweb="popover"] [data-testid="stVerticalBlockBorderWrapper"],body [data-baseweb="popover"] .block-container{background:transparent!important;background-color:transparent!important;}
.stPopover > div[data-testid="stPopoverContent"] [data-testid="stMarkdownContainer"] p,body [data-baseweb="popover"] [data-testid="stMarkdownContainer"] p{margin-bottom:0;}
.stPopover > div[data-testid="stPopoverContent"] .stCaptionContainer,body [data-baseweb="popover"] .stCaptionContainer{color:#8FA4BA!important;margin:.05rem 0 .95rem!important;background:transparent!important;}
.stPopover > div[data-testid="stPopoverContent"] [data-testid="stButton"] button,body [data-baseweb="popover"] [data-testid="stButton"] button{min-height:3.8rem!important;padding:.95rem 1.05rem .95rem 1.4rem!important;border-radius:22px!important;background:linear-gradient(135deg,rgba(12,26,45,.98),rgba(8,17,31,.92))!important;border:1px solid rgba(125,211,252,.12)!important;box-shadow:0 18px 36px rgba(2,8,23,.26),inset 0 1px 0 rgba(255,255,255,.04)!important;color:#EAF4FF!important;font-size:1rem!important;font-weight:800!important;justify-content:flex-start!important;text-align:left!important;position:relative;overflow:hidden;transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease,background .18s ease!important;}
.stPopover > div[data-testid="stPopoverContent"] [data-testid="stButton"] button::before,body [data-baseweb="popover"] [data-testid="stButton"] button::before{content:"";position:absolute;left:1rem;top:50%;width:.5rem;height:.5rem;border-radius:999px;transform:translateY(-50%);background:linear-gradient(135deg,#38BDF8,#34D399);box-shadow:0 0 18px rgba(56,189,248,.36);}
.stPopover > div[data-testid="stPopoverContent"] [data-testid="stButton"] button:hover,body [data-baseweb="popover"] [data-testid="stButton"] button:hover{transform:translateY(-1px);border-color:rgba(56,189,248,.22)!important;box-shadow:0 22px 40px rgba(2,8,23,.3),0 0 0 1px rgba(56,189,248,.06)!important;background:linear-gradient(135deg,rgba(13,33,57,.98),rgba(9,19,35,.94))!important;}
.stPopover > div[data-testid="stPopoverContent"] [data-testid="stButton"]:last-of-type button,body [data-baseweb="popover"] [data-testid="stButton"]:last-of-type button{background:linear-gradient(135deg,rgba(10,30,52,.98),rgba(7,22,38,.94))!important;border-color:rgba(56,189,248,.18)!important;}
.stPopover > div[data-testid="stPopoverContent"] [data-testid="stButton"]:last-of-type button::before,body [data-baseweb="popover"] [data-testid="stButton"]:last-of-type button::before{background:linear-gradient(135deg,#7DD3FC,#38BDF8);box-shadow:0 0 18px rgba(56,189,248,.34);}
</style>
"""

WELCOME_MESSAGE = "欢迎来到医疗 GraphRAG 指挥台。\n对话现在被放回视觉中心，右侧溯源看板改成了可折叠抽屉。\n可以直接提问，例如：\n• 肺炎有哪些常见症状？\n• 肺炎的并发症脑脓肿有哪些适用抗生素？\n• 糖尿病需要做哪些检查？"
STATUS_LABELS = {"intent": "意图锁定", "retrieval": "混合检索", "generation": "答案生成", "complete": "回合完成"}
STATUS_DETAILS = {"intent": "正在提炼问题中的实体与关系线索。", "retrieval": "同步拉取图谱邻接子图与文本证据。", "generation": "融合证据并生成最终回答。", "complete": "已完成本轮问答，结果可在溯源看板中继续展开。"}
ADMIN_ACTIONS = {
    "rebuild_graph": {"title": "重建本地图谱", "description": "重新抽取实体与关系，并覆盖本地 JSON 图谱文件。", "risk": "这会刷新当前图谱索引，可能影响后续命中结果。", "confirm": "确认重建图谱", "progress": "正在重建本地图谱..."},
    "rebuild_vector": {"title": "重建向量索引", "description": "重新切分语料并写入 Chroma 向量库。", "risk": "这会重置已有向量集合，期间检索结果可能短暂波动。", "confirm": "确认重建向量索引", "progress": "正在重建向量索引..."},
    "clear_chat": {"title": "清空当前会话", "description": "删除当前聊天历史，并回到初始欢迎态。", "risk": "这只影响当前前端会话，不会改动图谱和向量库。", "confirm": "确认清空会话", "progress": "正在重置会话..."},
}

@st.cache_resource(show_spinner=False)
def get_retriever() -> HybridRetriever:
    return HybridRetriever(settings)


@st.cache_resource(show_spinner=False)
def get_chain() -> GraphRAGQAChain:
    return GraphRAGQAChain(retriever=get_retriever(), settings=settings)


@st.cache_data(show_spinner=False)
def get_dataset_stats(data_dir: str) -> dict[str, int]:
    total_docs = 0
    total_bytes = 0
    for path in Path(data_dir).glob("*.json*"):
        total_bytes += path.stat().st_size
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                total_docs += sum(1 for line in handle if line.strip())
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            total_docs += len(payload) if isinstance(payload, list) else len(payload.get("items", []))
    return {"documents": total_docs, "megabytes": int(total_bytes / (1024 * 1024))}


@st.cache_data(show_spinner=False)
def get_graph_stats(graph_store_path: str) -> dict[str, int]:
    path = Path(graph_store_path)
    if not path.exists():
        return {"nodes": 0, "edges": 0, "documents": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"nodes": len(payload.get("nodes", [])), "edges": len(payload.get("edges", [])), "documents": len(payload.get("documents", []))}


def init_session_state() -> None:
    defaults = {
        "messages": [{"role": "assistant", "content": WELCOME_MESSAGE, "kind": "welcome", "confidence": 96}],
        "turns": [],
        "active_turn_id": None,
        "trace_panel_open": False,
        "pending_admin_action": None,
        "admin_notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation() -> None:
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE, "kind": "welcome", "confidence": 96}]
    st.session_state.turns = []
    st.session_state.active_turn_id = None


def invalidate_cached_views() -> None:
    get_dataset_stats.clear()
    get_graph_stats.clear()


def html_text(value: str) -> str:
    return html.escape(value).replace("\n", "<br/>")


def estimate_confidence(trace: dict[str, Any]) -> int:
    if trace.get("refused"):
        return 28
    graph_hits = len(trace.get("graph_triples", []))
    vector_hits = trace.get("vector_hits", [])
    linked_entities = len(trace.get("linked_entities", []))
    top_score = max((float(item.get("score", 0.0)) for item in vector_hits), default=0.0)
    score = 56 + min(graph_hits, 6) * 4 + min(linked_entities, 4) * 3 + int(top_score * 18)
    return max(24, min(score, 97))


def relation_chain(trace: dict[str, Any]) -> list[str]:
    relations: list[str] = []
    for relation in trace.get("intent_hints", []):
        if relation and relation not in relations:
            relations.append(relation)
    for triple in trace.get("graph_triples", []):
        relation = triple.get("relation")
        if relation and relation not in relations:
            relations.append(relation)
    return relations[:4]


def turn_label(turn: dict[str, Any]) -> str:
    question = turn["question"].replace("\n", " ").strip()
    short_question = question[:18] + ("..." if len(question) > 18 else "")
    return f"第 {turn['turn_id']} 轮 · {short_question}"


def assistant_card(text: str, confidence: int, streaming: bool = False, welcome: bool = False) -> str:
    cursor = '<span class="cursor">|</span>' if streaming else ""
    kicker = "SYSTEM READY" if welcome else "GRAPHRAG RESPONSE"
    extra = " welcome-card" if welcome else ""
    return f"<div class='card assistant-card{extra}'><div class='scanline'></div><div class='kicker'>{kicker}</div><div class='copy'>{html_text(text)}{cursor}</div><div class='meter'><span>{'系统稳定度' if welcome else '回答置信度'} {confidence}%</span><div class='track'><div class='fill' style='width:{confidence}%;'></div></div></div></div>"


def user_card(text: str) -> str:
    return f"<div class='card user-card'><div class='kicker'>USER QUERY</div><div class='copy'>{html_text(text)}</div></div>"


def signal_board(relations: list[str], caption: str) -> str:
    items = []
    for idx, relation in enumerate(relations or ["GRAPH", "VECTOR", "REASON"]):
        items.append(f"<span class='signal-chip'>{html.escape(relation)}</span>")
        if idx != len(relations or [1, 2, 3]) - 1:
            items.append("<span class='signal-link'></span>")
    return f"<div class='signal'><div class='meta' style='margin-bottom:.7rem'>{html.escape(caption)}</div><div class='signal-row'>{''.join(items)}</div></div>"


def trace_cards(trace: dict[str, Any]) -> str:
    metrics = [
        ("图谱实体", len(trace.get("linked_entities", []))),
        ("文本证据", len(trace.get("vector_hits", []))),
        ("关系边数", len(trace.get("graph_edges", []))),
        ("状态", "安全拦截" if trace.get("refused") else "回答完成"),
    ]
    cards = [f"<div class='trace-card'><div class='trace-label'>{label}</div><div class='trace-value'>{value}</div></div>" for label, value in metrics]
    return f"<div class='trace-grid'>{''.join(cards)}</div>"


def render_sidebar() -> None:
    ds, gs = get_dataset_stats(str(settings.data_dir)), get_graph_stats(str(settings.graph_store_path))
    mode = "LLM" if settings.llm_enabled else "规则"
    with st.sidebar:
        st.markdown(f"<div class='sidebar-box sidebar-item'><div class='orb'>✦</div><div class='sidebar-copy'><div style='font-weight:700;color:#F8FAFC'>系统概览</div><div class='meta'>轻量玻璃导航 / Hover Expand</div></div></div>", unsafe_allow_html=True)
        for icon, value, label in [("🗂", ds['documents'], '语料条目'), ("🧬", gs['nodes'], '图谱实体'), ("⇄", settings.max_graph_hops, '多跳层级'), ("⚕", mode, '运行模式')]:
            st.markdown(f"<div class='sidebar-kpi sidebar-item'><div class='kpi-icon'>{icon}</div><div class='sidebar-copy'><div class='kpi-value'>{value}</div><div class='kpi-label'>{label}</div></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='meta-card sidebar-box'><div class='meta-row'><span>Provider</span><span class='meta-badge'>{html.escape(settings.provider)}</span></div><div class='meta-row'><span>Model</span><span class='meta-badge'>{html.escape(settings.resolved_model)}</span></div><div class='meta-row'><span>图谱边数</span><span class='meta-badge'>{gs['edges']}</span></div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:16vh'></div>", unsafe_allow_html=True)
        with st.popover("⚙", use_container_width=False):
            st.markdown("<div class='admin-panel-head'><div class='admin-panel-kicker'>Admin Control</div><div class='admin-panel-title'>管理员维护</div><div class='admin-panel-sub'>高风险操作已被收纳到这里。所有重建与清空动作都会经过二次确认，不会直接执行。</div></div>", unsafe_allow_html=True)
            st.caption("深色玻璃卡片已统一到当前医疗科技蓝背景。")
            if st.button("重建本地图谱", key="admin-graph", use_container_width=True):
                st.session_state.pending_admin_action = "rebuild_graph"
            if st.button("重建向量索引", key="admin-vector", use_container_width=True):
                st.session_state.pending_admin_action = "rebuild_vector"
            if st.button("清空当前会话", key="admin-clear", use_container_width=True):
                st.session_state.pending_admin_action = "clear_chat"


def render_header() -> None:
    ds, gs = get_dataset_stats(str(settings.data_dir)), get_graph_stats(str(settings.graph_store_path))
    mode = "LLM 联动" if settings.llm_enabled else "规则兜底"
    head, actions = st.columns([4.5, 1.2], gap="small")
    with head:
        chips = "".join([f"<span class='chip'><b>{v}</b> {label}</span>" for v, label in [(gs['nodes'], '图谱实体'), (gs['edges'], '关系边'), (ds['documents'], '语料条目'), (settings.max_graph_hops, '跳检索'), (html.escape(mode), '')]])
        st.markdown(f"<div class='hero'><div class='kicker'>MEDICAL TECH / FROSTED GLASS / TRACE DRAWER</div><div class='hero-title'>医疗 <span>GraphRAG</span> 指挥台</div><div class='hero-sub'>对话流已经被重新放回视觉中心，右侧溯源看板改成抽屉式浮窗。检索与推理阶段会以关系脉冲、流式打字和扫描线光效呈现。</div><div class='hero-chips'>{chips}</div></div>", unsafe_allow_html=True)
    with actions:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("新会话", key="hero-reset", use_container_width=True):
            reset_conversation(); st.rerun()
        toggle = "打开溯源看板" if not st.session_state.trace_panel_open else "收起溯源看板"
        if st.button(toggle, key="hero-trace-toggle", use_container_width=True, type="primary"):
            st.session_state.trace_panel_open = not st.session_state.trace_panel_open; st.rerun()


def render_notice() -> None:
    notice = st.session_state.admin_notice
    if notice:
        css = "background:linear-gradient(135deg,rgba(5,46,22,.86),rgba(15,23,42,.76));color:#DCFCE7;" if notice.get("kind") == "success" else "background:linear-gradient(135deg,rgba(67,20,7,.88),rgba(15,23,42,.76));color:#FED7AA;"
        st.markdown(f"<div class='notice' style='padding:.95rem 1rem;border-radius:22px;{css}'>{html.escape(notice['text'])}</div>", unsafe_allow_html=True)
        st.session_state.admin_notice = None


def render_chat_history(container: Any) -> None:
    with container:
        for message in st.session_state.messages:
            role = message["role"]
            with st.chat_message(role, avatar="assistant" if role == "assistant" else "user"):
                if role == "assistant":
                    st.markdown(assistant_card(message["content"], int(message.get("confidence", 92)), welcome=message.get("kind") == "welcome"), unsafe_allow_html=True)
                    turn_id = message.get("turn_id")
                    if turn_id is not None:
                        left, right = st.columns([1.25, 2.2])
                        if left.button("查看溯源", key=f"focus-{turn_id}", use_container_width=True):
                            st.session_state.active_turn_id = turn_id; st.session_state.trace_panel_open = True; st.rerun()
                        right.caption(f"右侧抽屉可切换到第 {turn_id} 轮回答。")
                else:
                    st.markdown(user_card(message["content"]), unsafe_allow_html=True)


def render_graph(graph_nodes: list[dict[str, Any]], graph_edges: list[dict[str, Any]]) -> None:
    if not graph_nodes:
        st.info("当前轮次没有命中可视化子图。")
        return
    palette = {"Disease": "#38BDF8", "Symptom": "#34D399", "Drug": "#F97316", "DrugClass": "#7DD3FC", "Complication": "#A78BFA", "Department": "#F59E0B", "Examination": "#22D3EE", "RiskFactor": "#FB7185", "Pathogen": "#E879F9", "Alias": "#94A3B8"}
    nodes = [Node(id=n["id"], label=n["label"], size=28 if n["type"] == "Disease" else 20, color=palette.get(n["type"], "#64748B"), font={"color": "#F8FAFC", "size": 14}) for n in graph_nodes]
    edges = [Edge(source=e["source"], target=e["target"], label=e["relation"], width=2.8 if e.get("highlight") else 1.7, color="#38BDF8" if e.get("highlight") else "rgba(148,163,184,.42)", font={"color": "#CBD5E1", "strokeWidth": 0}, dashes=not e.get("highlight")) for e in graph_edges]
    config = Config(width="100%", height=420, directed=True, physics=True, hierarchical=False, nodeHighlightBehavior=True, highlightColor="#7DD3FC", collapsible=False, backgroundColor="#09111F")
    agraph(nodes=nodes, edges=edges, config=config)

def render_trace_panel(container: Any) -> None:
    with container:
        head, close = st.columns([4.3, 1])
        with head:
            st.markdown("<div class='drawer'><div class='kicker'>TRACE BOARD / COLLAPSIBLE DRAWER</div><div style='margin-top:.4rem;font-size:1.28rem;font-weight:800;color:#F8FAFC'>溯源看板</div><div class='drawer-sub' style='margin-top:.35rem'>在这里切换轮次、查看图谱命中、文本证据和混合上下文。</div></div>", unsafe_allow_html=True)
        with close:
            if st.button("收起", key="drawer-close", use_container_width=True):
                st.session_state.trace_panel_open = False; st.rerun()
        if not st.session_state.turns:
            st.markdown("<div class='notice' style='padding:1rem;border-radius:18px'>当前还没有可展开的问答轮次。发起一条问题后，这里会显示命中的实体、多跳关系线索、文本证据和最终混合上下文。</div>", unsafe_allow_html=True)
            return
        options = {turn_label(turn): turn["turn_id"] for turn in st.session_state.turns}
        if st.session_state.active_turn_id is None:
            st.session_state.active_turn_id = st.session_state.turns[-1]["turn_id"]
        default = next((label for label, turn_id in options.items() if turn_id == st.session_state.active_turn_id), list(options.keys())[-1])
        selected = st.selectbox("切换轮次", list(options.keys()), index=list(options.keys()).index(default), label_visibility="collapsed")
        st.session_state.active_turn_id = options[selected]
        trace = next(turn["trace"] for turn in st.session_state.turns if turn["turn_id"] == st.session_state.active_turn_id)
        st.markdown(trace_cards(trace), unsafe_allow_html=True)
        st.markdown(signal_board(relation_chain(trace), "关系脉冲"), unsafe_allow_html=True)
        tabs = st.tabs(["图谱子图", "三元组", "文本证据", "混合上下文"])
        with tabs[0]:
            render_graph(trace["graph_nodes"], trace["graph_edges"])
        with tabs[1]:
            if trace["graph_triples"]:
                rows = [{"Head": i.get("head", ""), "Relation": i.get("relation", ""), "Tail": i.get("tail", ""), "Confidence": i.get("confidence", "")} for i in trace["graph_triples"]]
                st.dataframe(rows, hide_index=True, use_container_width=True)
            else:
                st.info("这一轮没有图谱三元组命中。")
        with tabs[2]:
            if trace["vector_hits"]:
                for hit in trace["vector_hits"]:
                    title = hit.get("metadata", {}).get("title", "unknown")
                    score = float(hit.get("score", 0.0))
                    content = hit.get("content", "")
                    st.markdown(f"<div class='evidence'><div class='evidence-head'><div class='evidence-title'>{html.escape(str(title))}</div><div class='evidence-score'>score {score:.2f}</div></div><div class='copy' style='font-size:.84rem'>{html_text(str(content))}</div></div>", unsafe_allow_html=True)
            else:
                st.info("这一轮没有命中文本片段。")
        with tabs[3]:
            st.code(trace["hybrid_context"], language="text")


def append_turn(question: str, result: dict[str, Any]) -> None:
    turn_id = len(st.session_state.turns) + 1
    confidence = estimate_confidence(result["trace"])
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "turn_id": turn_id, "confidence": confidence})
    st.session_state.turns.append({"turn_id": turn_id, "question": question, "answer": result["answer"], "confidence": confidence, "trace": result["trace"]})
    st.session_state.active_turn_id = turn_id


def stream_answer(slot: Any, answer: str, confidence: int) -> None:
    step = 8 if len(answer) < 180 else 16
    for index in range(0, len(answer), step):
        slot.markdown(assistant_card(answer[: index + step], confidence, streaming=True), unsafe_allow_html=True)
        time.sleep(0.025 if len(answer) < 260 else 0.012)
    slot.markdown(assistant_card(answer, confidence), unsafe_allow_html=True)


def handle_new_prompt(prompt: str, chat_container: Any, trace_placeholder: Any | None) -> None:
    chain = get_chain()
    intent_hints = get_retriever().get_intent_hints(prompt)
    with chat_container:
        with st.chat_message("user", avatar="user"):
            st.markdown(user_card(prompt), unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="assistant"):
            signal_slot = st.empty()
            signal_slot.markdown(signal_board(intent_hints, "推理中的关系脉冲"), unsafe_allow_html=True)
            status_box = st.status("模型正在联动图谱与文本证据", expanded=True)
            answer_slot = st.empty()
            def status_callback(event: dict[str, str]) -> None:
                step = event.get("step", "")
                label = STATUS_LABELS.get(step, "处理中")
                detail = STATUS_DETAILS.get(step, "正在更新当前阶段状态。")
                if step == "intent" and intent_hints:
                    detail = f"已锁定 {len(intent_hints)} 条候选关系：{', '.join(intent_hints)}。"
                elif step == "intent":
                    detail = "未显式命中关系关键词，切换为宽检索模式。"
                status_box.write(f"{label}: {detail}")
            result = chain.run_pipeline(prompt, status_callback=status_callback)
            confidence = estimate_confidence(result["trace"])
            status_box.update(label="安全边界触发" if result["trace"]["refused"] else "GraphRAG 推理完成", state="error" if result["trace"]["refused"] else "complete", expanded=False)
            signal_slot.markdown(signal_board(relation_chain(result["trace"]), "最终命中的关系链"), unsafe_allow_html=True)
            stream_answer(answer_slot, result["answer"], confidence)
    append_turn(prompt, result)
    if trace_placeholder is not None:
        trace_placeholder.empty(); render_trace_panel(trace_placeholder.container())


def execute_admin_action(action_key: str) -> None:
    try:
        if action_key == "rebuild_graph":
            builder = MedicalKnowledgeGraphBuilder(settings)
            try:
                result = builder.build_graph(reset=True)
            finally:
                builder.close()
            get_retriever().refresh_graph_store(); invalidate_cached_views()
            st.session_state.admin_notice = {"kind": "success", "text": f"图谱重建完成：处理 {result['documents']} 篇文档，抽取 {result['relations']} 条关系。"}
            return
        if action_key == "rebuild_vector":
            result = get_retriever().rebuild_vector_index(reset=True)
            st.session_state.admin_notice = {"kind": "success", "text": f"向量索引重建完成：处理 {result['documents']} 篇文档，切分 {result['chunks']} 个片段。"}
            return
        if action_key == "clear_chat":
            reset_conversation(); st.session_state.admin_notice = {"kind": "success", "text": "当前会话已清空，界面已经回到初始欢迎态。"}; return
        st.session_state.admin_notice = {"kind": "error", "text": "未识别的管理员操作。"}
    except Exception as exc:
        st.session_state.admin_notice = {"kind": "error", "text": f"操作失败：{exc}"}


@st.dialog("管理员确认")
def render_admin_dialog(action_key: str) -> None:
    action = ADMIN_ACTIONS[action_key]
    st.markdown(f"<div class='dialog-card'><div style='font-size:1.1rem;font-weight:800;color:#FFF7ED'>{html.escape(action['title'])}</div><div class='dialog-copy'>{html.escape(action['description'])}</div><div class='dialog-risk'>{html.escape(action['risk'])}</div></div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    if left.button("取消", key=f"dialog-cancel-{action_key}", use_container_width=True):
        st.session_state.pending_admin_action = None; st.rerun()
    if right.button(action["confirm"], key=f"dialog-confirm-{action_key}", use_container_width=True, type="primary"):
        with st.spinner(action["progress"]):
            execute_admin_action(action_key)
        st.session_state.pending_admin_action = None; st.rerun()


def main() -> None:
    init_session_state()
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    render_sidebar()
    if st.session_state.pending_admin_action:
        render_admin_dialog(st.session_state.pending_admin_action)
    if st.session_state.trace_panel_open:
        workspace, trace_col = st.columns([4.8, 2.15], gap="large")
        trace_placeholder = trace_col.empty()
    else:
        workspace, trace_placeholder = st.container(), None
    with workspace:
        if st.session_state.trace_panel_open:
            _, center, _ = st.columns([0.04, 1, 0.05], gap="small")
        else:
            _, center, _ = st.columns([0.12, 1, 0.12], gap="small")
        with center:
            render_header(); render_notice()
            chat_container = st.container()
            render_chat_history(chat_container)
            prompt = st.chat_input("输入医疗问题，例如：肺炎的并发症脑脓肿有哪些适用抗生素？")
    if st.session_state.trace_panel_open and trace_placeholder is not None:
        render_trace_panel(trace_placeholder.container())
    if prompt:
        handle_new_prompt(prompt, chat_container, trace_placeholder)


if __name__ == "__main__":
    main()
