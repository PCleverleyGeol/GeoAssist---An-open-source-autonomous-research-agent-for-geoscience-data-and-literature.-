######################################################################################
# Title: GEOAssist Agent V3.1 - An open-source autonomous research agent for geoscience data and literature. 
# Description: LLM driven natural language to generate summaries, bibliography lists, download PDF reports autonomously, build knowledge-graphs and extract and visualise domain data.
# Additional: Added a Knowledge Graph extraction to V1.0, added a map visualisation of minerals to V2.0
# V3.1 update includes a fix for map projection for satellite imagery (Mercator) to fully align with country outlines and data points.
# Author: Dr Paul H Cleverley FGS (www.paulhcleverley.com)
# Using Python V3.11
# 30th August 2025
# Comment: This is just several weekends work - shared as is to help raise capabilities for AI in geoscience. I will be updating periodically !!
# This code was compiled with help from GPT-5 especially understanding the APIs
# Check out video here: https://www.linkedin.com/posts/paulhcleverley_gplates-geology-geoscience-activity-7363126587233370113-uw5Q?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAJdxjgBSnNfsxghi8atnlNooAgz4mP6AtE
# Opensource MIT license
# Searches the Internet for references (up to 600) and creates short Gen AI summary for a user query
# Extracts out any geological ages and locations in user query
# Uses these to generate a plate reconstruction over that age period at that location using GPlates https://www.gplates.org/
# Uses these to generate a Lithostratigraphic table for that age period at that location using Macrostrat https://macrostrat.org/
# Uses these to generate a map of mineral occurrences using Mindat https://www.mindat.org/
######################################################################################

### LIBRARY IMPORTS ######################################################################################
import os, re, json, math, time, tempfile, shutil
import requests
import gradio as gr
from typing import List, Dict, Any, Optional, Tuple
from huggingface_hub import InferenceClient
from huggingface_hub.errors import BadRequestError, InferenceTimeoutError, HfHubHTTPError
import tempfile, os, requests
from io import BytesIO
import io, sys, contextlib, logging
import tempfile, shutil, os, time
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from requests.adapters import HTTPAdapter
import time
from tempfile import NamedTemporaryFile
from PIL import Image, ImageDraw, ImageFont
import concurrent.futures
import json, re
from functools import lru_cache
from typing import Tuple, Optional
from datetime import datetime
from duckduckgo_search import DDGS  # pip install duckduckgo_search
from ddgs import DDGS
import unicodedata
from textwrap import dedent

### PARAMETER SETUP ######################################################################################
# Model and internet searching
os.environ.setdefault("HF_TOKEN", "API_KEY_HERE") # You can of course down load an open model like Gemma, Llama etc locally
HF_MODEL_ID = os.getenv("HF_MODEL", "google/gemma-2-9b-it").strip() # I have used Gemma, you can swap to amy model
    
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "120"))
SECTION_TEMPERATURE = float(os.getenv("SECTION_TEMPERATURE", "0.15"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "640"))  # ~50 words comfortably

# Crossref / arXiv
USER_AGENT = "GeoResearchSimple/1.0 (+https://example.local)"
CROSSREF_TIMEOUT = int(os.getenv("CROSSREF_TIMEOUT", "18"))
REF_MAX = int(os.getenv("REF_MAX", "300"))
REF_TARGET_DEFAULT = int(os.getenv("REF_TARGET_DEFAULT", "50"))

# --- ICON SETUP (safe, file-only) ---
ICON_URL = "https://cdn.pixabay.com/photo/2020/10/17/01/48/earth-5660940_1280.png"
ICON_DIR = os.path.join(os.getcwd(), "assets")
ICON_FILE = "app_icon.png"
ICON_PATH = os.path.join(ICON_DIR, ICON_FILE)
ICON_URL = os.getenv(
    "ICON_URL",
    "https://cdn.pixabay.com/photo/2020/10/17/01/48/earth-5660940_1280.png"
).strip()
ICON_DIR = "assets"; os.makedirs(ICON_DIR, exist_ok=True)
ICON_PATH = os.path.join(ICON_DIR, "app_icon.png")
ICON_SIZE = int(os.getenv("ICON_SIZE", "120"))
ICON_URL = "https://cdn.pixabay.com/photo/2020/10/17/01/48/earth-5660940_1280.png"
ICON_PATH = os.path.join("assets", "earth-5660940_1280.png")
ICON_SIZE = int(os.environ.get("ICON_SIZE", "150"))

print(f"[GeoResearch] Using HF chat model: {HF_MODEL_ID}")

# GPlates models to offer (spelling matters for GWS)
GPLATES_MODELS = ["Zahirovic2022", "Muller2019", "Muller2016", "Seton2012"]
DEFAULT_GPLATES_MODEL = "Zahirovic2022"

# --- Map extent tuning (zoom out) ---
BBOX_EXPAND = float(os.getenv("BBOX_EXPAND", "4.0"))      # multiplier (↑ to zoom out more)
BBOX_MIN_LON_DEG = float(os.getenv("BBOX_MIN_LON_DEG", "40"))  # enforce ≥ this width in degrees
BBOX_MIN_LAT_DEG = float(os.getenv("BBOX_MIN_LAT_DEG", "24"))  # enforce ≥ this height in degrees
BBOX_PAD_DEG     = float(os.getenv("BBOX_PAD_DEG", "2"))       # extra absolute padding

# --- DuckDuckGo PDF Agent config (no API key required) ---
DDG_AGENT_ENABLED_DEFAULT = True
DDG_AGENT_ROUNDS_DEFAULT = int(os.getenv("DDG_AGENT_ROUNDS_DEFAULT", "20"))
DDG_AGENT_MAX_PER_ROUND = int(os.getenv("DDG_AGENT_MAX_PER_ROUND", "50"))
DDG_AGENT_PER_SITE_LIMIT = int(os.getenv("DDG_AGENT_PER_SITE_LIMIT", "20"))
DDG_AGENT_MAX_FILE_MB = int(os.getenv("DDG_AGENT_MAX_FILE_MB", "150"))
DDG_AGENT_CONCURRENCY = int(os.getenv("DDG_AGENT_CONCURRENCY", "4"))

# --- LLM geoscience classifier config ---
USE_LLM_GEOCHECK = bool(int(os.getenv("USE_LLM_GEOCHECK", "1")))  # 1=on, 0=off
LLM_GEOCHECK_CONF_THRESH = float(os.getenv("LLM_GEOCHECK_CONF_THRESH", "0.60"))  # 0..1

# --- LLM term-ranker config ---
USE_LLM_TERM_RANKER = bool(int(os.getenv("USE_LLM_TERM_RANKER", "1")))  # 1=on, 0=off
LLM_TERM_RANKER_MAX_CANDS = int(os.getenv("LLM_TERM_RANKER_MAX_CANDS", "100"))  # top-N bigrams to send

# --- Mindat API config ---
MINDAT_BASE = "https://api.mindat.org/v1"
MINDAT_API_TOKEN = os.environ.get("MINDAT_API_TOKEN", "").strip()
if not MINDAT_API_TOKEN:
    # paste your token here
    MINDAT_API_TOKEN = "API_KEY_HERE"
    # expose it to any code that calls os.getenv("MINDAT_API_TOKEN")
    os.environ["MINDAT_API_TOKEN"] = MINDAT_API_TOKEN
MINDAT_API_TOKEN = (os.getenv("MINDAT_API_TOKEN") or "").strip()
MINDAT_SIZE_LIMIT = int(os.getenv("MINDAT_SIZE_LIMIT", "5000"))    # cap results
print("has Mindat token?", bool(os.getenv("MINDAT_API_TOKEN")))
MINDAT_MAX_PLOT = int(os.getenv("MINDAT_MAX_PLOT", "3000"))
MINDAT_MAP_SIZE = (1400, 820)  # width, height in px

### KNOWLEDGE GRAPH CONFIG #######################################################
os.environ["DOWNLOAD_DIR"] = r"C:\Users\paulh\Downloads\Geological_AI_download"
KG_MAX_PDFS = int(os.getenv("KG_MAX_PDFS", "25"))           # safety cap
KG_MAX_PAGES = int(os.getenv("KG_MAX_PAGES", "40"))         # pages per PDF cap
KG_SENTENCES_PER_PDF = int(os.getenv("KG_SENTENCES_PER_PDF", "400"))
KG_BATCH_SIZE = int(os.getenv("KG_BATCH_SIZE", "4"))        # sentences per LLM call
KG_IMG_DPI = int(os.getenv("KG_IMG_DPI", "140"))
KG_LAYOUT_SEED = int(os.getenv("KG_LAYOUT_SEED", "42"))

KG_BATCH_SIZE   = int(os.getenv("KG_BATCH_SIZE", "3"))   # smaller batches = steadier JSON
KG_DEBUG        = bool(int(os.getenv("KG_DEBUG", "1")))  # 1 to print debug to console
KG_DEBUG_SENTS  = int(os.getenv("KG_DEBUG_SENTS", "2"))  # show first N sentences per batch

KG_LABEL_FONT_SIZE = 5   # ← smaller labels; try 5–7

globals()["_KG_NODES_COUNTER"] = globals().get("_KG_NODES_COUNTER", {})
globals()["_KG_EDGES_COUNTER"] = globals().get("_KG_EDGES_COUNTER", {})

### CONFIG SETP ######################################################################################
MACROSTRAT_BASE = "https://macrostrat.org/api/v2"
# Ensure Gradio uses a writable temp folder (and never your project root)
os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(tempfile.gettempdir(), "gradio"))
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)  # ← add this line
os.makedirs("assets", exist_ok=True)
os.makedirs(ICON_DIR, exist_ok=True)
try:
    if not os.path.isfile(ICON_PATH):
        r = requests.get(ICON_URL, timeout=15)
        r.raise_for_status()
        with open(ICON_PATH, "wb") as f:
            f.write(r.content)
except Exception:
    # If anything goes wrong, DO NOT pass a bogus path to Gradio later
    ICON_PATH = None

try:
    # urllib3 v2 name
    from urllib3.util import Retry
except Exception:
    # urllib3 v1 name
    from urllib3.util.retry import Retry

# --- Knowledge graph procs
import collections
try:
    import networkx as nx
except Exception:
    nx = None  # graceful fallback if networkx isn't installed

### PDF DOWNLOAD PROC ######################################################################################

def _downloads_dir() -> str:
    """
    If DOWNLOAD_DIR is set, treat it as the *final* folder.
    Otherwise fall back to ~/Downloads/... logic.
    """
    base = os.getenv("DOWNLOAD_DIR")
    if base:
        path = os.path.expanduser(base)
        os.makedirs(path, exist_ok=True)
        return path

    # fallback (unchanged)
    home = os.path.expanduser("~")
    candidates = [os.path.join(home, "Downloads"), home, os.getcwd()]
    for c in candidates:
        if os.path.isdir(c):
            base = c
            break
    if not base:
        base = home
    path = os.path.join(base, "Geological_AI_download")
    os.makedirs(path, exist_ok=True)
    return path

def _first_author_surname(authors: str) -> str:
    """
    Get a plausible surname for the first author from a comma/and-separated string.
    Examples:
      "Jane Q. Smith, John R. Doe" -> "Smith"
      "Smith JQ; Doe JR" -> "Smith"
      "John van der Meer" -> "Meer" (best effort)
    """
    if not authors:
        return "Unknown"
    # take the first author chunk (comma/semicolon/and separators)
    first = re.split(r"\s*(?:,|;|\band\b)\s*", authors, maxsplit=1)[0].strip()
    if not first:
        return "Unknown"
    # if it's already "Surname, Given", take left
    if "," in first:
        left = first.split(",")[0].strip()
        if left:
            return left.split()[-1]
    # otherwise take last token as surname
    return first.split()[-1]

def _pick_year(years: str) -> str:
    """Pick the first 4-digit year from a string like '1998, 2001'."""
    m = re.search(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", years or "")
    return m.group(1) if m else ""

def _sanitize_filename(stem: str, ext: str = ".pdf", maxlen: int = 140) -> str:
    """
    Sanitize to a Windows-safe filename, clamp length, and ensure extension.
    """
    if not stem:
        stem = "untitled"
    # normalize and strip control chars
    s = unicodedata.normalize("NFKD", stem)
    s = s.encode("ascii", "ignore").decode("ascii", errors="ignore")
    # replace illegal chars: \ / : * ? " < > | and control chars
    s = re.sub(r'[\\/:*?"<>|\x00-\x1F]', " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .\u200b")
    # clamp
    if len(s) > maxlen:
        s = s[:maxlen].rstrip(" .,-_")
    # ensure non-empty
    if not s:
        s = "file"
    # ensure extension
    if ext and not s.lower().endswith(ext.lower()):
        s = f"{s}{ext}"
    return s

def _filename_from_detail(detail: dict) -> str:
    """
    Build 'SurnameYYYY – Title.pdf' from extracted detail fields.
    Falls back gracefully if any part is missing.
    """
    title = (detail.get("title") or "").strip() or "untitled"
    authors = (detail.get("authors") or "").strip()
    years = (detail.get("years") or "").strip()
    surname = _first_author_surname(authors)
    year = _pick_year(years)
    left = surname + (year if year else "")
    stem = f"{left} – {title}" if left else title
    return _sanitize_filename(stem, ext=".pdf", maxlen=140)

### SPATIAL REFERENCE DATA (FALL BACK IN CASE LLM OUTPUT IS NOT OPTIMAL) ######################################################################################
# Bounding box for Gplates
# Reasonable bounding boxes for quick zooms (lon_min, lon_max, lat_min, lat_max)
REGION_BBOX = {
    "north sea": (-10.0, 12.0, 51.0, 62.5),
    "north sea basin": (-10.0, 12.0, 51.0, 62.5),
    "barents": (15.0, 65.0, 68.0, 82.0),
    "barents sea": (15.0, 65.0, 68.0, 82.0),
    "gulf of mexico": (-98.0, -80.0, 18.0, 31.0),
    "western australia": (108.0, 130.0, -36.0, -12.0),
    "arabian plate": (30.0, 65.0, 10.0, 35.0),
    "india": (65.0, 95.0, 5.0, 35.0),
    "north atlantic": (-60.0, 5.0, 35.0, 70.0),
    "south atlantic": (-60.0, 20.0, -55.0, 15.0),
    "east africa": (25.0, 52.0, -15.0, 18.0),
    "andes": (-80.0, -60.0, -56.0, 12.0),
    "africa": (-20.0, 55.0, -36.0, 38.0),
    "west africa": (-20.0, 15.0, -5.0, 28.0), 
}

# Very light heuristics (edit/extend as needed)
REGION_CENTER = {
    "north sea": (2.0, 57.0),
    "barents": (35.0, 75.0),
    "and es": (-70.0, -20.0),
    "andes": (-70.0, -20.0),
    "gulf of mexico": (-90.0, 24.0),
    "east africa": (38.0, 3.0),
    "western australia": (118.0, -22.0),
    "arabian plate": (45.0, 25.0),
    "india": (77.0, 21.0),
    "tethys": (60.0, 20.0),
    "north atlantic": (-30.0, 55.0),
    "south atlantic": (-15.0, -20.0),
    "barents sea": (35.0, 75.0),
    "north sea basin": (2.0, 57.0),
    "west africa": (-5.0, 12.0),
}

# Minimal country centroids as fall back - LLM actually computes so not really needed
COUNTRY_CENTER = {
    "gabon": (11.75, -0.8),
    "angola": (17.88, -11.2),
    "nigeria": (8.2, 9.6),
    "ghana": (-1.2, 7.95),
    "cameroon": (12.35, 5.7),
    "congo": (15.22, -0.7),          # Republic of the Congo
    "democratic republic of the congo": (23.7, -2.9),
    "drc": (23.7, -2.9),
    "equatorial guinea": (10.5, 1.7),
    "benin": (2.34, 9.5),
    "togo": (0.98, 8.6),
    "cote d'ivoire": (-5.55, 7.6),
    "ivory coast": (-5.55, 7.6),
    "senegal": (-14.45, 14.5),
    "mauritania": (-10.3, 20.3),
    "morocco": (-6.0, 31.8),
    "algeria": (2.9, 28.1),
    "libya": (17.0, 27.0),
    "egypt": (30.3, 26.8),
}

### CHRONOLOGICAL REFERENCE DATA (FALL BACK IN CASE LLM OUTPUT IS NOT OPTIMAL) ######################################################################################
# --- Age range parsing (ICS-ish, integer Ma for simplicity) ---
# --- NEW: Era-level ranges (rounded ints, older -> younger) ---
ERA_RANGES = {
    "phanerozoic": (541, 0),
    "paleozoic":   (541, 252),
    "mesozoic":    (252, 66),
    "cenozoic":    (66, 0),

    # Optional deep time (will be clamped elsewhere if needed)
    "proterozoic": (2500, 541),
    "neoproterozoic": (1000, 541),
    "mesoproterozoic": (1600, 1000),
    "paleoproterozoic": (2500, 1600),
    "archean": (4000, 2500),
    "hadean":  (4600, 4000),
}

# --- NEW: Sub-era phrases (early/middle/late) ---
SUBERA_RANGES = {
    # Paleozoic
    "early paleozoic":  (541, 419),  # Cambrian–Silurian
    "middle paleozoic": (419, 359),  # Devonian
    "late paleozoic":   (359, 252),  # Carboniferous–Permian

    # Mesozoic
    "early mesozoic":   (252, 145),  # Triassic–Jurassic
    "middle mesozoic":  (201, 145),  # Jurassic
    "late mesozoic":    (145, 66),   # Cretaceous

    # Cenozoic
    "early cenozoic":   (66, 23),    # Paleogene
    "late cenozoic":    (23, 0),     # Neogene–Quaternary
}

PERIOD_RANGES = {
    # Phanerozoic (rounded ints)
    "cambrian": (541, 485),
    "ordovician": (485, 444),
    "silurian": (444, 419),
    "devonian": (419, 359),
    "carboniferous": (359, 299),
    "permian": (299, 252),
    "triassic": (252, 201),
    "jurassic": (201, 145),
    "cretaceous": (145, 66),
    "paleogene": (66, 23),
    "neogene": (23, 3),      
    "quaternary": (3, 0),
    # Epochs (coarse ints)
    "paleocene": (66, 56),
    "eocene": (56, 34),
    "oligocene": (34, 23),
    "miocene": (23, 5),
    "pliocene": (5, 3),
    "pleistocene": (3, 0),
    "holocene": (1, 0),      # ~11.7 ka rounded to 1 for display
}

SUBPERIOD_RANGES = {
    # Jurassic sub-epochs
    "early jurassic": (201, 174),
    "middle jurassic": (174, 163),
    "late jurassic": (163, 145),
    # Cretaceous examples (optional; extend as needed)
    "early cretaceous": (145, 100),
    "late cretaceous": (100, 66),
}

_RANGE_PATTERNS = [
    # "start_ma=201, end_ma=145"
    re.compile(r"start[_\s]*ma\s*=\s*(\d{1,3}).*?end[_\s]*ma\s*=\s*(\d{1,3})", re.I | re.S),
    # "201–145 Ma" or "201-145 Ma"
    re.compile(r"\b(\d{2,3})\s*[–-]\s*(\d{1,3})\s*ma\b", re.I),
    # "from 201 to 145 Ma" / "between 201 and 145 Ma"
    re.compile(r"\b(?:from|between)\s+(\d{2,3})\s+(?:to|and)\s+(\d{1,3})\s*ma\b", re.I),
]

### MINERAL REFERENCE DATA (SHOULD BE DETECTED BY LLM - THESE ARE FALLBACK)
# Common typos / normalizations and element symbols → mineral names
_MINERAL_FIXUPS = {
    "qtz": "quartz",
}
_SYMBOL_TO_MINERAL = {
    "cu": "copper",
    "au": "gold",
    "fe": "iron",
    "pb": "lead",
    "zn": "zinc",
    "ag": "silver",
    "sn": "tin",
    "ni": "nickel",
    "mo": "molybdenite",  # or "molybdenum" → try both if you like
}

# --- Mindat commodity aliases (put near your other constants) ---
MINDAT_COMMODITY_ALIASES = {
    "copper": ("Cu", "Copper"),
    "gold": ("Au", "Gold"),
    "silver": ("Ag", "Silver"),
    "lead": ("Pb", "Lead"),
    "zinc": ("Zn", "Zinc"),
    "nickel": ("Ni", "Nickel"),
    "cobalt": ("Co", "Cobalt"),
    "iron": ("Fe", "Iron"),
    "tin": ("Sn", "Tin"),
    "molybdenum": ("Mo", "Molybdenum"),
    "tungsten": ("W", "Tungsten"),
    "uranium": ("U", "Uranium"),
    "lithium": ("Li", "Lithium"),
}

### INTERNET SEARCHING REFERENCE DATA (STOPWORDS ######################################################################################

STOPWORDS = {
    # tiny, fast stoplist; add more as you like
    "the","and","for","that","with","from","this","have","not","are","was","were","has",
    "but","you","your","their","its","our","can","may","between","within","across",
    "into","during","after","before","about","over","under","above","below","than",
    "into","onto","per","via","also","such","these","those","most","more","less",
    "based","using","use","used","new","results","study","paper","pdf","file",
    "area","box","com","paper","document","pdf","report","study","analysis","results","discussion",
    "introduction","conclusion","appendix","supplementary","reference","figures","tables","copyright",
    "filetype","version","draft","preprint","full","text","open","access","link","view","download",
    "discuss","discussion","author","authors",
}

JUNK_TERMS = {"area","box","com","filetype","pdf","download","full","text","open","access","discuss","site","index"}

### INTERNET SEARCHING REFERENCE DATA (GEOKEYWORDS ######################################################################################
### Fall back keywords as LLM used to clasify if a PDF is geological with single shot learning
GEO_KEYWORDS = {
    # lithology / facies / sedimentology
    'volcanology', 'mineralogy', 'petrology', 'paleontology', 'palaeontology', 'biostratigraphy', 'hydrogeology', 'seismology',
    "stratigraphy","sedimentology","lithology","lithofacies","facies","sequence stratigraphy", "tectonics", "diagenesis"
    "reservoir","source rock","seal","porosity","permeability","diagenesis","carbonate","clastic", "geophysics", "seismic",
    "siliciclastic","sandstone","siltstone","mudstone","shale","limestone","dolomite","evaporite","gypsum","anhydrite",
    "turbidite","deltaic","fluvial","lacustrine","marine","alluvial","aeolian","outcrop","seismic","well log", "geochemistry"
    "rift","basin","foreland","passive margin","rift shoulder","transfer fault","transform","subduction","orogeny","orogenic",
    "tectonic","tectonics","paleogeography","plate reconstruction","inversion","unconformity","angular unconformity",
    'basin analysis', 'remote sensing', 'gravity and magnetics', 'subsurface', 'structural geology', 'geomorphology', 'environmental geology',
    # time & chronostrat
    "ma","myr","mya","ga","bp","kya","radiometric","isotope","biostratigraphy","chemostratigraphy","magnetostratigraphy",
    # systems / organizations / journals (weak positive)
    "usgs","bgs","copernicus","geoscienceworld","gsa","pangaea","onepetro","geosphere", "sciencedirect"
    # formation language
    "formation","fm.","mbr.","member","group","supergroup","beds",
    # petroleum terms (often geoscience context)
    "play","trap","charge","migration","maturation","kerogen","toc","vitrinite",
}

CHRONO_STAGES = {
        # Chronostratigraphy
        'cambrian', 'ordovician', 'silurian', 'devonian', 'carboniferous', 'triasic', 'jurassic', 'cretaceous', 'paleogene', 
        'neogene', 'pliocene', 'miocene', 'oligocene', 'paleocene', 'eocene', 'quaternary','variscan', 'caledonian',
}
NEGATIVE_KEYWORDS = {
    # movie/pop-culture noise
    "jurassic park","jurassic world","spielberg","box office","franchise","screenplay","script","film","movie",
    "trailer","animatronic","lego","theme park","control room","outpost","jurassicquest"
}

# domains considered "scientific" → allow higher per-site limit & light trust
SCIENTIFIC_DOMAIN_TOKENS = {
    # orgs & hosts
    "usgs.gov","bgs.ac.uk","copernicus.org","copernicus.eu","geoscienceworld.org","geosociety.org","pangaea.de",
    "onepetro.org","lpi.usra.edu","cambridge.org","springer.com","wiley.com","nature.com","science.org",
    "eartharxiv.org", "mdpi.com","frontiersin.org","researchgate.net","academia.edu","elsevier.com","els-cdn.com","jm.copernicus.org",
    "palass.org","gsw","escubed.co.uk","sp.lyellcollection.org","searchanddiscovery.com",
}

### PROCEDURES INTERNET QUERYING - REFERENCING ######################################################################################
def _http_session(retries: int = 1, backoff: float = 0.4,
                  timeout_connect: float = 2.5, timeout_read: float = 5.0) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    r = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"])
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://",  HTTPAdapter(max_retries=r))
    s._timeouts = (timeout_connect, timeout_read)  # store for convenience
    return s

# ---------------- HF Client ----------------
def _client() -> InferenceClient:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set.")
    return InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN, timeout=MODEL_TIMEOUT)

# ---------------- Reference Discovery (paged + DOI dedupe) ----------------
def _best_title_from_crossref(msg: Dict[str, Any], fallback: str = "") -> str:
    tlist = msg.get("title") or []
    if tlist:
        t = " ".join(tlist[:1]).strip()
        # Light de-shouting
        if t.isupper():
            try:
                return t.title()
            except Exception:
                return t
        return t
    return fallback or "[untitled]"

def _format_authors(authors: List[Dict[str, Any]], max_authors: int = 6) -> str:
    if not authors: return ""
    def _cap(s): return s if not s.isupper() else s.title()
    names = []
    for a in authors[:max_authors]:
        fam = _cap((a.get("family") or "").strip())
        giv = (a.get("given") or "").strip()
        inits = "".join([w[0] for w in giv.split() if w]) if giv else ""
        if fam and inits: names.append(f"{fam}, {inits}")
        elif fam: names.append(fam)
        elif giv: names.append(giv)
    if len(authors) > max_authors: names.append("et al.")
    return "; ".join(names)

def _year_from_crossref(msg: Dict[str, Any]) -> Optional[int]:
    def _parts(k): return ((msg.get(k) or {}).get("date-parts") or [[]])[0]
    for key in ("published-online","published-print","issued","created"):
        parts = _parts(key)
        if parts and isinstance(parts[0], int):
            return parts[0]
    return None

def crossref_search_paged(query: str, max_results: int = 100) -> List[Dict[str, Any]]:
    """Paged Crossref search; filters to journal articles; returns up to max_results; DOI-deduped."""
    url = "https://api.crossref.org/works"
    rows = min(100, max(10, int(max_results)))
    offset = 0
    out: List[Dict[str, Any]] = []
    seen_doi = set()

    while len(out) < max_results:
        params = {
            "query.bibliographic": query,
            "rows": rows,
            "offset": offset,
            "filter": "type:journal-article",   # add ",language:en" if you want EN-only
        }
        try:
            r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=CROSSREF_TIMEOUT)
            r.raise_for_status()
            items = (r.json().get("message", {}).get("items") or [])
        except Exception:
            break
        if not items:
            break

        for it in items:
            doi = (it.get("DOI") or "").strip().lower()
            if not doi or doi in seen_doi:
                continue
            seen_doi.add(doi)

            cont = it.get("container-title") or []
            journal = " ".join(cont[:1]).strip() if cont else ""
            rec = {
                "title": _best_title_from_crossref(it, fallback="Untitled"),
                "url": f"https://doi.org/{doi}",
                "doi": doi,
                "journal": journal,
                "year": _year_from_crossref(it),
                "authors": _format_authors(it.get("author") or []),
                "source": "crossref",
            }
            out.append(rec)
            if len(out) >= max_results:
                break
        offset += rows
    return out[:max_results]

def arxiv_search(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    base = "http://export.arxiv.org/api/query"
    try:
        r = requests.get(
            base,
            params={"search_query": query, "start": 0, "max_results": min(100, max_results)},
            headers={"User-Agent": USER_AGENT},
            timeout=CROSSREF_TIMEOUT,
        )
        r.raise_for_status()
        titles = re.findall(r"<title>(.*?)</title>", r.text, flags=re.S)[1:]
        links  = re.findall(r"<id>(.*?)</id>", r.text, flags=re.S)[1:]
        dates  = re.findall(r"<published>(\d{4})-\d{2}-\d{2}</published>", r.text)
        out = []
        for i, (t, u) in enumerate(zip(titles, links)):
            year = None
            if i < len(dates):
                try: year = int(dates[i])
                except: year = None
            out.append({
                "title": re.sub(r"\s+", " ", t).strip(),
                "url": u,
                "doi": None,
                "journal": "arXiv",
                "year": year,
                "authors": "",
                "source": "arXiv",
            })
        return out[:max_results]
    except Exception:
        return []

def discover_references(query: str, ref_target: int) -> List[Dict[str, Any]]:
    ref_target = max(5, min(ref_target, REF_MAX))
    refs = crossref_search_paged(query, max_results=ref_target)
    if len(refs) < ref_target:
        # pad with arXiv (no DOI, won't clash with DOI dedupe)
        pad = arxiv_search(query, max_results=(ref_target - len(refs)))
        refs.extend(pad)
    # final DOI-based dedupe (just in case)
    seen = set(); out = []
    for it in refs:
        key = (it.get("doi") or it.get("url") or "").lower()
        if key and key in seen: continue
        seen.add(key); out.append(it)
        if len(out) >= ref_target: break
    return out

# ---------------- Sources Index & Formatting ----------------
def build_sources_index(refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    numbered, lookup = [], {}
    for i, it in enumerate(refs, 1):
        title = (it.get("title") or "[untitled]").strip()
        url = it.get("url") or (f"https://doi.org/{it['doi']}" if it.get("doi") else "")
        numbered.append(f"[{i}] {title} — {url}")
        if url:
            lookup[url] = i
    return {"numbered": numbered, "lookup": lookup}

def format_sources_md(refs: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, s in enumerate(refs, 1):
        left = f"[{idx}] {s.get('title','[untitled]')}"
        parts = []
        journal = s.get("journal")
        year = s.get("year")
        authors = s.get("authors")

        if journal:
            parts.append(f"{journal} ({year})" if year else f"{journal}")
        elif year:
            parts.append(f"({year})")

        if authors:
            parts.append(authors)

        mid = " — " + " — ".join(parts) if parts else ""
        url = s.get("url") or (f"https://doi.org/{s['doi']}" if s.get("doi") else "")
        lines.append(f"{left}{mid} — {url}")
    return "  \n".join(lines)

# Generates 50 word Gen AI Summary
def write_50w_answer(client: InferenceClient, question: str, sources_index_md: str) -> str:
    system_msg = (
        "You are GeoResearch, a precise geoscience assistant. "
        "Write a single paragraph of ~50 words (±10 words), technical but concise. "
        "Use inline numeric citations [1], [2] that map ONLY to the provided Sources Index. "
        "No heading. No sources list in the body. Do not invent sources."
    )
    user_msg = (
        f"Question:\n{question}\n\n"
        "Sources Index (use these numbers for [#] citations; do NOT add anything else):\n"
        f"{sources_index_md}"
    )
    backoffs = [MODEL_MAX_TOKENS, 480, 360, 240]
    text = ""
    for mt in backoffs:
        try:
            resp = client.chat.completions.create(
                model=HF_MODEL_ID,
                messages=[{"role":"system","content":system_msg},
                          {"role":"user","content":user_msg}],
                max_tokens=int(mt),
                temperature=SECTION_TEMPERATURE,
            )
            text = (resp.choices[0].message.content or "").strip()
            break
        except (BadRequestError, InferenceTimeoutError, HfHubHTTPError):
            continue
    return text or "Unable to generate an answer."

### PROCEDURES - AUTONOMOUS SEARCH AGENT ######################################################################################


def _tokenize_words(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z\-]{1,}", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) >= 3]

def _site_of(url: str) -> str:
    try:
        return re.sub(r"^www\.", "", re.findall(r"://([^/]+)/?", url, flags=re.I)[0].lower())
    except Exception:
        return ""

def _safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:180] or "file"

def _guess_filename_from_url(url: str) -> str:
    base = url.split("?")[0].split("#")[0].rstrip("/")
    leaf = base.split("/")[-1] or "download.pdf"
    if not leaf.lower().endswith(".pdf"):
        leaf += ".pdf"
    return _safe_filename(leaf)

def _extract_pdf_links_from_html(url: str, session: Optional[requests.Session] = None, max_links: int = 5) -> list[str]:
    sess = session or _http_session(retries=1, backoff=0.3, timeout_connect=3.0, timeout_read=8.0)
    try:
        r = sess.get(url, timeout=sess._timeouts, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        html = r.text
    except Exception:
        return []
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    from urllib.parse import urljoin
    out = []
    for h in hrefs:
        if ".pdf" in h.lower():
            out.append(urljoin(url, h))
            if len(out) >= max_links:
                break
    return out

def _pdf_extract_text(pdf_path: str, max_pages: int = 3) -> str:
    """
    Extract raw text from the first few pages of a PDF.
    Tries pdfminer.six first, then PyPDF2.
    """
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        txt = extract_text(pdf_path, maxpages=max_pages) or ""
        if txt.strip():
            return txt
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(open(pdf_path, "rb"))
        out = []
        for p in reader.pages[:max_pages]:
            try:
                out.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(out)
    except Exception:
        return ""

def _ddg_search_pdf_urls(query: str, count: int = 30) -> list[tuple[str, str]]:
    q = _canon_query(query)  # <— ensures exactly one filetype:pdf and normalized spacing
    results: list[tuple[str, str]] = []
    pages: list[tuple[str, str]] = []
    try:
        with DDGS(timeout=25) as ddgs:
            for r in ddgs.text(q, region="wt-wt", safesearch="off", max_results=max(1, min(50, count))):
                if not isinstance(r, dict):
                    continue
                url = (r.get("href") or r.get("url") or "").strip()
                title = (r.get("title") or r.get("body") or "").strip()
                if not url:
                    continue
                (results if url.lower().endswith(".pdf") else pages).append((title, url))
                if len(results) >= count:
                    break
    except Exception:
        pass

    if len(results) < count and pages:
        sess = _http_session(retries=1, backoff=0.3, timeout_connect=3.0, timeout_read=8.0)
        for _, page_url in pages[:20]:
            for pl in _extract_pdf_links_from_html(page_url, session=sess, max_links=4):
                results.append(("", pl))
                if len(results) >= count:
                    break
            if len(results) >= count:
                break
    return results[:count]
    
def _is_pdf_url(url: str, session: Optional[requests.Session] = None) -> tuple[bool, str]:
    """
    Return (is_pdf, reason). reason is useful for the agent log.
    """
    sess = session or _http_session(retries=1, backoff=0.3, timeout_connect=2.0, timeout_read=6.0)
    try:
        r = sess.head(url, allow_redirects=True, timeout=sess._timeouts, headers={"User-Agent": USER_AGENT})
        ct = (r.headers.get("Content-Type") or "").lower()
        if "application/pdf" in ct:
            return True, "HEAD content-type=application/pdf"
        # some servers don't like HEAD; try quick GET
        r = sess.get(url, stream=True, timeout=sess._timeouts, headers={"User-Agent": USER_AGENT})
        ct = (r.headers.get("Content-Type") or "").lower()
        r.close()
        if "application/pdf" in ct:
            return True, "GET content-type=application/pdf"
        if url.lower().endswith(".pdf"):
            return True, "URL endswith .pdf"
        return False, f"content-type={ct or 'unknown'}"
    except Exception as e:
        return False, f"request error: {type(e).__name__}"
    
def _download_pdf_temp(url: str, max_mb: int) -> Optional[str]:
    sess = _http_session(retries=2, backoff=0.6, timeout_connect=4.0, timeout_read=30.0)
    try:
        r = sess.get(url, stream=True, timeout=sess._timeouts, headers={"User-Agent": USER_AGENT})
        r.raise_for_status()
        size = 0
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        path = os.path.abspath(tmp.name); tmp.close()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=262144):  # 256 KB
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_mb * 1024 * 1024:
                    try: os.unlink(path)
                    except: pass
                    return None
                f.write(chunk)
        return path if os.path.isfile(path) and os.path.getsize(path) > 0 else None
    except Exception:
        return None

def _first_n_words(text: str, n_words: int = 250) -> str:
    words = re.findall(r"\S+", text)
    return " ".join(words[:n_words]).strip()

def _extract_first_n_words(pdf_path: str, n_words: int = 200) -> str:
    # Try pdfminer → PyPDF2 → fallback empty
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        txt = extract_text(pdf_path) or ""
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            txt = ""
            reader = PdfReader(open(pdf_path, "rb"))
            for i, page in enumerate(reader.pages[:3]):  # first 3 pages usually suffice
                try:
                    txt += page.extract_text() or ""
                except Exception:
                    continue
        except Exception:
            txt = ""
    # keep first n words
    words = re.findall(r"\S+", txt)
    return " ".join(words[:n_words])

def _parse_title_authors_years(text: str) -> tuple[str, str, str]:
    """
    Very light heuristics to pull a plausible Title, Authors, Years from the head text.
    - Title: first nontrivial line in the first 25 lines that is not 'Abstract' etc.
    - Authors: next line(s) with 2+ proper-name patterns.
    - Years: distinct years in first ~40 lines.
    Returns (title, authors_str, years_str) — empty strings if unknown.
    """
    lines = [re.sub(r"\s+", " ", (ln or "").strip()) for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    title = ""
    # Pick a title candidate
    for ln in lines[:25]:
        if len(ln) < 8 or len(ln) > 200:
            continue
        if re.search(r"\b(abstract|introduction|contents)\b", ln, re.I):
            continue
        if re.search(r"\bdoi\s*:", ln, re.I):
            continue
        title = ln
        break
    # Authors: look after title
    authors = ""
    start_idx = 0
    if title and title in lines:
        try:
            start_idx = lines.index(title)
        except ValueError:
            start_idx = 0
    name_pat = re.compile(r"\b[A-Z][a-z]+(?:[-' ][A-Z][a-z]+)+\b")
    for ln in lines[start_idx + 1:start_idx + 8]:
        names = name_pat.findall(ln)
        if len(names) >= 2:
            # Unique, order-preserving
            seen = set(); uniq = []
            for n in names:
                if n not in seen:
                    seen.add(n); uniq.append(n)
            authors = ", ".join(uniq)
            break
    # Years
    head_text = "\n".join(lines[:40])
    years = sorted(set(re.findall(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", head_text)))
    years_str = ", ".join(years)
    return title, authors, years_str

def _save_pdf_year_histogram(details: list[dict], title: str = "PDFs by Year", pre_year: int = 1980) -> Optional[str]:
    """
    Build a bar chart of reports per year.
    - Rolls up all years < pre_year into a single 'pre {pre_year}' bin on the far left.
    - Ignores items with no detectable year.
    - Ignores future years (> current year).
    """
    years: list[int] = []
    current_year = datetime.utcnow().year  # or datetime.now().year

    for d in details or []:
        y = _pick_year(d.get("years", "") or "")
        if not y:
            # Fallback: try filename
            fn = d.get("filename") or ""
            m = re.search(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", fn)
            if m:
                y = m.group(1)
        if y:
            try:
                yi = int(y)
                if yi <= current_year:          # <-- ignore future years
                    years.append(yi)
            except Exception:
                pass
    if not years:
        return None

    # Count pre-bin and per-year (>= pre_year)
    pre_count = sum(1 for yy in years if yy < pre_year)
    counts: dict[int, int] = {}
    for yy in years:
        if yy >= pre_year:
            counts[yy] = counts.get(yy, 0) + 1

    # Build labels/heights with 'pre {pre_year}' on the far left (only if nonzero)
    labels: list[str] = []
    heights: list[int] = []

    if pre_count > 0:
        labels.append(f"pre {pre_year}")
        heights.append(pre_count)

    ys_sorted = sorted(counts.keys())
    labels.extend([str(y) for y in ys_sorted])
    heights.extend([counts[y] for y in ys_sorted])

    if not labels:
        return None

    # Plot (categorical x)
    fig = plt.figure(figsize=(7.8, 3.2), dpi=120)
    ax = plt.gca()
    xpos = list(range(len(labels)))
    ax.bar(xpos, heights)
    ax.set_xlabel("Year")
    ax.set_ylabel("# Reports")
    ax.set_title(title)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.margins(x=0.01)
    plt.tight_layout()

    out = os.path.join(tempfile.gettempdir(), f"pdf_year_hist_{int(time.time())}.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out if os.path.isfile(out) and os.path.getsize(out) > 0 else None

def _top_terms_for_refine(snippets: list[str], base_query: str, k: int = 8) -> list[str]:
    base_tokens = set(_tokenize_words(base_query)) | STOPWORDS | JUNK_TERMS
    df_uni: dict[str, int] = {}
    df_bi: dict[tuple[str,str], int] = {}

    for snip in snippets:
        toks = [t for t in _tokenize_words(snip) if 3 <= len(t) <= 24 and t.isalpha()]
        toks = [t for t in toks if t not in base_tokens]
        # doc-level uniques (unigrams)
        for t in set(toks):
            df_uni[t] = df_uni.get(t, 0) + 1
        # bigrams (document-level)
        for b in set(zip(toks, toks[1:])):
            if any(w in STOPWORDS or w in JUNK_TERMS for w in b):
                continue
            df_bi[b] = df_bi.get(b, 0) + 1

    ranked_uni = sorted(df_uni.items(), key=lambda kv: (-kv[1], kv[0]))
    ranked_bi  = sorted(df_bi.items(),  key=lambda kv: (-kv[1], kv[0]))

    # --- NEW: let LLM pick bigrams first ---
    llm_bigrams: list[str] = []
    if USE_LLM_TERM_RANKER and df_bi:
        llm_bigrams = _llm_rank_bigrams(df_bi, base_query, k=k)

    out: list[str] = []
    seen = set()

    # prefer LLM bigrams (if any)
    for t in llm_bigrams:
        if t and t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= k: break

    # Fallback / fill with DF bigrams then unigrams
    j = 0
    while len(out) < k and j < len(ranked_bi):
        t = " ".join(ranked_bi[j][0])
        if t not in seen:
            seen.add(t); out.append(t)
        j += 1

    i = 0
    while len(out) < k and i < len(ranked_uni):
        t = ranked_uni[i][0]
        if t not in seen:
            seen.add(t); out.append(t)
        i += 1

    return out[:k]

def _safe_json_extract(txt: str) -> dict | None:
    try:
        m = re.search(r"\{.*\}", txt.strip(), flags=re.S)
        return json.loads(m.group(0) if m else txt)
    except Exception:
        return None

def _llm_rank_bigrams(bigram_counts: dict[tuple[str, str], int],
                      base_query: str,
                      k: int = 8) -> list[str]:
    """
    Ask the LLM to pick the best geology bigrams.
    Input: bigram_counts {("source","rock"): 5, ...}
    Return: ["source rock", "plate tectonics", ...] (<= k)
    """
    if not bigram_counts:
        return []

    # Keep the prompt small: send top-N by DF
    cands_sorted = sorted(bigram_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    cands_sorted = cands_sorted[:max(1, min(LLM_TERM_RANKER_MAX_CANDS, len(cands_sorted)))]
    cand_list = [{"term": " ".join(b), "df": int(df)} for (b, df) in cands_sorted]

    system = (
        "You are a geoscience term selector. From candidate bigrams, pick those most relevant to "
        "geology/geoscience (e.g., stratigraphy, sedimentology, tectonics, geophysics, paleontology, "
        "petroleum systems). Exclude universitry, faculty, department, generic, pop culture, policy, software, and stopwordy phrases. "
        "Prefer geological domain-specific multiword terms. Output STRICT JSON only:\n"
        '{"bigrams":["...","..."],"reasons":["..."]}'
    )

    user = (
        "Base query (context):\n"
        f"{base_query}\n\n"
        "Candidate bigrams with document frequency (DF):\n"
        f"{json.dumps(cand_list, ensure_ascii=False)}\n\n"
        f"Return at most {k} bigrams."
    )

    try:
        cli = _client()
        resp = cli.chat.completions.create(
            model=HF_MODEL_ID,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            temperature=0.1,
            max_tokens=220,
        )
        js = _safe_json_extract(resp.choices[0].message.content or "")
        out = []
        if isinstance(js, dict):
            arr = js.get("bigrams") or []
            for s in arr:
                if isinstance(s, str) and s.strip():
                    out.append(s.strip().lower())
        # Dedupe & clamp
        seen = set(); sel = []
        for t in out:
            if t not in seen:
                seen.add(t); sel.append(t)
            if len(sel) >= k: break
        return sel
    except Exception:
        return []
    
def _refined_queries(base_query: str, terms: list[str], max_queries: int = 3) -> list[str]:
    out = []
    for t in terms[:max_queries]:
        out.append(f"{base_query} {t} filetype:pdf")
    return out

def _is_scientific_domain(host: str) -> bool:
    h = (host or "").lower()
    if any(tok in h for tok in SCIENTIFIC_DOMAIN_TOKENS):
        return True
    return (".edu" in h) or (".ac." in h)  # many university repos

def _geo_score(text: str) -> int:
    t = (text or "").lower()
    score = 0
    # positive signals
    for kw in GEO_KEYWORDS:
        if kw in t:
            score += 1
    for st in CHRONO_STAGES:
        if st in t:
            score += 2  # stages are strong signal
    # time units give a gentle boost when attached to numbers (e.g., "150 Ma")
    if re.search(r"\b\d{1,3}\s*(ma|myr|mya|ga)\b", t):
        score += 2
    # formation/member pattern
    if re.search(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s(Formation|Fm\.|Member|Mbr\.)\b", text):
        score += 2
    return score

def _nongeo_hit(text: str) -> bool:
    t = (text or "").lower()
    return any(bad in t for bad in NEGATIVE_KEYWORDS)

def _parse_pdf_metadata(pdf_path: str) -> tuple[str, str, str]:
    """
    Read Title/Author/Year from PDF info/XMP if available.
    Returns (title, authors, year) — empty strings if unknown.
    """
    try:
        from PyPDF2 import PdfReader
        rdr = PdfReader(open(pdf_path, "rb"))
        meta = getattr(rdr, "metadata", None) or getattr(rdr, "documentInfo", None) or {}
        def _get(k, *alts):
            for kk in (k,) + alts:
                v = meta.get(kk) if isinstance(meta, dict) else getattr(meta, kk, None)
                if v: return str(v).strip()
            return ""
        title = _get("/Title", "title")
        authors = _get("/Author", "author")
        # Year from CreationDate/ModDate like D:20210314...
        raw_date = _get("/CreationDate") or _get("/ModDate")
        year = ""
        m = re.search(r"(?:D:)?(\d{4})", raw_date or "")
        if m: year = m.group(1)
        return (title, authors, year)
    except Exception:
        return ("", "", "")

def _largest_font_title_by_layout(pdf_path: str) -> str:
    """
    Use pdfminer to find the largest-font text lines on the first page.
    Concatenate top few lines if they look like a title.
    """
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer, LTChar, LTTextLine
        first = True
        lines = []
        for page_layout in extract_pages(pdf_path, maxpages=1):
            if not first: break
            first = False
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    for line in element:
                        if not isinstance(line, LTTextLine): 
                            continue
                        text = line.get_text().strip()
                        if not text: 
                            continue
                        # avg font size
                        sizes = [ch.size for ch in line if isinstance(ch, LTChar)]
                        if not sizes: 
                            continue
                        avg_size = sum(sizes) / len(sizes)
                        # ignore headers/footers by y position? (pdfminer coords vary)
                        # Keep it simple: filter page numbers / running headers
                        if re.fullmatch(r"(\d+|[IVXLC]+)", text.strip()):
                            continue
                        lines.append((avg_size, text))
        if not lines:
            return ""
        # pick top-N by size, but keep order of appearance
        lines_sorted = sorted(lines, key=lambda t: t[0], reverse=True)
        top_thresh = max(lines_sorted[0][0] * 0.85, lines_sorted[0][0] - 1.5)
        candidates = [t for (sz, t) in lines if sz >= top_thresh]
        # join if multiple consecutive title-like lines
        title = " ".join(x.strip() for x in candidates)
        title = _clean_title_line(title)
        return title
    except Exception:
        return ""

def _clean_title_line(s: str) -> str:
    if not s: return s
    s = re.sub(r"\s+", " ", s).strip()
    # drop common section words
    if re.search(r"^\s*(abstract|contents|introduction)\b", s, re.I): 
        return ""
    # remove trailing author blocks accidentally glued
    s = re.sub(r"\s+by\s+.*$", "", s, flags=re.I)
    # avoid all-caps shouting unless it's acronyms
    if len(s) > 6 and s.isupper():
        try: s = s.title()
        except Exception: pass
    # reasonable length clamp
    if len(s) > 220: s = s[:220].rstrip(" ,.;:-")
    return s

def _dehyphenate(text: str) -> str:
    # join hyphenated linebreaks: e.g., "strati-\ngraphy" -> "stratigraphy"
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

def _find_doi(text: str) -> str:
    m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text or "", flags=re.I)
    return (m.group(0).strip().rstrip(").,;")) if m else ""

def _crossref_lookup(doi: str) -> tuple[str, str, str]:
    """
    Query Crossref for canonical (title, authors, year). Returns empty strings if not found.
    """
    try:
        url = f"https://api.crossref.org/works/{doi}"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        r.raise_for_status()
        msg = (r.json().get("message") or {})
        # title
        title = " ".join((msg.get("title") or [])[:1]).strip()
        # authors
        auth = msg.get("author") or []
        names = []
        for a in auth[:12]:
            fam = (a.get("family") or "").strip()
            giv = (a.get("given") or "").strip()
            if fam and giv: names.append(f"{giv} {fam}")
            elif fam: names.append(fam)
            elif giv: names.append(giv)
        authors = ", ".join(names)
        # year
        year = ""
        for k in ("published-print","published-online","issued","created"):
            parts = (msg.get(k,{}).get("date-parts") or [[]])[0]
            if parts and isinstance(parts[0], int):
                year = str(parts[0]); break
        return (title, authors, year)
    except Exception:
        return ("", "", "")

def _smart_title_authors_years(pdf_path: str, head_text: str) -> tuple[str, str, str]:
    """
    Best-effort extraction using:
    1) PDF metadata; 2) layout-based title; 3) DOI→Crossref; 4) heuristic from head text.
    """
    # 1) PDF metadata
    t, a, y = _parse_pdf_metadata(pdf_path)
    if t and len(t) > 6:
        return (_clean_title_line(t), a, y)

    # 2) Layout-based (largest font on page 1)
    t2 = _largest_font_title_by_layout(pdf_path)
    if t2 and len(t2) > 6:
        # authors from lines under the title in head_text
        authors = _authors_from_headtext(head_text, t2)
        year = _year_from_text(head_text)
        return (t2, authors, year)

    # 3) DOI → Crossref
    doi = _find_doi(head_text)
    if doi:
        t3, a3, y3 = _crossref_lookup(doi)
        if t3:
            return (_clean_title_line(t3), a3, y3)

    # 4) Heuristic from head text
    return _heuristic_title_authors_years(head_text)

def _authors_from_headtext(text: str, title_hint: str = "") -> str:
    # look at first ~40 non-empty lines after (possible) title
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    if title_hint and title_hint in lines:
        start = lines.index(title_hint) + 1
    else:
        start = 0
    block = " \n".join(lines[start:start+12])
    # names like "A. B. Surname", "Firstname Lastname", comma/and-separated
    pat = r"([A-Z][A-Za-z\-']+(?:\s+[A-Z]\.){0,3}(?:\s+[A-Z][A-Za-z\-']+){1,2})"
    cands = re.findall(pat, block)
    # keep uniques, drop likely affiliations (University, Department)
    out = []
    for n in cands:
        if re.search(r"(University|Department|Institute|Laboratory|School|Center|Centre)", n, re.I):
            continue
        if 5 <= len(n) <= 80 and n not in out:
            out.append(n)
    return ", ".join(out[:10])

def _year_from_text(text: str) -> str:
    m = re.search(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", text or "")
    return m.group(1) if m else ""

def _heuristic_title_authors_years(text: str) -> tuple[str, str, str]:
    # Dehyphenate and get first ~30 lines
    t = _dehyphenate(text or "")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in t.splitlines()]
    lines = [ln for ln in lines if ln]
    # pick first nontrivial line that isn't a section header or running header
    title = ""
    for ln in lines[:30]:
        if len(ln) < 8 or len(ln) > 180: 
            continue
        if re.search(r"\b(abstract|introduction|contents|references|acknowledg(e)?ments)\b", ln, re.I):
            continue
        if re.fullmatch(r"(\d+|[IVXLC]+)", ln): 
            continue
        title = _clean_title_line(ln)
        if title: break
    authors = _authors_from_headtext(t, title_hint=title)
    year = _year_from_text(t)
    return (title or "", authors or "", year or "")

# LLM Classifier to detect if its geoscience content very relevant to the search query
@lru_cache(maxsize=1024)
def _llm_is_geoscience(
    title: str,
    snippet: str,
    domain: str = "",
    url: str = "",
    user_query: str = "",
    ) -> dict | None:
    try:
        _llm_is_geoscience.cache_clear()
    except Exception:
        pass
    """
    Single-shot LLM classifier with relevance check:
      Returns {
        "is_geo":"YES|NO",
        "confidence":0..1,
        "reasons":[...],
        "tags":[...],
        "relevance":"YES|NO",
        "relevance_confidence":0..1
      }
    Returns None on failure.

    Notes:
      - "is_geo" judges if the doc (title+snippet) is predominantly geoscience.
      - "relevance" judges if the user's query is relevant to the doc content.
        (If the query is empty, we set relevance to "UNSPECIFIED" with 0.0.)
    """
    import json, re

    title = (title or "").strip()[:220]
    snippet = (snippet or "").strip()
    if len(snippet) > 2000:
        snippet = snippet[:2000]
    q = (user_query or "").strip()[:500]

    system = dedent("""\
    You are a STRICT query–relevance judge for geoscience search.
    USE ONLY the TITLE and SNIPPET. Do not assume unseen content.
    
    TASK:
    1) Silently extract NOUNS and NOUN PHRASES (NPs) from USER_QUERY:
       - Include proper nouns (places, mines), commodities/elements, deposit types, geologic terms.
       - Normalize: lowercase lemmas; drop function words (prepositions, determiners); collapse hyphenation;
         handle plural/singular variants.
       - Accept common abbreviations/symbols (e.g., copper↔Cu, gold↔Au, molybdenum↔Mo). Treat close variants as the same concept
         (e.g., “porphyry copper”↔“porphyry Cu”↔“Cu–Au porphyry”).
    
    DECISION RULE (is_geo = YES means “relevant to the USER_QUERY”):
    - If the query yields 1–2 NPs: ALL of those NPs (or close variants) must be explicitly present in TITLE+SNIPPET.
    - If the query yields ≥3 NPs: at least TWO DISTINCT NPs must be explicitly present; 
    - If the noun or noun phrase is a country, geographical place name or entity name this must be present in TITLE+SNIPPET.
    - Policy/economics-only pages or entertainmment movie content are NO unless the geologic subject is explicit.
    - If USER_QUERY is empty or yields no meaningful NPs, return is_geo=NO with low confidence.
    
    EVIDENCE (MANDATORY FOR YES):
    - Provide 1–4 SHORT phrases COPIED VERBATIM from TITLE/SNIPPET that show the matched NP(s).
    
    CONFIDENCE (0..1):
    - 0.85–1.00: clear, explicit NP matches (multiple or co-located in a sentence/paragraph).
    - 0.60–0.84: partial but adequate matches (NPs present but not co-located).
    - 0.20–0.59: weak/ambiguous matches.
    - 0.00–0.19: clearly unrelated.
    
    OUTPUT STRICT JSON ONLY:
    {"is_geo":"YES|NO","confidence":0..1,"reasons":["..."],"tags":["..."],"evidence_phrases":["..."]}
    """)    
    # Wrong region (NP “zambia” missing)
    ex_user_1 = (
      "USER_QUERY: copper porphyry in Zambia\n"
      "TITLE: Fluid inclusions from Voia porphyry Cu–Au (Romania)\n"
      "SNIPPET: ... Metaliferi Mountains, Romania ..."
    )
    ex_assistant_1 = (
      "{\"is_geo\":\"NO\",\"confidence\":0.98,"
       "\"reasons\":[\"query NPs not present in text (zambia)\"],"
       "\"tags\":[\"np-missing\"],"
       "\"evidence_phrases\":[\"Romania\",\"Metaliferi Mountains\"]}"
    )
    
    # Wrong topic (NP “porphyry” missing)
    ex_user_2 = (
      "USER_QUERY: copper porphyry in Zambia\n"
      "TITLE: Stratiform Cu–Co deposits of the Central African Copperbelt (Zambia and DRC)\n"
      "SNIPPET: ... stratiform copper–cobalt mineralization ..."
    )
    ex_assistant_2 = (
      "{\"is_geo\":\"NO\",\"confidence\":0.96,"
       "\"reasons\":[\"query NPs not present in text (porphyry)\"],"
       "\"tags\":[\"np-missing\"],"
       "\"evidence_phrases\":[\"stratiform copper–cobalt\"]}"
    )
    
    # Positive (NPs present and co-located)
    ex_user_3 = (
      "USER_QUERY: copper porphyry in Zambia\n"
      "TITLE: Kansanshi, Zambia: porphyry-style Cu–Au mineralization\n"
      "SNIPPET: ... Kansanshi mine in north-western Zambia exhibits porphyry Cu–Au ..."
    )
    ex_assistant_3 = (
      "{\"is_geo\":\"YES\",\"confidence\":0.96,"
       "\"reasons\":[\"multiple query NPs explicitly present\"],"
       "\"tags\":[\"np-match\"],"
       "\"evidence_phrases\":[\"porphyry Cu–Au\",\"Kansanshi\",\"Zambia\"]}"
    )    
    try:
        cli = _client()
        resp = cli.chat.completions.create(
          model=HF_MODEL_ID,
          messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": ex_user_1},
            {"role": "assistant", "content": ex_assistant_1},
            {"role": "user", "content": ex_user_2},
            {"role": "assistant", "content": ex_assistant_2},
            {"role": "user", "content": ex_user_3},
            {"role": "assistant", "content": ex_assistant_3},
            {"role": "user", "content":
              f"USER_QUERY: {user_query or '[none]'}\n"
              f"DOMAIN: {domain or '[unknown]'}\n"
              f"URL: {url or '[unknown]'}\n"
              f"TITLE: {title or '[untitled]'}\n"
              f"SNIPPET: {snippet or '[none]'}\n"
            },
          ],
          temperature=0.0,
          max_tokens=220,
        )
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", txt, flags=re.S)
        js = json.loads(m.group(0) if m else txt)

        # --- normalize fields ---
        js["is_geo"] = str(js.get("is_geo","NO")).strip().upper()
        try:
            js["confidence"] = float(js.get("confidence", 0.0))
        except Exception:
            js["confidence"] = 0.0

        rel = str(js.get("relevance", "UNSPECIFIED")).strip().upper()
        if rel not in {"YES","NO","UNSPECIFIED"}:
            rel = "UNSPECIFIED"
        js["relevance"] = rel

        try:
            js["relevance_confidence"] = float(js.get("relevance_confidence", 0.0))
        except Exception:
            js["relevance_confidence"] = 0.0

        if not isinstance(js.get("reasons"), list): js["reasons"] = []
        if not isinstance(js.get("tags"), list): js["tags"] = []

        # If caller provided no query, force UNSPECIFIED relevance
        if not q:
            js["relevance"] = "UNSPECIFIED"
            js["relevance_confidence"] = 0.0

        return js
    except Exception:
        return None
    
def _canon_query(q: str) -> str:
    q = (q or "").strip()
    q = re.sub(r"\s+", " ", q).lower()
    # remove any existing filetype:pdf tokens, then append exactly one
    q = re.sub(r"\bfiletype\s*:\s*pdf\b", "", q).strip()
    if q:
        q = f"{q} filetype:pdf"
    else:
        q = "filetype:pdf"
    return q
    
# MAIN DUCKDUCKGO AUTONOMOUS SEARCH ORCHESTRATION PROC
def ddg_pdf_search_agent(
    user_query: str,
    rounds: int = DDG_AGENT_ROUNDS_DEFAULT,
    max_per_round: int = DDG_AGENT_MAX_PER_ROUND,
    per_site_limit: int = DDG_AGENT_PER_SITE_LIMIT,
    max_file_mb: int = DDG_AGENT_MAX_FILE_MB,
    concurrency: int = DDG_AGENT_CONCURRENCY,
):
    """
    Returns (saved_paths, logs, queries_used, domain_counts, details)
    (If your current code returns 4 values, add 'details' and update run_simple accordingly.)
    """
    logs: list[str] = []
    queries_used: list[str] = []
    seen_urls: set[str] = set()
    accepted_tmp: dict[str, str] = {}      # url -> tmp path (only if accepted)
    site_counts: dict[str, int] = {}
    details_by_url: dict[str, dict] = {}   # url -> detail dict

    used_qset: set[str] = set()      # canonical forms we've already sent to DDG
    queries_used: list[str] = []     # human-readable originals for the UI
    
    def _register_query(q_raw: str) -> str | None:
        cq = _canon_query(q_raw)
        if cq in used_qset:
            logs.append(f"Skip duplicate query: '{q_raw}'")
            return None
        used_qset.add(cq)
        queries_used.append(q_raw)  # keep original wording for the summary panel
        return cq

    def _download_and_skim(url: str) -> dict:
        tmp = _download_pdf_temp(url, max_mb=max_file_mb)
        dom = _site_of(url)
        detail = {
            "url": url, "domain": dom, "tmp_path": tmp or "", "saved_path": None, "filename": None,
            "title": "", "authors": "", "years": "", "snippet": "",
            "geo_score": 0, "is_geo": False, "reject_reason": "",  # keep fields for summaries
            "llm_is_geo": None, "llm_conf": None, "llm_tags": [], "llm_reasons": [],
        }
        if not tmp:
            detail["reject_reason"] = "download failed or too large"
            return detail
    
        text = _pdf_extract_text(tmp, max_pages=3)
        detail["snippet"] = _first_n_words(text, n_words=250)
        
        # NEW: robust title/authors/year
        t, a, y = _smart_title_authors_years(tmp, text[:4000])
        detail["title"], detail["authors"], detail["years"] = t, a, y
    
        # ---------------- LLM classification (primary) ----------------
        verdict = None
        if USE_LLM_GEOCHECK:
            verdict = _llm_is_geoscience(
                detail["title"],
                text[:1600],
                dom,
                url,
                user_query=user_query,   # <-- pass it through
            )
        if verdict:
            detail["llm_is_geo"] = verdict.get("is_geo")
            detail["llm_conf"] = verdict.get("confidence")
            detail["llm_tags"] = verdict.get("tags") or []
            detail["llm_reasons"] = verdict.get("reasons") or []
            if verdict["is_geo"] == "YES" and float(verdict.get("confidence", 0.0)) >= LLM_GEOCHECK_CONF_THRESH:
                detail["is_geo"] = True
            else:
                detail["is_geo"] = False
                detail["reject_reason"] = f"LLM NO (conf={verdict.get('confidence',0):.2f})"
    
        # ---------------- Heuristic backstop if LLM unavailable ----------------
        if verdict is None:
            gs = _geo_score(text)
            detail["geo_score"] = gs
            if _nongeo_hit(text):
                detail["is_geo"] = False
                detail["reject_reason"] = "pop-culture terms detected"
            elif gs >= 3 or _is_scientific_domain(dom):
                detail["is_geo"] = True
            else:
                detail["is_geo"] = False
                detail["reject_reason"] = f"geo_score={gs} below threshold"
    
        return detail

    def _collect_candidates(q: str) -> list[str]:
        hits = _ddg_search_pdf_urls(q, count=max_per_round * 20)
        sess = _http_session(retries=1, backoff=0.3, timeout_connect=2.0, timeout_read=6.0)
        urls: list[str] = []
        for _, url in hits:
            if url in seen_urls:
                continue
            ok, why = _is_pdf_url(url, session=sess)
            if not ok:
                logs.append(f"Skip (not PDF): {url} — {why}")
                continue
            urls.append(url)
        logs.append(f"Query '{q}' → {len(urls)} PDF candidates.")
        return urls[:max_per_round]

    # ---- Round 1 ----
    q0 = f"{user_query} filetype:pdf"
    cq0 = _register_query(q0)
    cand = _collect_candidates(cq0) if cq0 else []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_download_and_skim, u) for u in cand]
        for f in concurrent.futures.as_completed(futs):
            d = f.result()
            url = d["url"]; seen_urls.add(url); details_by_url[url] = d

            # per-site limit: lenient for scientific hosts
            sci = _is_scientific_domain(d["domain"])
            # Allow higher cap only if the content itself passed the geo check
            cap = (per_site_limit * 3) if (sci and d["is_geo"]) else per_site_limit

            if not d["is_geo"]:
                logs.append(f"Reject (non-geo): {url} — {d['reject_reason']}")
                # cleanup
                if d["tmp_path"]:
                    try: os.unlink(d["tmp_path"])
                    except Exception: pass
                continue

            if cap and site_counts.get(d["domain"], 0) >= cap:
                logs.append(f"Reject (per-site cap {cap}): {url}")
                # cleanup
                if d["tmp_path"]:
                    try: os.unlink(d["tmp_path"])
                    except Exception: pass
                continue

            accepted_tmp[url] = d["tmp_path"]
            site_counts[d["domain"]] = site_counts.get(d["domain"], 0) + 1

    # ---- Further rounds (refine on accepted snippets only) ----
    snips = [details_by_url[u]["snippet"] for u in accepted_tmp.keys() if details_by_url[u].get("snippet")]
    for r in range(2, max(1, rounds) + 1):
        if not snips:
            logs.append(f"Round {r}: no accepted snippets to refine; stopping.")
            break
    
        terms = _top_terms_for_refine(snips, base_query=user_query, k=24)
        queries = _refined_queries(user_query, terms, max_queries=10)
        logs.append(f"Round {r}: refining with terms {terms[:5]} → {len(queries)} candidate queries.")
    
        for qi in queries:
            cqi = _register_query(qi)       # only adds to queries_used if truly new
            if not cqi:
                continue                    # duplicate → skip
    
            cand = _collect_candidates(cqi) # use canonical when searching
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = [ex.submit(_download_and_skim, u) for u in cand]
                for f in concurrent.futures.as_completed(futs):
                    d = f.result()
                    url = d["url"]; seen_urls.add(url); details_by_url[url] = d
                    sci = _is_scientific_domain(d["domain"])
                    cap = (per_site_limit * 3) if (sci and d["is_geo"]) else per_site_limit
    
                    if not d["is_geo"]:
                        logs.append(f"Reject (non-geo): {url} — {d['reject_reason']}")
                        if d["tmp_path"]:
                            try: os.unlink(d["tmp_path"])
                            except Exception: pass
                        continue
    
                    if cap and site_counts.get(d["domain"], 0) >= cap:
                        logs.append(f"Reject (per-site cap {cap}): {url}")
                        if d["tmp_path"]:
                            try: os.unlink(d["tmp_path"])
                            except Exception: pass
                        continue
    
                    accepted_tmp[url] = d["tmp_path"]
                    site_counts[d["domain"]] = site_counts.get(d["domain"], 0) + 1
                    if d.get("snippet"):
                        snips.append(d["snippet"])
                    
            
# ---- Move accepted temps to Downloads (rename: AuthorYear – Title.pdf) ----
    saved_paths: list[str] = []
    target_dir = _downloads_dir()
    for url, tmp in accepted_tmp.items():
        try:
            d = details_by_url.get(url, {})  # has title/authors/years
            name = _filename_from_detail(d)  # NEW: SurnameYYYY – Title.pdf
            out = os.path.join(target_dir, name)
    
            # de-dup if file exists
            base, ext = os.path.splitext(out)
            i = 1
            while os.path.exists(out):
                out = f"{base}({i}){ext}"
                i += 1
    
            shutil.move(tmp, out)
            saved_paths.append(out)
    
            # annotate details
            d["saved_path"] = out
            d["filename"] = os.path.basename(out)
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            except Exception:
                pass

    logs.append(f"Saved {len(saved_paths)} PDF(s) to: {target_dir}")

    # Prepare return
    domain_counts = dict(sorted(site_counts.items(), key=lambda kv: (-kv[1], kv[0])))
    details = sorted(
        [details_by_url[u] for u in accepted_tmp.keys()],
        key=lambda d: (d.get("domain",""), d.get("filename") or "")
    )
    return (saved_paths, logs, queries_used, domain_counts, details)
            
### PROCEDURES - detect location in user query and determine lat longs as required ######################################################################################
def _llm_geocodable_places(client: InferenceClient, q: str) -> list[str]:
    """
    Normalize up to 3 present-day geocodable place names from the user's query.
    Returns e.g. ["Gabon", "West Africa"].
    """
    sys = (
        "You normalize locations for geocoding. "
        "Extract up to 3 present-day geographic names, continents, countries, states or regions from the user's query. "
        "If the query uses geologic terms (plates/basins), provide a present-day geocodable alias "
        "(e.g., 'Arabian plate' -> 'Arabian Peninsula'). "
        "Return STRICT JSON: {\"best\": \"...\", \"candidates\": [\"...\", \"...\"]}. No extra text."
    )
    user = f"Query: {q}"
    try:
        resp = _client().chat.completions.create(
            model=HF_MODEL_ID,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            max_tokens=160,
            temperature=0.1,
        )
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", txt, flags=re.S)
        js = json.loads(m.group(0) if m else txt)
        cand = []
        if isinstance(js, dict):
            if isinstance(js.get("best"), str) and js["best"].strip():
                cand.append(js["best"].strip())
            for c in (js.get("candidates") or []):
                if isinstance(c, str) and c.strip():
                    cand.append(c.strip())
        # unique, ordered, max 3
        seen = set(); out = []
        for c in cand:
            k = c.lower()
            if k not in seen:
                seen.add(k); out.append(c)
            if len(out) >= 3: break
        return out
    except Exception:
        return []

@lru_cache(maxsize=256)
def _geocode_nominatim(name: str) -> tuple[float, float] | None:
    """Geocode a name with OSM Nominatim; return (lon, lat) centroid."""
    if not name: return None
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": name, "format": "jsonv2", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        arr = r.json() or []
        if not arr: return None
        rec = arr[0]
        bb = rec.get("boundingbox")
        if isinstance(bb, list) and len(bb) == 4:
            south, north, west, east = map(float, bb)
            lon = (west + east) / 2.0
            lat = (south + north) / 2.0
        else:
            lat = float(rec["lat"]); lon = float(rec["lon"])
        time.sleep(1.0)  # be polite to Nominatim
        return (lon, lat)
    except Exception:
        return None

def _infer_point_from_query(q: str) -> tuple[float, float]:
    """
    1) Use LLM to extract geocodable place strings from the query.
    2) Geocode the first candidate via Nominatim.
    3) Last resort: geocode the full query string.
    4) If all fails, return (0,0).
    """
    # LLM → place strings → geocode
    try:
        client = _client()
        for name in _llm_geocodable_places(client, q):
            xy = _geocode_nominatim(name)
            if xy:
                return xy  # (lon, lat)
    except Exception:
        pass

    # Last-resort: try the whole query phrase
    xy = _geocode_nominatim((q or "").strip())
    if xy:
        return xy

    return (0.0, 0.0)

def _infer_bbox_from_query(q: str) -> tuple[float, float, float, float] | None:
    """
    1) LLM -> candidate place strings -> Nominatim bounding box.
    2) Fallback to REGION_BBOX keyword matches.
    3) Else None (caller can go global).
    """
    # LLM-first
    try:
        client = _client()
        expand = float(os.getenv("BBOX_EXPAND", "1.6"))
        for name in _llm_geocodable_places(client, q):
            bb = _geocode_nominatim_bbox(name)
            if bb:
                return expand_bbox(bb, factor=expand)
    except Exception:
        pass

    # Fallback to your curated list
    ql = (q or "").lower()
    for key, bbox in REGION_BBOX.items():
        if key in ql:
            return bbox

    return None

def expand_bbox(bb, factor=1.04, min_lon_span=2.0, min_lat_span=1.2, pad_deg=0.3):
    lon_min, lon_max, lat_min, lat_max = bb
    cx = (lon_min + lon_max) / 2.0
    cy = (lat_min + lat_max) / 2.0
    w = max(min_lon_span, (lon_max - lon_min) * factor)
    h = max(min_lat_span, (lat_max - lat_min) * factor)
    return (cx - w/2 - pad_deg, cx + w/2 + pad_deg, cy - h/2 - pad_deg, cy + h/2 + pad_deg)

def _detect_region_label(q: str) -> Optional[str]:
    ql = (q or "").lower()
    # prefer more specific keys first
    keys = sorted(set(list(REGION_CENTER.keys()) + list(REGION_BBOX.keys())), key=len, reverse=True)
    for k in keys:
        if k in ql:
            return k
    return None

### PROCEDURES - Infer Geological Age(s) from text ######################################################################################
def _llm_age_range(q: str) -> tuple[int, int] | None:
    """
    Ask the LLM to extract an age range (older->younger, Ma) with high reliability.
    Handles eras (Paleozoic/Palaeozoic, Mesozoic, Cenozoic, Phanerozoic), periods,
    epochs, and 'Upper/Lower' == 'Late/Early'. Returns None if unsure.
    """
    # Normalize UK spelling in the input to make life easier for the LLM
    q_norm = re.sub(r"\bPalaeo", "Paleo", q, flags=re.I)

    sys = (
        "Extract a geological age RANGE in millions of years before present (Ma). "
        "Accept eras (Paleozoic, Mesozoic, Cenozoic, Phanerozoic), periods (Cambrian…Cretaceous), "
        "epochs (Paleocene…Holocene), and qualifiers Early/Middle/Late (synonyms: Lower/Middle/Upper). "
        "If a numeric range is present (e.g., '201–145 Ma', 'from 201 to 145 Ma'), use it. "
        "If only names are present, convert to numeric using standard ICS boundaries (rounded to ints). "
        "If multiple terms define a span (e.g., 'Early Jurassic to Late Cretaceous'), map each and span them. "
        "If you cannot determine a valid range, return nulls. "
        "Return STRICT JSON only with integers: "
        '{\"start_ma\": <older integer or null>, \"end_ma\": <younger integer or null>}. '
        "No commentary."
    )

    # A few minimal few-shot examples to anchor eras + synonyms
    shots = [
        {"role": "user", "content": "Text: Paleozoic extension in the North Sea"},
        {"role": "assistant", "content": '{"start_ma": 541, "end_ma": 252}'},
        {"role": "user", "content": "Text: Palaeozoic rifting (Lower–Upper Carboniferous)"},
        {"role": "assistant", "content": '{"start_ma": 359, "end_ma": 299}'},
        {"role": "user", "content": "Text: Early Jurassic to Late Cretaceous"},
        {"role": "assistant", "content": '{"start_ma": 201, "end_ma": 66}'},
        {"role": "user", "content": "Text: between 120–80 Ma in the South Atlantic"},
        {"role": "assistant", "content": '{"start_ma": 120, "end_ma": 80}'},
    ]

    try:
        msgs = [{"role":"system","content":sys}] + shots + [{"role":"user","content": f"Text: {q_norm}"}]
        resp = _client().chat.completions.create(
            model=HF_MODEL_ID,
            messages=msgs,
            max_tokens=80,
            temperature=0.0,
            top_p=1.0,
        )
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", txt, flags=re.S)
        js = json.loads(m.group(0) if m else txt)
        a, b = js.get("start_ma"), js.get("end_ma")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return _normalize_ma_pair(a, b)
    except Exception:
        pass
    return None

def infer_age_range_from_query_llm_first(q: str, fallback: tuple[int, int] = (200, 0)) -> tuple[int, int]:
    """
    Prefer LLM extraction; fallback to the current regex/ICS heuristic.
    """
    llm = _llm_age_range(q)
    if llm: 
        return llm
    return infer_age_range_from_text(q, fallback=fallback)

@lru_cache(maxsize=256)
def _geocode_nominatim_bbox(name: str) -> tuple[float, float, float, float] | None:
    """
    Return (lon_min, lon_max, lat_min, lat_max) for a geocoded place.
    """
    if not name: 
        return None
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": name, "format": "jsonv2", "limit": 1}
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        arr = r.json() or []
        if not arr:
            return None
        rec = arr[0]
        bb = rec.get("boundingbox")
        if isinstance(bb, list) and len(bb) == 4:
            south, north, west, east = map(float, bb)
            time.sleep(1.0)  # be polite to Nominatim
            return (west, east, south, north)
    except Exception:
        pass
    return None

def _first_nonempty(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v is not None and v != "" and not isinstance(v, (list, dict)):
            return str(v)
    return ""

def _pluck_nameish(x):
    """Return a human name from str/dict/list structures seen in Macrostrat."""
    if not x:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        for k in ("name", "strat_name", "label", "long_name"):
            if k in x and x[k]:
                return str(x[k]).strip()
        # Sometimes there is a 'names' array
        if "names" in x and isinstance(x["names"], list):
            for n in x["names"]:
                s = _pluck_nameish(n)
                if s:
                    return s
        return ""
    if isinstance(x, list):
        parts = [ _pluck_nameish(i) for i in x ]
        parts = [ p for p in parts if p ]
        # unique, order-preserving
        seen = {}
        return ", ".join([seen.setdefault(p, p) for p in parts if p not in seen])
    return str(x).strip()

def _pluck_lith(x):
    """Flatten lithology fields that may be str/list/dict."""
    if not x:
        return ""
    if isinstance(x, str):
        return x.strip()
    vals = []
    if isinstance(x, dict):
        for k in ("lith", "name", "lith_type", "class", "descrip"):
            v = x.get(k)
            if v:
                vals.append(str(v))
    if isinstance(x, list):
        for i in x:
            vals.append(_pluck_lith(i))
    # unique, tidy
    vals = [v.strip() for v in vals if v and str(v).strip()]
    seen = {}
    return "; ".join([seen.setdefault(v, v) for v in vals if v not in seen])

def infer_age_range_from_text(q: str, fallback: tuple[int, int] = (200, 0)) -> tuple[int, int]:
    """
    Return (start_ma, end_ma) older->younger.
    Now understands eras (Paleozoic/Mesozoic/Cenozoic) including UK 'Palaeozoic'
    and 'early/middle/late' sub-era phrases.
    """
    if not q:
        return fallback
    ql = q.lower()

    # Normalize UK spelling 'palaeo' -> 'paleo'
    ql = re.sub(r"\bpalaeo", "paleo", ql)

    # 1) Explicit numeric ranges
    for pat in _RANGE_PATTERNS:
        m = pat.search(ql)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            start, end = (max(a, b), min(a, b))
            return (start, end)

    # 2) Named sub-periods first (more specific)
    for name, (start, end) in SUBPERIOD_RANGES.items():
        if name in ql:
            return (start, end)

    # 3) Sub-era phrases (e.g., 'early paleozoic')
    for name, (start, end) in SUBERA_RANGES.items():
        if name in ql:
            return (start, end)

    # 4) Periods / epochs
    for name, (start, end) in PERIOD_RANGES.items():
        if name in ql:
            return (start, end)

    # 5) Eras (e.g., 'paleozoic', 'mesozoic', 'cenozoic', 'phanerozoic')
    for name, (start, end) in ERA_RANGES.items():
        if name in ql:
            # keep within 0–540 for services that clamp
            start = int(min(540, max(0, start)))
            end   = int(min(540, max(0, end)))
            return (start, end)

    # 6) Fallback
    return fallback

### PROCEDURES - MACROSTRAT ######################################################################################

def _macrostrat_params_from_query(q: str) -> dict:
    start_ma, end_ma = infer_age_range_from_text(q, fallback=(200, 0))
    older, younger = int(max(start_ma, end_ma)), int(min(start_ma, end_ma))
    age_top, age_bottom = younger, older  # Macrostrat expects: top = younger, bottom = older
    lon, lat = _infer_point_from_query(q)
    region_label = _detect_region_label(q) or next((k for k in COUNTRY_CENTER if k in (q or "").lower()), "unspecified region")
    age_label = f"{older}–{younger} Ma"
    return {"age_top": age_top, "age_bottom": age_bottom, "lat": float(lat), "lng": float(lon),
            "region_label": region_label, "age_label": age_label}

def _macro_cols_url(lat: float, lng: float, dist_km: int = 200) -> str:
    return f"{MACROSTRAT_BASE}/columns?lat={lat:.6f}&lng={lng:.6f}&dist={int(dist_km)}"

def _macro_units_by_col_url(col_id: int, age_top: int, age_bottom: int) -> str:
    return f"{MACROSTRAT_BASE}/units?col_id={int(col_id)}&age_top={int(age_top)}&age_bottom={int(age_bottom)}"

def _macro_map_url(lat: float, lng: float) -> str:
    return f"{MACROSTRAT_BASE}/geologic_units/map?lat={lat:.6f}&lng={lng:.6f}"

def _macro_units_by_map_url(map_id: int, age_top: int, age_bottom: int) -> str:
    return f"{MACROSTRAT_BASE}/units?map_id={int(map_id)}&age_top={int(age_top)}&age_bottom={int(age_bottom)}"

def _first_nonempty(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v not in (None, "", []):
            return str(v)
    return ""

def _pluck_nameish(x):
    if not x: return ""
    if isinstance(x, str): return x.strip()
    if isinstance(x, dict):
        for k in ("name", "strat_name", "label", "long_name"):
            if x.get(k): return str(x[k]).strip()
        if isinstance(x.get("names"), list):
            for n in x["names"]:
                s = _pluck_nameish(n)
                if s: return s
        return ""
    if isinstance(x, list):
        seen = set(); out = []
        for i in x:
            s = _pluck_nameish(i)
            if s and s not in seen:
                seen.add(s); out.append(s)
        return ", ".join(out)
    return str(x).strip()

def _pluck_lith(x):
    if not x: return ""
    if isinstance(x, str): return x.strip()
    vals = []
    if isinstance(x, dict):
        for k in ("lith", "name", "lith_type", "class", "descrip"):
            v = x.get(k)
            if v: vals.append(str(v))
    if isinstance(x, list):
        for i in x: 
            v = _pluck_lith(i)
            if v: vals.append(v)
    seen = set(); out = []
    for v in [v.strip() for v in vals if v and str(v).strip()]:
        if v not in seen:
            seen.add(v); out.append(v)
    return "; ".join(out)

def _rows_from_units(data):
    rows = []
    for it in (data or []):
        unit_name  = _first_nonempty(it.get("name"), it.get("unit_name"), _pluck_nameish(it.get("strat_name")))
        strat_name = _pluck_nameish(it.get("strat_name"))
        best_int   = _first_nonempty(it.get("best_int_name"), it.get("int_name"), it.get("t_int_name"), it.get("b_int_name"))
        display    = _first_nonempty(unit_name, strat_name, best_int)
        t_age = it.get("t_age"); b_age = it.get("b_age")
        lith  = _pluck_lith(it.get("lith") or it.get("liths") or it.get("lith1") or it.get("lith2"))
        source_id = it.get("source_id") or (it.get("source") or {}).get("source_id")
        map_id    = it.get("map_id")    or (it.get("map") or {}).get("map_id")
        rows.append([display, strat_name, best_int, t_age, b_age, lith, source_id, map_id])
    return rows

def _nearby_col_ids(
    lat: float, lng: float, age_top: int, age_bottom: int, dist_km: int = 200,
    session: Optional[requests.Session] = None
) -> list[tuple[int, float]]:
    """
    Use columns at point + adjacents (v2/columns?lat=...&lng=...&adjacents=true)
    and keep only those with centroids within dist_km AND overlapping the age window.
    Returns a sorted list of (col_id, distance_km).
    """
    sess = session or _http_session(retries=2, backoff=0.6, timeout_connect=4.0, timeout_read=12.0)
    url = f"{MACROSTRAT_BASE}/columns"
    params = {
        "lat": f"{lat:.6f}",
        "lng": f"{lng:.6f}",
        "adjacents": "true",             # <-- key change
        "age_top": int(age_top),         # younger
        "age_bottom": int(age_bottom),   # older
    }
    try:
        r = sess.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=sess._timeouts)
        r.raise_for_status()
        js = r.json()
        data = js.get("data") or js.get("success", {}).get("data") or []
    except Exception:
        data = []

    out: list[tuple[int, float]] = []
    for d in (data if isinstance(data, list) else []):
        cid = d.get("col_id")
        clat = d.get("lat") or d.get("clat")  # rmacrostrat lists both; v2 returns lat/lng for columns
        clng = d.get("lng") or d.get("clng")
        if cid is None or clat is None or clng is None:
            continue
        try:
            dk = _haversine_km(lat, lng, float(clat), float(clng))
        except Exception:
            continue
        if dk <= float(dist_km):
            out.append((int(cid), dk))
    out.sort(key=lambda t: t[1])
    return out

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlamb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlamb/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _macro_map_units_url(lat: float, lng: float, fmt: str | None = None) -> str:
    base = f"{MACROSTRAT_BASE}/geologic_units/map"
    q = f"?lat={lat:.6f}&lng={lng:.6f}"
    return base + (q + f"&format={fmt}" if fmt else q)

def _filter_units_by_age(data: list, age_top: int, age_bottom: int) -> list:
    """Keep units whose [t_age (younger), b_age (older)] overlaps [age_top, age_bottom]."""
    out = []
    for it in (data or []):
        t = it.get("t_age")
        b = it.get("b_age")
        if t is None or b is None:
            continue
        # overlap test in Ma (b=older, t=younger)
        if (b >= age_top) and (t <= age_bottom):
            out.append(it)
    return out

def _normalize_ma_pair(a: float | int, b: float | int) -> tuple[int, int]:
    a = int(round(float(a))); b = int(round(float(b)))
    # clamp to the window your app supports (0..540 Ma)
    a = max(0, min(540, a)); b = max(0, min(540, b))
    if a < b: a, b = b, a
    return a, b

# MAIN MACROSTRAT ORCHESTRATION PROC    
def query_macrostrat_from_text(q: str, session: Optional[requests.Session] = None) -> tuple[str, list[list]]:
    P = _macrostrat_params_from_query(q)
    sess = session or _http_session(retries=2, backoff=0.6, timeout_connect=4.0, timeout_read=12.0)

    md_lines = [f"**Detected**: {P['region_label']} · {P['age_label']}  "]
    rows: list[list] = []

    # --- 0) Strict local whitelist of columns (age + radius) ------------------
    dist_km = int(os.getenv("MACRO_DIST_KM", "200"))  # ★ make adjustable via env if you like
    local_cols_info = _nearby_col_ids(P["lat"], P["lng"], P["age_top"], P["age_bottom"], dist_km=dist_km, session=sess)
    local_cols = [cid for (cid, d) in local_cols_info]
    local_col_set = set(local_cols)

    if local_cols_info:
        preview = ", ".join(f"{cid}({d:.0f} km)" for cid, d in local_cols_info[:6])
        md_lines.append(f"**Geo constraint**: {len(local_cols)} column(s) within {dist_km} km → {preview}{' …' if len(local_cols_info) > 6 else ''}  ")
    else:
        md_lines.append(f"**Geo constraint**: no Macrostrat columns within {dist_km} km for this age window  ")

    # --- 1) Point strat column (map endpoint) → age filter → STRICT col_id filter
    if local_col_set:
        try:
            murl = f"{MACROSTRAT_BASE}/geologic_units/map?lat={P['lat']:.6f}&lng={P['lng']:.6f}"
            r = sess.get(murl, headers={"User-Agent": USER_AGENT}, timeout=sess._timeouts)
            r.raise_for_status()
            mj = r.json()
            mdata = mj.get("data") or mj.get("success", {}).get("data") or []
            local_units = _filter_units_by_age(mdata, P["age_top"], P["age_bottom"])
            # ★ keep ONLY units whose col_id is strictly in the local set
            local_units = [u for u in local_units if int(u.get("col_id", -1)) in local_col_set]
            if local_units:
                md_lines.append("**Source**: point strat column (map endpoint), age + strict col filter  ")
                rows.extend(_rows_from_units(local_units))
        except Exception:
            pass

    # --- 2) Fallback: nearest (strictly local) columns → units by col_id ------
    if not rows and local_cols:
        for cid in local_cols[:3]:  # closest few only
            uurl = f"{MACROSTRAT_BASE}/units?col_id={cid}&age_top={P['age_top']}&age_bottom={P['age_bottom']}"
            try:
                rr = sess.get(uurl, headers={"User-Agent": USER_AGENT}, timeout=sess._timeouts)
                rr.raise_for_status()
                uj = rr.json()
                udata = uj.get("data") or uj.get("success", {}).get("data") or []
                rows.extend(_rows_from_units(udata))
            except Exception:
                continue
        if rows:
            md_lines.append("**Source**: nearest strictly-local column(s) → units (age-filtered)  ")

    # de-dup
    seen = set(); uniq = []
    for r in rows:
        key = (r[0], r[3], r[4])
        if key not in seen:
            seen.add(key); uniq.append(r)

    # report the resolved place & point used
    resolved = (_llm_geocodable_places(_client(), q)[:1] or [(q or '').strip() or 'n/a'])[0]
    md_lines.append(f"**Resolved location**: {resolved}")
    md_lines.append(f"**Point used**: lat={P['lat']:.3f}, lng={P['lng']:.3f}")

    # exports (strictly local)
    md_lines.append(
        f"**Exports**: "
        f"[Map JSON]({MACROSTRAT_BASE}/geologic_units/map?lat={P['lat']:.6f}&lng={P['lng']:.6f}) · "
        f"[Map CSV]({MACROSTRAT_BASE}/geologic_units/map?lat={P['lat']:.6f}&lng={P['lng']:.6f}&format=csv) · "
        f"[Columns JSON]({MACROSTRAT_BASE}/columns?lat={P['lat']:.6f}&lng={P['lng']:.6f}&dist={dist_km}&age_top={P['age_top']}&age_bottom={P['age_bottom']})"
    )
    md = "\n".join(md_lines)
    return md, uniq

# PROCEDURES GPLATES #####################################################################################

def make_gif_from_positions(positions, out_path: str | None = None, frame_duration_ms: int = 800) -> str | None:
    if not positions:
        return None

    tmpdir = tempfile.mkdtemp(prefix="gplates_frames_")
    frame_files = []
    try:
        for (t, x, y) in positions:
            fig = plt.figure(figsize=(6, 3), dpi=110)
            fig.patch.set_facecolor("white")
            ax = plt.gca()
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.set_title(f"GPlates reconstruction — {t} Ma")
            ax.grid(True, linewidth=0.3)
            ax.plot([-180, 180, 180, -180, -180], [-90, -90, 90, 90, -90], linewidth=0.5)
            ax.scatter([x], [y], s=80)
            fp = os.path.join(tmpdir, f"frame_{t:04d}.png")
            plt.tight_layout()
            plt.savefig(fp, dpi=110, facecolor="white")
            plt.close(fig)
            frame_files.append(fp)

        frames = []
        for fp in frame_files:
            im = Image.open(fp)
            frames.append(_composite_to_rgb(im))
            im.close()

        if not frames:
            return None

        if out_path is None:
            out_path = os.path.join(tempfile.gettempdir(), f"gplates_{int(time.time())}.gif")
        out_path = os.path.abspath(out_path)

        delay_ms = max(50, int(frame_duration_ms))
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=delay_ms,
            loop=0,
            optimize=False,
            format="GIF",
        )        
        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            print("[GPlates][positions] saved:", out_path, "bytes:", os.path.getsize(out_path))
            return out_path
        return None
    finally:
        try: shutil.rmtree(tmpdir)
        except Exception: pass
        
def _evenly_pick(items, k):
    if not items: return []
    if len(items) <= k: return list(items)
    step = max(1, len(items) // k)
    return items[::step][:k]

def _lonlat_to_px_bbox(w: int, h: int, lon: float, lat: float,
                       bbox: tuple[float, float, float, float]) -> tuple[int, int]:
    """Map lon/lat to pixel in an equirectangular image clipped to bbox=(lon_min,lon_max,lat_min,lat_max)."""
    lon_min, lon_max, lat_min, lat_max = bbox
    x = (lon - lon_min) / max(1e-9, (lon_max - lon_min)) * w
    y = (lat_max - lat) / max(1e-9, (lat_max - lat_min)) * h
    return int(round(x)), int(round(y))

def _classify_boundary(props: dict) -> str:
    """
    Return 'ridge' | 'subduction' | 'transform' | 'other'
    based on common property names/values in GWS outputs.
    """
    if not isinstance(props, dict):
        return "other"
    txt = " ".join(str(v) for v in props.values()).lower()
    if any(k in txt for k in ["subduction", "trench"]):
        return "subduction"
    if any(k in txt for k in ["transform"]):
        return "transform"
    if any(k in txt for k in ["ridge", "spreading"]):
        return "ridge"
    return "other"

def _fetch_plate_boundaries_geojson(
    t: int,
    model: str,
    bbox: tuple[float, float, float, float],
    timeout_connect: float = 2.5,
    timeout_read: float = 5.0
) -> list[tuple[str, list[tuple[float, float]]]]:
    """
    Return a list of (btype, [(lon,lat), ...]) plate-boundary segments.
    Tries Topology API first with robust model-name variants, then
    subduction-only, then static polygons as a last resort.
    """
    import requests

    def _collect_segments_from_features(features):
        segs: list[tuple[str, list[tuple[float, float]]]] = []
        for f in features or []:
            props = f.get("properties") or {}
            geom = f.get("geometry") or {}
            gtype = (geom.get("type") or "").lower()
            btype = _classify_boundary(props)

            if gtype == "linestring":
                line = []
                for c in (geom.get("coordinates") or []):
                    if isinstance(c, (list, tuple)) and len(c) >= 2:
                        line.append((float(c[0]), float(c[1])))
                if len(line) >= 2:
                    segs.append((btype, line))

            elif gtype == "multilinestring":
                for part in (geom.get("coordinates") or []):
                    line = []
                    for c in (part or []):
                        if isinstance(c, (list, tuple)) and len(c) >= 2:
                            line.append((float(c[0]), float(c[1])))
                    if len(line) >= 2:
                        segs.append((btype, line))

            elif gtype in ("polygon", "multipolygon"):
                # Draw polygon rings as 'other' — useful fallback if lines are unavailable
                def _ring_to_line(ring):
                    line = []
                    for c in (ring or []):
                        if isinstance(c, (list, tuple)) and len(c) >= 2:
                            line.append((float(c[0]), float(c[1])))
                    return line if len(line) >= 2 else None

                if gtype == "polygon":
                    for ring in (geom.get("coordinates") or []):
                        L = _ring_to_line(ring);  L and segs.append(("other", L))
                else:  # MultiPolygon
                    for poly in (geom.get("coordinates") or []):
                        for ring in (poly or []):
                            L = _ring_to_line(ring);  L and segs.append(("other", L))
        return segs

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    # Build a robust list of model candidates to try
    cand = []
    if model:
        cand.extend([model, model.title(), model.upper()])
    for m in ["Zahirovic2022", "Muller2019", "Muller2016", "Seton2012", "MULLER2019", "ZAHIROVIC2022"]:
        if m not in cand:
            cand.append(m)

    # Normalized bbox string (can help some endpoints trim server-side payload)
    extent_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    # 1) Topology plate boundaries (preferred)
    for m in cand:
        try:
            r = requests.get(
                "https://gws.gplates.org/topology/plate_boundaries",
                params={"time": int(t), "model": m, "fmt": "geojson", "extent": extent_str, "wrap": "true"},
                headers=headers,
                timeout=(timeout_connect, timeout_read),
            )
            if r.status_code >= 400:
                continue
            js = r.json()
            feats = None
            if isinstance(js, dict):
                feats = js.get("features") or js.get("data") or js.get("reconstructed_features")
            if not feats and isinstance(js, list):
                feats = js
            segs = _collect_segments_from_features(feats or [])
            if segs:
                return segs
        except Exception:
            continue

    # 2) Subduction-only topology (still useful if full boundaries aren’t present)
    for m in cand:
        try:
            r = requests.get(
                "https://gws.gplates.org/topology/get_subduction_zones",
                params={"time": int(t), "model": m, "fmt": "geojson", "extent": extent_str, "wrap": "true"},
                headers=headers,
                timeout=(timeout_connect, timeout_read),
            )
            if r.status_code >= 400:
                continue
            js = r.json()
            feats = (js.get("features") if isinstance(js, dict) else None) or []
            segs = _collect_segments_from_features(feats)
            if segs:
                # mark as subduction explicitly
                return [("subduction", line) for (_, line) in segs]
        except Exception:
            continue

    # 3) Fallback: static polygons outlines (works for all models)
    for m in cand:
        try:
            r = requests.get(
                "https://gws.gplates.org/reconstruct/static_polygons",
                params={"time": int(t), "model": m, "fmt": "geojson", "extent": extent_str, "wrap": "true"},
                headers=headers,
                timeout=(timeout_connect, timeout_read),
            )
            if r.status_code >= 400:
                continue
            js = r.json()
            feats = (js.get("features") if isinstance(js, dict) else None) or []
            segs = _collect_segments_from_features(feats)
            if segs:
                return segs
        except Exception:
            continue

    return []

def _draw_boundaries_on_image(
    im: "Image.Image",
    segments: list[tuple[str, list[tuple[float, float]]]],
    bbox: tuple[float, float, float, float],
    width: int = 3
) -> None:
    """
    Draw colored boundaries on a PIL image in-place.
    """
    from PIL import ImageDraw
    color_map = {
        "ridge":       (255, 90, 60, 255),    # red/orange
        "subduction":  (50, 110, 255, 255),   # blue
        "transform":   (240, 200, 40, 255),   # yellow
        "other":       (130, 130, 130, 255),  # gray
    }
    d = ImageDraw.Draw(im, mode="RGBA")
    w, h = im.size

    for btype, line in segments:
        col = color_map.get(btype, color_map["other"])
        # convert lon/lat to px and draw a polyline
        pts = [_lonlat_to_px_bbox(w, h, lon, lat, bbox) for (lon, lat) in line]
        # Draw as short segments to avoid great-circle artifacts on bbox edges
        for i in range(1, len(pts)):
            d.line([pts[i-1], pts[i]], fill=col, width=width)

            
def create_local_placeholder_gif(user_query: str) -> str:
    """Always returns a small local GIF with a *moving* marker (no network)."""
    w, h = 640, 320
    lon0, lat0 = _infer_point_from_query(user_query)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # 4 frames ~150→0 Ma; synthesize a small drift so the point moves
    ages = [150, 100, 50, 0]
    frames = []
    for i, t in enumerate(ages):
        # synthetic drift (deg): ~0.15° lon and 0.07° lat per 10 Ma
        drift_lon = -0.15 * (t / 10.0)
        drift_lat =  0.07 * (t / 10.0)
        lon = ((lon0 + drift_lon + 540) % 360) - 180
        lat = max(-85, min(85, lat0 + drift_lat))

        # draw
        img = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([5, 5, w-5, h-5], outline=(0, 0, 0, 255), width=2)
        x = int((lon + 180) / 360 * w)
        y = int(h - ((lat + 90) / 180 * h))
        d.ellipse([x-6, y-6, x+6, y+6], fill=(0, 0, 0, 255))
        if font:
            d.text((10, 10), f"t={t} Ma  ({lon:.1f},{lat:.1f})", fill=(0,0,0,255), font=font)
        frames.append(img)

    tmp = NamedTemporaryFile(delete=False, suffix=".gif")
    path = os.path.abspath(tmp.name); tmp.close()
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=350, loop=0, disposal=2)
    return path if os.path.isfile(path) and os.path.getsize(path) > 0 else ""

def _find_coords_anywhere(obj):
    if isinstance(obj, dict):
        if "coordinates" in obj and isinstance(obj["coordinates"], (list, tuple)) and len(obj["coordinates"]) >= 2:
            return obj["coordinates"][:2]
        for v in obj.values():
            c = _find_coords_anywhere(v)
            if c: return c
    elif isinstance(obj, list):
        for v in obj:
            c = _find_coords_anywhere(v)
            if c: return c
    return None
    
def _composite_to_rgb(img: Image.Image) -> Image.Image:
    """
    Composite any transparency onto a white background and return RGB.
    Handles RGBA, LA, and P (palette) images with transparency.
    """
    if img.mode == "P":
        # If palette has transparency, convert to RGBA first
        if "transparency" in img.info:
            img = img.convert("RGBA")
        else:
            return img.convert("RGB")

    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        alpha = img.split()[-1]
        bg.paste(img, mask=alpha)
        return bg.convert("RGB")

    return img.convert("RGB")

# MAIN GPLATES ORCHESTRATION PROC
def build_gplates_animation_gif_fast(
    user_query: str,
    start_ma: int = 140,
    end_ma: int = 66,
    max_frames: int = 6,
    model: str = "ZAHIROVIC2022",
    extent: tuple[float, float, float, float] | None = None,  # (lon_min, lon_max, lat_min, lat_max)
    per_request_connect: float = 2.5,
    per_request_read: float = 5.0,
    global_deadline_s: float = 15.0,
    concurrency: int = 4,
    show_point: bool = True,
    show_boundaries: bool = True,
    frame_duration_ms: int = 800,   # ↑ bigger = slower playback
    show_age_range: bool = True,    # keep label
    step_ma: float | None = None,   
) -> str:
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def _coast_png_url(t, extent_str):
        return (
            "https://gws.gplates.org/reconstruct/coastlines/"
            f"?time={int(t)}&model={model}&fmt=png"
            f"&edgecolor=black&alpha=1&wrap=true&extent={extent_str}"
        )

    def _recon_point(lon, lat, t):
        url = (
            "https://gws.gplates.org/reconstruct/reconstruct_points/"
            f"?points={lon:.6f},{lat:.6f}&time={int(t)}&model={model}"
        )
        try:
            r = requests.get(url, timeout=(per_request_connect, per_request_read))
            r.raise_for_status()
            js = r.json()
            coords = (js.get("coordinates") or [])
            if coords and isinstance(coords[0], (list, tuple)) and len(coords[0]) >= 2:
                return float(coords[0][0]), float(coords[0][1])
        except Exception:
            pass
        return None

    def _text_badge(draw: "ImageDraw.ImageDraw", im: "Image.Image", text: str, corner: str = "tr", pad: int = 8):
        if not text:
            return
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = _measure_text(draw, text, font=font)
        W, H = im.size
        if corner == "tr":
            x, y = W - tw - pad - 6, pad
        elif corner == "br":
            x, y = W - tw - pad - 6, H - th - pad - 4
        elif corner == "bl":
            x, y = pad, H - th - pad - 4
        else:
            x, y = pad, pad
        draw.rectangle([x - 4, y - 2, x + tw + 4, y + th + 2], fill=(255, 255, 255, 180))
        draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)

    # ---------------- Deterministic integer ages incl. endpoints ----------------
    if step_ma is None:
        span = max(1, int(round(float(start_ma) - float(end_ma))))
        step_ma = max(1, int(math.ceil(span / max(2, int(max_frames)))))
    else:
        step_ma = max(1, int(round(step_ma)))

    ages = list(range(int(start_ma), int(end_ma) - 1, -int(step_ma)))
    if not ages or ages[-1] != int(end_ma):
        ages.append(int(end_ma))

    # Label (older → younger)
    age_range_label = f"{int(max(ages))}–{int(min(ages))} Ma" if ages else ""
    # ---------------------------------------------------------------------------

    # extent: infer from query or global
    bbox = extent or _infer_bbox_from_query(user_query) or (-180.0, 180.0, -90.0, 90.0)
    extent_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    # present-day point for reconstruction per frame
    try:
        lon0, lat0 = _infer_point_from_query(user_query)
    except Exception:
        lon0, lat0 = (0.0, 0.0)

    start_time = time.time()
    results: dict[int, Image.Image] = {}

    def fetch_frame(t: int):
        try:
            # 1) Coastlines base (transparent PNG)
            url = _coast_png_url(t, extent_str)
            r = requests.get(url, timeout=(per_request_connect, per_request_read))
            r.raise_for_status()
            base = Image.open(BytesIO(r.content)).convert("RGBA")
            d = ImageDraw.Draw(base, mode="RGBA")

            # 2) Plate boundaries
            if show_boundaries:
                segs = _fetch_plate_boundaries_geojson(
                    t=t, model=model, bbox=bbox,
                    timeout_connect=per_request_connect, timeout_read=per_request_read
                )
                if segs:
                    _draw_boundaries_on_image(base, segs, bbox, width=max(2, base.width // 512 + 2))

            # 3) Labels: frame time & age range
            _text_badge(d, base, f"t={int(t)} Ma", corner="bl")
            if show_age_range and age_range_label:
                _text_badge(d, base, age_range_label, corner="tr")
    
            # 4) Reconstructed query point
            if show_point:
                xy = _recon_point(lon0, lat0, t)
                if xy:
                    xpix, ypix = _lonlat_to_px_bbox(base.width, base.height, xy[0], xy[1], bbox)
                    rpx = max(3, base.width // 150)
                    d.ellipse([xpix - rpx, ypix - rpx, xpix + rpx, ypix + rpx],
                              fill=(220, 50, 50, 255), outline=(255, 255, 255, 255))
            return (t, base)
        except Exception:
            return (t, None)

    # --------- Concurrent fetch with overall deadline ---------
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(fetch_frame, int(t)): int(t) for t in ages}
        while futs and (time.time() - start_time) < global_deadline_s:
            done, pending = concurrent.futures.wait(
                futs, timeout=0.4, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for f in done:
                t, im = f.result()
                if im:
                    results[int(t)] = im
            futs = {p: futs[p] for p in pending}

    # --------- Ensure we have endpoints (start & end) ---------
    got = {int(round(t)) for t in results.keys()}
    need = {int(start_ma), int(end_ma)}
    missing_endpoints = sorted(list(need - got), reverse=True)
    for t in missing_endpoints:
        t_back, im_back = fetch_frame(int(t))
        if im_back:
            results[int(t_back)] = im_back

    # --------- NEW: Smooth-fill any missing ages within a small budget ---------
    fill_missing_after_deadline_s = 12.0  # adjust if you want more/less smoothing time
    fill_deadline = time.time() + fill_missing_after_deadline_s
    missing_all = [int(t) for t in ages if int(t) not in results]
    for t in missing_all:
        if time.time() > fill_deadline:
            break
        t2, im2 = fetch_frame(int(t))
        if im2:
            results[int(t2)] = im2
    # ---------------------------------------------------------------------------

    if not results:
        return ""

    frames = [results[t] for t in sorted(results.keys(), reverse=True)]
    w, h = frames[0].size
    frames = [f.resize((w, h)) for f in frames]

    tmp = NamedTemporaryFile(delete=False, suffix=".gif")
    gif_path = os.path.abspath(tmp.name); tmp.close()

    delay_ms = max(20, int(frame_duration_ms))  # GIF expects ms; clamp to 20ms
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=delay_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )    
    return gif_path if os.path.isfile(gif_path) and os.path.getsize(gif_path) > 0 else ""

### MINDAT MINERAL PROC #####################################################################
# ==================== MINDAT OCCURRENCES PANEL ====================

def _mindat_headers():
    h = {"User-Agent": USER_AGENT}
    if MINDAT_API_TOKEN:
        h["Authorization"] = f"Token {MINDAT_API_TOKEN}"
    return h

@lru_cache(maxsize=1)
def _mindat_country_names() -> list[str]:
    """
    Fetch country names from Mindat (fallback to a small list if offline).
    """
    url = f"{MINDAT_BASE}/countries"
    try:
        r = requests.get(url, params={"size": 300}, headers=_mindat_headers(), timeout=12)
        r.raise_for_status()
        js = r.json()
        items = js.get("results") or js.get("data") or js if isinstance(js, list) else (js.get("results") or js.get("data") or [])
        names = []
        for rec in (items if isinstance(items, list) else []):
            nm = (rec.get("name") or rec.get("country") or "").strip()
            if nm:
                names.append(nm)
        names = sorted(set(names), key=str.lower)
        return names if names else []
    except Exception:
        # small fallback set
        return ["United States", "United Kingdom", "Canada", "Australia", "Norway", "Sweden",
                "Finland", "Germany", "France", "Spain", "Portugal", "Italy",
                "Brazil", "Argentina", "South Africa", "Namibia", "Gabon", "Angola",
                "Nigeria", "Ghana", "Morocco", "Algeria", "Peru", "South Africa", "Libya", "Egypt", "China", "India"]

    
def _mindat_detect_country(text: str) -> Optional[str]:
    q = (text or "").lower()
    for nm in _mindat_country_names():
        if nm and nm.lower() in q:
            return nm
    return None

def _mindat_try_geomaterial_search(name: str) -> tuple[Optional[int], Optional[str]]:
    """
    Resolve a mineral/geomaterial to Mindat id.
    Try '/geomaterials' with q=..., then '/geomaterials/search' if needed.
    """
    sess = _http_session(retries=2, backoff=0.6, timeout_connect=4.0, timeout_read=10.0)
    for path in ("geomaterials", "geomaterials/search"):
        try:
            r = sess.get(f"{MINDAT_BASE}/{path}",
                         params={"q": name, "e": 1, "ima": 1, "size": 1, "fields": "id,name"},
                         headers=_mindat_headers(), timeout=sess._timeouts)
            if r.status_code >= 400:
                continue
            js = r.json()
            items = js.get("results") or js.get("data") or js
            if isinstance(items, dict):  # sometimes dict with 'results'
                items = items.get("results") or []
            if not isinstance(items, list):
                continue
            if items:
                mid = items[0].get("id")
                mname = items[0].get("name")
                if mid and mname:
                    return int(mid), str(mname)
        except Exception:
            continue
    return None, None

def _mindat_detect_mineral(text: str, country_hint: Optional[str] = None) -> tuple[Optional[int], Optional[str]]:
    t = text or ""
    if country_hint:
        t = re.sub(re.escape(country_hint), " ", t, flags=re.I)

    # Grab candidate tokens (Proper-Case & lowercase words)
    proper = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,2}\b", t)
    lowers = [w for w in re.findall(r"[a-z]{2,}", t.lower()) if w not in STOPWORDS]

    # Include chemical symbols if present
    symbols = re.findall(r"\b([A-Za-z]{1,2})\b", t)
    for s in symbols:
        k = s.lower()
        if k in _SYMBOL_TO_MINERAL:
            lowers.append(_SYMBOL_TO_MINERAL[k])

    # Apply fixups (typos → correct names)
    fixed = []
    for w in lowers:
        fixed.append(_MINERAL_FIXUPS.get(w, w))
    # Deduplicate while keeping order
    seen = set(); cands = []
    for s in proper + [w.capitalize() for w in fixed]:
        k = s.lower()
        if len(k) < 3 or k in seen:
            continue
        seen.add(k)
        cands.append(s)

    # Resolve against Mindat
    for cand in cands[:30]:
        mid, mname = _mindat_try_geomaterial_search(cand)
        if mid:
            return mid, mname
    return None, None

def _mindat_parse_mineral_or_commodity(q: str) -> tuple[Optional[str], Optional[tuple[str,str]]]:
    """Return (mineral_name, commodity_tuple=(element_symbol, commodity_name))"""
    text = (q or "").lower()
    # tolerate simple typos like 'coper'
    if re.search(r"\bcop+er\b", text):
        return None, MINDAT_COMMODITY_ALIASES["copper"]
    for k, v in MINDAT_COMMODITY_ALIASES.items():
        if k in text:
            return None, v
    # fall back: try a proper mineral name the user typed (e.g., 'chalcopyrite')
    m = re.search(r"\b(chalcopyrite|bornite|chalcocite|tennantite|covellite|enargite|malachite|azurite|native copper)\b", text)
    if m:
        name = m.group(1).title()
        if name == "Native Copper":  # keep species if they explicitly asked
            return "Native Copper", None
        return name, None
    return None, None

def _mindat_loc_points_by_country_and_mineral(country: str, mineral_id: int, mineral_name: str) -> list[dict]:
    """
    Prefer /locentries?country=&geomaterial= (occurrence pairs).
    Fallback to /localities?country=&expand=geomaterials and filter client-side.
    Returns [{'lat':..,'lon':..,'name':..}, ...]
    """
    pts: list[dict] = []
    sess = _http_session(retries=2, backoff=0.6, timeout_connect=4.0, timeout_read=12.0)

    # --- Primary: locentries (locality–geomaterial pairs)
    try:
        url = f"{MINDAT_BASE}/locentries"
        params = {
            "country": country,
            "geomaterial": int(mineral_id),
            "size": int(MINDAT_SIZE_LIMIT),
            "fields": "id,latitude,longitude,locality,id_locality,locality_name"
        }
        r = sess.get(url, params=params, headers=_mindat_headers(), timeout=sess._timeouts)
        if r.status_code < 400:
            js = r.json()
            items = js.get("results") or js.get("data") or js
            if isinstance(items, dict):
                items = items.get("results") or []
            for rec in (items or []):
                lat = rec.get("latitude") or (rec.get("locality") or {}).get("latitude")
                lon = rec.get("longitude") or (rec.get("locality") or {}).get("longitude")
                nm  = rec.get("locality_name") or (rec.get("locality") or {}).get("name")
                try:
                    if lat is not None and lon is not None:
                        pts.append({"lat": float(lat), "lon": float(lon), "name": nm or ""})
                except Exception:
                    continue
    except Exception:
        pass

    # --- Fallback: localities expanded with geomaterials; filter
    if not pts:
        try:
            url = f"{MINDAT_BASE}/localities"
            params = {
                "country": country,
                "expand": "geomaterials",
                "size": int(MINDAT_SIZE_LIMIT),
                "fields": "id,name,latitude,longitude,geomaterials"
            }
            r = sess.get(url, params=params, headers=_mindat_headers(), timeout=sess._timeouts)
            if r.status_code < 400:
                js = r.json()
                items = js.get("results") or js.get("data") or js
                if isinstance(items, dict):
                    items = items.get("results") or []
                for rec in (items or []):
                    glist = rec.get("geomaterials") or []
                    match = any(
                        (isinstance(g, dict) and (g.get("id") == mineral_id or str(g.get("name","")).lower() == str(mineral_name or "").lower()))
                        for g in glist
                    )
                    if not match:
                        continue
                    lat, lon = rec.get("latitude"), rec.get("longitude")
                    nm = rec.get("name") or ""
                    try:
                        if lat is not None and lon is not None:
                            pts.append({"lat": float(lat), "lon": float(lon), "name": nm})
                    except Exception:
                        continue
        except Exception:
            pass

    # de-dup near-identical points (within ~1e-4 deg)
    seen = set(); uniq = []
    for p in pts:
        key = (round(p["lat"], 4), round(p["lon"], 4))
        if key not in seen:
            seen.add(key); uniq.append(p)
    return uniq

# --- Mindat proc: country extraction + simple map drawer -------------------

def _extract_country(q: str) -> str | None:
    """
    Best-effort country name from free text.
    1) quick synonyms (handles 'US', 'UK', etc.)
    2) pycountry (if installed)
    3) minimal fallback list (includes Peru)
    Returns a title-cased country string or None.
    """
    if not q:
        return None
    txt = q.lower()

    # 1) handy synonyms / common spellings
    synonyms = {
        "usa": "United States", "us": "United States", "u.s.": "United States",
        "united states": "United States", "america": "United States",
        "uk": "United Kingdom", "u.k.": "United Kingdom", "britain": "United Kingdom",
        "great britain": "United Kingdom", "england": "United Kingdom",
        "scotland": "United Kingdom", "wales": "United Kingdom",
        "drc": "Democratic Republic of the Congo",
        "dr congo": "Democratic Republic of the Congo",
        "democratic republic of the congo": "Democratic Republic of the Congo",
        "congo": "Republic of the Congo",
        "peru": "Peru", "perú": "Peru",
        "mexico": "Mexico", "méxico": "Mexico",
        "chile": "Chile", "argentina": "Argentina", "brazil": "Brazil",
        "canada": "Canada", "australia": "Australia",
        "south africa": "South Africa", "namibia": "Namibia", "gabon": "Gabon", "angola": "Angola",
        "nigeria": "Nigeria", "ghana": "Ghana", "morocco": "Morocco", "algeria": "Algeria",
        "libya": "Libya", "egypt": "Egypt", "spain": "Spain", "france": "France",
        "germany": "Germany", "italy": "Italy", "norway": "Norway", "sweden": "Sweden",
        "finland": "Finland", "china": "China", "india": "India",
    }
    for k, v in synonyms.items():
        if f" {k} " in f" {txt} ":
            return v

    # 2) try pycountry if available (broad coverage)
    try:
        import pycountry  # type: ignore
        for c in pycountry.countries:
            names = {c.name.lower()}
            for attr in ("official_name", "common_name"):
                if hasattr(c, attr):
                    names.add(getattr(c, attr).lower())
            if any(f" {n} " in f" {txt} " for n in names):
                return c.name
    except Exception:
        pass

    # 3) minimalist fallback scan (keeps you covered if pycountry isn't present)
    fallback = [v for v in set(synonyms.values())]
    for n in fallback:
        if f" {n.lower()} " in f" {txt} ":
            return n

    return None

# ==================== LLM-powered Mindat parsing + fetch ====================

MINDAT_BASE = "https://api.mindat.org/v1"
MINDAT_API_TOKEN = (os.getenv("MINDAT_API_TOKEN") or "").strip()

def _measure_text(draw, text: str, font=None) -> tuple[int, int]:
    """
    Robust text measurement across Pillow versions.
    Tries draw.textbbox → font.getbbox → font.getsize → heuristic.
    """
    try:
        # Pillow ≥8.0
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        pass
    try:
        # Pillow ≥8.0 (font method)
        bbox = font.getbbox(text) if font else None
        if bbox:
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])
    except Exception:
        pass
    try:
        # Older fallback (still exists on many builds)
        return font.getsize(text) if font else (len(text) * 6, 12)
    except Exception:
        # Heuristic last resort
        return (max(1, len(text)) * 6, 12)
    

def _mindat_localities_by_elements(country: str, elements: list[str], token: str,
                                   limit=10000, page_size=300) -> list[dict]:
    cid = _mindat_country_id(country, token)
    out, seen = [], set()
    for el in (elements or []):
        params = {"hascoords": 1, "elements_inc": el}
        if cid is not None: params["country_id"] = cid
        else:               params["country"]    = country
        rows = _mindat_fetch_all("localities", params, token, limit=limit, page_size=page_size)
        for r in rows:
            k = r.get("id")
            if k in seen: continue
            seen.add(k); out.append(r)
    return out

def _mindat_localities_by_mineral(country: str, mineral_name: str, token: str,
                                  limit=10000, page_size=200) -> list[dict]:
    gm_id = _mindat_geomaterial_id(mineral_name, token)
    if gm_id is None: return []
    cid = _mindat_country_id(country, token)
    params = {"hascoords": 1, "geomaterial_id": gm_id}
    if cid is not None: params["country_id"] = cid
    else:               params["country"]    = country
    return _mindat_fetch_all("localities", params, token, limit=limit, page_size=page_size)

# LLM Mindat PROCs
# ==================== LLM-powered Mindat parsing + fetch ====================

MINDAT_BASE = "https://api.mindat.org/v1"
MINDAT_API_TOKEN = (os.getenv("MINDAT_API_TOKEN") or "").strip()

def _mindat_llm_extract(query: str) -> dict:
    sys = (
        "Extract exactly one present-day COUNTRY and either a COMMODITY or a MINERAL from the user's text.\n"
        "- Generic metals/symbols (copper, Cu, Pb-Zn) → kind=commodity with elements array.\n"
        "- Species name (chalcopyrite, bornite, native copper) → kind=mineral.\n"
        "- Prefer ISO 3166 country names.\n"
        "- If both appear, prefer commodity for 'copper deposits', 'Cu deposits'.\n"
        "Respond JSON only: {country, kind, name, elements}."
    )
    shots = [
        {"role":"user","content":"Discuss the copper deposits of Peru"},
        {"role":"assistant","content":'{"country":"Peru","kind":"commodity","name":"Copper","elements":["Cu"]}'},
        {"role":"user","content":"Native copper occurrences in Chile"},
        {"role":"assistant","content":'{"country":"Chile","kind":"mineral","name":"Native Copper","elements":[]}'},
    ]
    try:
        resp = _client().chat.completions.create(
            model=HF_MODEL_ID, temperature=0.0, max_tokens=180,
            messages=[{"role":"system","content":sys}] + shots + [{"role":"user","content": query}],
        )
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", txt, flags=re.S); js = json.loads(m.group(0) if m else txt)
        return {
            "country": (js.get("country") or "").strip(),
            "kind": (js.get("kind") or "").strip().lower(),
            "name": (js.get("name") or "").strip(),
            "elements": [e.strip() for e in (js.get("elements") or []) if isinstance(e, str) and e.strip()],
        }
    except Exception:
        return {"country":"", "kind":"", "name":"", "elements":[]}

def _mindat_country_id(country_name: str, token: str) -> Optional[int]:
    try:
        r = requests.get(f"{MINDAT_BASE}/countries",
                         params={"name": country_name, "page_size": 1},
                         headers=_mindat_headers(), timeout=12)
        r.raise_for_status()
        arr = (r.json() or {}).get("results") or []
        return int(arr[0]["id"]) if arr else None
    except Exception:
        return None
    

def _mindat_geomaterial_id(mineral_name: str, token: str) -> Optional[int]:
    try:
        r = requests.get(f"{MINDAT_BASE}/geomaterials",
                         params={"q": mineral_name, "page_size": 1, "fields":"id,name"},
                         headers=_mindat_headers(), timeout=12)
        r.raise_for_status()
        arr = (r.json() or {}).get("results") or []
        return int(arr[0]["id"]) if arr else None
    except Exception:
        return None
    
def _mindat_points(country: str, kind: str, name: str, elements: list[str], token: str):
    pts, label = [], ""
    if kind == "commodity" and elements:
        rows = _mindat_localities_by_elements(country, elements, token)
        for L in rows:
            lat, lon = L.get("latitude"), L.get("longitude")
            if lat is not None and lon is not None:
                pts.append((float(lon), float(lat), L.get("name","")))
        label = f"{name} commodity ({', '.join(elements)})"
    elif kind == "mineral" and name:
        rows = _mindat_localities_by_mineral(country, name, token)
        for L in rows:
            lat, lon = L.get("latitude"), L.get("longitude")
            if lat is not None and lon is not None:
                pts.append((float(lon), float(lat), L.get("name","")))
        label = name
        if not pts and name.lower() in ("copper","native copper"):
            rows = _mindat_localities_by_elements(country, ["Cu"], token)
            for L in rows:
                lat, lon = L.get("latitude"), L.get("longitude")
                if lat is not None and lon is not None:
                    pts.append((float(lon), float(lat), L.get("name","")))
            label = "Copper commodity (fallback Cu)"
    return pts, label

def _projector_from_bbox(W: int, H: int, bbox: tuple[float, float, float, float]):
    """
    Return a function (lon,lat)->(x,y) that FITS the bbox into W×H
    while preserving aspect ratio (adds margins as needed).
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_span = max(1e-9, lon_max - lon_min)
    lat_span = max(1e-9, lat_max - lat_min)

    sx = W / lon_span
    sy = H / lat_span
    s = min(sx, sy)  # fit inside (letterbox)

    offx = (W - s * lon_span) / 2.0
    offy = (H - s * lat_span) / 2.0

    def to_px(lon: float, lat: float) -> tuple[int, int]:
        x = offx + (lon - lon_min) * s
        y = offy + (lat_max - lat) * s     # north at top
        return int(round(x)), int(round(y))

    return to_px

def mindat_panel_from_query(q: str, return_groups: bool = False):
    if not MINDAT_API_TOKEN:
        return (None, "Set environment variable MINDAT_API_TOKEN to enable the Mindat panel.", []) if return_groups else (None, "Set environment variable MINDAT_API_TOKEN to enable the Mindat panel.")

    info = _mindat_llm_extract_multi(q)
    country = (info.get("country") or "").strip()
    targets = info.get("targets") or []

    if not country:
        return (None, "Mindat: no country detected in your query.", []) if return_groups else (None, "Mindat: no country detected in your query.")
    if not targets:
        return (None, "Mindat: no mineral/commodity detected.", []) if return_groups else (None, "Mindat: no mineral/commodity detected.")

    groups = _mindat_points_multi(country, targets, MINDAT_API_TOKEN)
    if not groups:
        return (None, f"Mindat: no occurrences found in **{country}** (or none with coordinates).", []) if return_groups else (None, f"Mindat: no occurrences found in **{country}** (or none with coordinates).")

    labels = ", ".join(g["label"] for g in groups)
    img = _draw_points_on_country_map(country, groups, title=f"Mindat: {labels} in {country}",
                                  imagery="esri")   # "maptiler" if you export MAPTILER_KEY

    parts = [f"**{g['label']}**: {g['count']}" for g in groups]
    md = f"**Mindat query** → Country: **{country}**  \n" + "  •  ".join(parts)

    return (img, md, groups) if return_groups else (img, md)

def _mindat_groups_to_csv(groups: list[dict]) -> str:
    import csv, os, time, tempfile
    rows = []
    for g in groups or []:
        series = g.get("label","")
        for p in g.get("points", []):
            lon = p.get("longitude") or p.get("lon")
            lat = p.get("latitude")  or p.get("lat")
            nm  = p.get("name","")
            rows.append({"series": series, "locality": nm, "longitude": lon, "latitude": lat})

    out = os.path.join(tempfile.gettempdir(), f"mindat_occurrences_{int(time.time())}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["series","locality","longitude","latitude"])
        w.writeheader()
        w.writerows(rows)
    return out

# --- Mindat pagination proc -----------------------------------------------
def _mindat_fetch_all(endpoint: str,
                      base_params: dict,
                      token: str,
                      limit: int = 10000,
                      page_size: int = 100,
                      sleep_s: float = 0.25) -> list[dict]:
    headers = {"Authorization": f"Token {token}", "User-Agent": USER_AGENT}
    url = f"{MINDAT_BASE}/{endpoint}"
    params = dict(base_params)
    params["page"] = 1
    params["page_size"] = min(int(page_size), 500)

    out, seen = [], set()
    backoff = 0.6
    while url and len(out) < limit:
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            r.raise_for_status()
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code in (500, 502, 503, 504):           # soften on server hiccups
                params = params or {}
                new_ps = max(25, int((params.get("page_size") or page_size) // 2))
                if new_ps < (params.get("page_size") or page_size):
                    params["page_size"] = new_ps
                time.sleep(backoff); backoff = min(2.0, backoff * 1.7)
                continue
            raise
        js = r.json() or {}
        rows = js.get("results") or js.get("data") or []
        for row in rows:
            key = row.get("id") or (row.get("latitude"), row.get("longitude"), row.get("name"))
            if key in seen: 
                continue
            seen.add(key); out.append(row)
            if len(out) >= limit: break
        url = js.get("next") or None      # absolute URL after first page
        params = None
        if sleep_s: time.sleep(sleep_s)
    return out

# MINDAT Country Outlines
# --- Country outline via Nominatim (GeoJSON) ---------------------------------
from functools import lru_cache

@lru_cache(maxsize=64)
def _nominatim_country_outline_rings(country: str) -> list[list[tuple[float, float]]]:
    """Return list of rings; each is [(lon,lat), ...] from OSM/Nominatim."""
    if not country:
        return []

    def _normalize_lonlat(a: float, b: float) -> tuple[float, float]:
        a, b = float(a), float(b)
        # Only swap when it CLEARLY looks like (lat, lon)
        #  - first is in latitude range AND second is NOT in latitude range
        #  - or first is clearly not a longitude while second is
        if (-90.0 <= a <= 90.0) and not (-90.0 <= b <= 90.0) and (-180.0 <= b <= 180.0):
            return b, a
        if not (-180.0 <= a <= 180.0) and (-180.0 <= b <= 180.0):
            return b, a
        return a, b  # assume GeoJSON's (lon, lat)

    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": country, "format": "jsonv2", "polygon_geojson": 1, "polygon_threshold": 0.0008},
            headers={"User-Agent": USER_AGENT}, timeout=20,
        )
        r.raise_for_status()
        arr = r.json() or []
        g = arr and (arr[0].get("geojson") or arr[0].get("polygon_geojson"))
        if not g:
            return []

        rings: list[list[tuple[float, float]]] = []

        def _add_ring(coords):
            ring = []
            for c in coords or []:
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    lon, lat = _normalize_lonlat(c[0], c[1])
                    ring.append((lon, lat))
            if len(ring) >= 2:
                rings.append(ring)

        t = (g.get("type") or "").lower()
        if t == "polygon":
            for ring in g.get("coordinates", []):
                _add_ring(ring)
        elif t == "multipolygon":
            for poly in g.get("coordinates", []):
                for ring in poly or []:
                    _add_ring(ring)

        time.sleep(1.0)  # be polite to Nominatim
        return rings
    except Exception:
        return []
    
def _decimate_points(seq: list[tuple[float, float]], max_points: int = 2000) -> list[tuple[float, float]]:
    """Keep at most max_points by stepping; avoids super heavy outlines."""
    n = len(seq)
    if n <= max_points:
        return seq
    step = max(1, n // max_points)
    return seq[::step]


def _draw_points_on_country_map(country: str,
                                points_or_groups: list,
                                title: str = "",
                                size: tuple[int, int] = (900, 520),
                                clamp_deg: float = 3.0,
                                point_radius: int = 4,
                                imagery: str | None = "esri") -> str | None:
    """
    Robust, centered, aspect-filled country map with legend.
    Accepts either a flat list of points or a list of groups:
      - points: [(lon,lat), ...]  OR  [{"longitude":..,"latitude":..,"name":..}, ...]
      - groups: [{"label":str, "points":[...]} , ...]
    """
    import os, time, tempfile, math
    from PIL import Image, ImageDraw, ImageFont

    def _measure(draw, text, font):
        try:
            b = draw.textbbox((0, 0), text, font=font); return (b[2]-b[0], b[3]-b[1])
        except Exception:
            try: return font.getsize(text)
            except Exception: return (len(text)*6, 12)

    def _text_with_bg(draw, xy, text, font, pad=6, fg=(255,255,255), bg=(0,0,0,150)):
        x, y = xy
        try:
            w, h = draw.textbbox((0, 0), text, font=font)[2:]
        except Exception:
            w, h = font.getsize(text) if font else (len(text)*6, 12)
        # background box
        draw.rectangle([x - pad, y - pad, x + w + pad, y + h + pad], fill=bg)
        # text
        draw.text((x, y), text, fill=fg, font=font)
    
    def _norm_pair(a, b):
        a = float(a); b = float(b)
        # Looks like (lat, lon) → swap
        if (-90.0 <= a <= 90.0) and (-180.0 <= b <= 180.0) and (abs(b) > 90.0):
            return b, a
        # First clearly not a longitude but second is → swap
        if not (-180.0 <= a <= 180.0) and (-180.0 <= b <= 180.0):
            return b, a
        # Otherwise assume already (lon, lat)
        return a, b

    def _as_lonlat(p):
        if isinstance(p, dict):
            lon = p.get("lon", p.get("lng", p.get("longitude")))
            lat = p.get("lat", p.get("latitude"))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            lon, lat = p[0], p[1]
        else:
            return None
        try:
            return _norm_pair(lon, lat)
        except Exception:
            return None

    def _flatten_points(groups_):
        pts = []
        for g in groups_:
            for p in g["points"]:
                pair = _as_lonlat(p)
                if pair: pts.append(pair)
        return pts

    def _bbox_from_pairs(arr):
        if not arr: return None
        lons = [x for (x, _) in arr]; lats = [y for (_, y) in arr]
        return (min(lons), max(lons), min(lats), max(lats))

    def _quantile(vals, q):
        vals = sorted(vals); n = len(vals)
        if n == 0: return 0.0
        if n == 1: return vals[0]
        pos = q*(n-1); i = int(pos); frac = pos - i
        return vals[i]*(1-frac) + vals[min(i+1, n-1)]*frac

    def _trim_bbox(arr, qlo=0.02, qhi=0.98):
        if not arr: return None
        lons = sorted([x for (x, _) in arr]); lats = sorted([y for (_, y) in arr])
        if len(lons) < 3: return _bbox_from_pairs(arr)
        return (_quantile(lons, qlo), _quantile(lons, qhi),
                _quantile(lats, qlo), _quantile(lats, qhi))

    def _bbox_union(a, b):
        if not a: return b
        if not b: return a
        return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))

    def _expand_bbox(bb, factor=1.04, min_lon_span=2.0, min_lat_span=1.2, pad_deg=0.3):
        lon_min, lon_max, lat_min, lat_max = bb
        lon_span = (lon_max - lon_min) * factor; lat_span = (lat_max - lat_min) * factor
        lon_span = max(lon_span, min_lon_span); lat_span = max(lat_span, min_lat_span)
        lon_c = (lon_min + lon_max)/2.0; lat_c = (lat_min + lat_max)/2.0
        lon_min = max(-180.0, lon_c - lon_span/2.0 - pad_deg)
        lon_max = min( 180.0, lon_c + lon_span/2.0 + pad_deg)
        lat_min = max( -90.0, lat_c - lat_span/2.0 - pad_deg)
        lat_max = min(  90.0, lat_c + lat_span/2.0 + pad_deg)
        return (lon_min, lon_max, lat_min, lat_max)

    def _projector(W, H, bb):
        lon_min, lon_max, lat_min, lat_max = bb
        s = min(W/(lon_max-lon_min+1e-9), H/(lat_max-lat_min+1e-9))
        offx = (W - s*(lon_max-lon_min)) / 2.0
        offy = (H - s*(lat_max-lat_min)) / 2.0
        def to_px(lon, lat):
            return int(round(offx + (lon-lon_min)*s)), int(round(offy + (lat_max-lat)*s))
        return to_px

    # ---------- detect multi vs single ----------
    multi = bool(points_or_groups and isinstance(points_or_groups[0], dict) and "points" in points_or_groups[0])
    groups = points_or_groups if multi else [{"label": "Occurrences", "points": points_or_groups}]

    # ---------- country outline (Natural Earth → fallback to Nominatim), normalized ----------
    rings_data = _country_outline_rings(country)       # returns rings or (rings, provider)
    rings = rings_data[0] if isinstance(rings_data, tuple) else rings_data
    rings = [[_norm_pair(lon, lat) for (lon, lat) in ring] for ring in (rings or [])]
    
    # bbox from the outline ONLY (don’t let outlier points skew the view)
    ring_bbox = _bbox_from_rings_and_points(rings, []) or (-80.0, -60.0, -25.0, 5.0)  # harmless fallback
    
    # final viewport: small expansion + fill canvas aspect, centered on outline
    bbox = expand_bbox(ring_bbox, factor=1.02, min_lon_span=2.0, min_lat_span=1.2, pad_deg=0.35)
    W, H = size
    bbox = _fit_bbox_to_aspect(bbox, W, H)
    
    # optional: clip plotted points to outline bbox ± clamp (keeps legend counts sane)
    lon_min, lon_max, lat_min, lat_max = ring_bbox
    clamp = (max(-180.0, lon_min - clamp_deg), min(180.0, lon_max + clamp_deg),
             max( -90.0, lat_min - clamp_deg), min( 90.0, lat_max + clamp_deg))
    def _inside(p):
        return (clamp[0] <= p[0] <= clamp[1]) and (clamp[2] <= p[1] <= clamp[3])
    for g in groups:
        g["points"] = [p for p in g["points"] if (_as_lonlat(p) and _inside(_as_lonlat(p)))]
    
    # clamp points to ring bbox ± clamp_deg (keeps map centered on the country)
    if ring_bbox:
        lon_min, lon_max, lat_min, lat_max = ring_bbox
        clamp = (max(-180.0, lon_min - clamp_deg), min(180.0, lon_max + clamp_deg),
                 max( -90.0, lat_min - clamp_deg), min( 90.0, lat_max + clamp_deg))
        def _inside(p):
            return (clamp[0] <= p[0] <= clamp[1]) and (clamp[2] <= p[1] <= clamp[3])
        for g in groups:
            g["points"] = [p for p in g["points"] if (_as_lonlat(p) and _inside(_as_lonlat(p)))]

        # recompute bbox after clamping
        all_pts = _flatten_points(groups)
        if all_pts:
            bbox = _bbox_union(bbox, _trim_bbox(all_pts, 0.01, 0.99))

    # fill the canvas (match aspect)
    W, H = size
    target = W / float(H)
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_span, lat_span = lon_max - lon_min, lat_max - lat_min
    ratio = lon_span / max(1e-9, lat_span)
    if ratio > target:
        need_lat = lon_span / target; extra = (need_lat - lat_span)/2.0
        lat_min = max(-90.0, lat_min - extra); lat_max = min(90.0, lat_max + extra)
    elif ratio < target:
        need_lon = lat_span * target; extra = (need_lon - lon_span)/2.0
        lon_min = max(-180.0, lon_min - extra); lon_max = min(180.0, lon_max + extra)
    bbox = (lon_min, lon_max, lat_min, lat_max)


    # --- build the canvas (optionally with satellite basemap) ---
    W, H = size
    base = None
    if imagery:  # Esri/MapTiler tiles -> Web Mercator
        try:
            base = _satellite_mosaic_for_bbox(bbox, size, provider=imagery)
        except Exception:
            base = None
    
    # Use a projector that matches the basemap
    to_px = _projector_mercator(W, H, bbox) if imagery and base is not None else _projector(W, H, bbox)
    
    img = base if base is not None else Image.new("RGB", (W, H), (255, 255, 255))
    
    # --- draw the country outline on a transparent overlay (halo + stroke) ---
    if rings:
        stroke = max(1, W // 450)
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for ring in rings:
            step = max(1, len(ring) // 3500)
            pts = [to_px(lon, lat) for (lon, lat) in ring[::step]]
            if len(pts) >= 2:
                od.line(pts + [pts[0]], fill=(255, 255, 255, 180), width=stroke*2 + 2)  # halo
                od.line(pts + [pts[0]], fill=(40, 40, 40, 220),  width=stroke)          # stroke
        img = img.convert("RGBA")
        img.alpha_composite(overlay)
        img = img.convert("RGB")
    
    # --- (re)create draw context and fonts on the final image ---
    d = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", int(W * 0.045))
        font       = ImageFont.truetype("DejaVuSans.ttf", int(W * 0.030))
    except Exception:
        title_font = font = ImageFont.load_default()
    
    # --- plot points (and collect totals for the legend/title) ---
    palette = [(220,50,50), (40,110,210), (20,160,80), (180,90,20),
               (150,60,180), (20,160,160), (120,120,20), (120,70,70)]
    
    max_plot = int(os.getenv("MINDAT_MAX_PLOT", "3000"))
    totals = []
    for i, g in enumerate(groups):
        col = palette[i % len(palette)]
        pts = [_as_lonlat(p) for p in g["points"]]
        pts = [p for p in pts if p is not None]
        total = len(pts)
        totals.append((g.get("label","Series "+str(i+1)), total))
        if total > max_plot:
            step = max(1, total // max_plot)
            pts = pts[::step][:max_plot]
        r = max(point_radius, W // 280)
        for (lon, lat) in pts:
            x, y = to_px(lon, lat)
            if -4 <= x <= W+4 and -4 <= y <= H+4:
                d.ellipse([x-r, y-r, x+r, y+r], fill=col, outline=(255,255,255))
    
    # --- frame, title, legend, footer ---
    d.rectangle([6, 6, W - 6, H - 6], outline=(0, 0, 0), width=2)
    
    # compute ttl *after* totals exist
    ttl = title or f"{country} — " + ", ".join([f"{lbl} ({cnt})" for lbl, cnt in totals])
    d.text((14, 12), ttl, fill=(0, 0, 0), font=title_font)

    # palette & legend
    palette = [(220,50,50), (40,110,210), (20,160,80), (180,90,20),
               (150,60,180), (20,160,160), (120,120,20), (120,70,70)]

    max_plot = int(os.getenv("MINDAT_MAX_PLOT", "3000"))
    totals = []
    for i, g in enumerate(groups):
        col = palette[i % len(palette)]
        pts = [_as_lonlat(p) for p in g["points"]]
        pts = [p for p in pts if p is not None]
        total = len(pts)
        totals.append((g.get("label","Series "+str(i+1)), total))
        if total > max_plot:
            step = max(1, total // max_plot)
            pts = pts[::step][:max_plot]
        r = max(point_radius, W // 280)   # was: max(2, W // 360)
        for (lon, lat) in pts:
            x, y = to_px(lon, lat)
            if -4 <= x <= W+4 and -4 <= y <= H+4:
                d.ellipse([x-r, y-r, x+r, y+r], fill=col, outline=(255,255,255))

    # title & legend
    ttl = title or f"{country} — " + ", ".join([f"{lbl} ({cnt})" for lbl, cnt in totals])
    if font:
        d.text((14, 12), ttl, fill=(0, 0, 0), font=font)
        line_h = max(14, _measure(d, "Ag", font)[1] + 2)
        rows = [f"{lbl} ({cnt})" for lbl, cnt in totals]
        text_w = max((_measure(d, s, font)[0] for s in rows), default=0)
        pad, margin = 10, 12
        box_w = 28 + text_w + pad*2
        title_h = _measure(d, ttl, font)[1]
        x0 = W - margin - box_w
        y0 = margin + title_h + 6
        box_h = pad*2 + line_h*len(rows)
        d.rectangle([x0, y0, x0+box_w, y0+box_h], fill=(245,245,245), outline=(100,100,100), width=1)
        for j, row in enumerate(rows):
            cy = y0 + pad + j*line_h + line_h//2
            col = palette[j % len(palette)]
            rr = max(4, int(r * 0.9))
            d.ellipse([x0+pad+6-rr, cy-rr, x0+pad+6+rr, cy+rr], fill=col, outline=(255,255,255))
            d.text((x0+pad+18, cy - line_h//2 + 1), row, fill=(0,0,0), font=font)
    
    # save
    slug = "".join(ch if ch.isalnum() else "_" for ch in (country or "").lower()).strip("_") or "country"
    out = os.path.join(tempfile.gettempdir(), f"mindat_{slug}_{int(time.time())}.png")
    
    # --- attribution footer (draw LAST so it sits on top of imagery) ---
    outline_src = "Natural Earth" if _LAST_OUTLINE_SRC == "ne" else "OSM/Nominatim"
    basemap_note = {
        None: "None",
        "esri": "Esri WorldImagery (Esri, Maxar, Earthstar Geographics)",
        "maptiler": "MapTiler Satellite",
        "bing": "Bing Aerial",
        "sentinel": "Sentinel-2",
    }.get((imagery or "").lower() if isinstance(imagery, str) else imagery, "—")
    
    note = f"Basemap: {basemap_note}  •  Data: Mindat.org  •  Outline: {outline_src}"
    tw, th = _measure(d, note, font)
    _text_with_bg(d, (W - tw - 12, H - th - 12), note, font, pad=6,
                  fg=(255, 255, 255), bg=(0, 0, 0, 150))    
    
    img.save(out, "PNG")
    return out if os.path.isfile(out) and os.path.getsize(out) > 0 else None

def _fit_bbox_to_aspect(bb: tuple[float,float,float,float], W: int, H: int):
    """Center bbox and expand the shorter dimension so it exactly fills W×H."""
    lon_min, lon_max, lat_min, lat_max = bb
    lon_span = max(1e-9, lon_max - lon_min)
    lat_span = max(1e-9, lat_max - lat_min)
    lon_c = (lon_min + lon_max) / 2.0
    lat_c = (lat_min + lat_max) / 2.0
    target = W / float(H)
    ratio  = lon_span / lat_span

    if ratio > target:  # too wide → add latitude
        needed = lon_span / target
        lat_min = max(-90.0, lat_c - needed/2.0)
        lat_max = min( 90.0, lat_c + needed/2.0)
    else:               # too tall → add longitude
        needed = lat_span * target
        lon_min = max(-180.0, lon_c - needed/2.0)
        lon_max = min( 180.0, lon_c + needed/2.0)
    return (lon_min, lon_max, lat_min, lat_max)

def _normalize_lonlat(a: float, b: float) -> tuple[float, float]:
    """
    Heuristic: return (lon, lat). If pair looks like (lat, lon), swap.
    """
    a, b = float(a), float(b)
    # If first looks like a latitude and second like a longitude -> swap
    if -90.0 <= a <= 90.0 and (b < -90.0 or b > 90.0) and -180.0 <= b <= 180.0:
        return b, a
    # If first is clearly outside lon range but second is inside, also swap
    if not (-180.0 <= a <= 180.0) and (-180.0 <= b <= 180.0):
        return b, a
    return a, b


def _bbox_from_rings_and_points(rings, points):
    lons, lats = [], []
    for ring in rings or []:
        for lon, lat in ring: lons.append(lon); lats.append(lat)
    for p in points or []:
        if isinstance(p, dict):
            lon, lat = p.get("longitude"), p.get("latitude")
        else:
            lon, lat = (p[0], p[1]) if (isinstance(p,(list,tuple)) and len(p)>=2) else (None,None)
        if lon is not None and lat is not None:
            lons.append(float(lon)); lats.append(float(lat))
    if not lons or not lats: return None
    return (min(lons), max(lons), min(lats), max(lats))

def _area_rings_osm(query: str) -> list[list[tuple[float, float]]]:
    # generic OSM/Nominatim rings fetch for any named area
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "jsonv2", "polygon_geojson": 1, "polygon_threshold": 0.0008},
            headers={"User-Agent": USER_AGENT}, timeout=20,
        )
        r.raise_for_status()
        arr = r.json() or []
        g = arr and (arr[0].get("geojson") or arr[0].get("polygon_geojson"))
        if not g: return []
        rings = []
        def add_ring(seq):
            ring=[]
            for c in seq or []:
                if isinstance(c,(list,tuple)) and len(c)>=2:
                    rings.append((float(c[0]), float(c[1])))
            if len(ring)>=2: rings.append(ring)
        t = (g.get("type") or "").lower()
        if t == "polygon":
            return [[(float(x), float(y)) for x,y in ring] for ring in g.get("coordinates", [])]
        elif t == "multipolygon":
            out=[]
            for poly in g.get("coordinates", []):
                for ring in (poly or []):
                    out.append([(float(x), float(y)) for x,y in ring])
            return out
    except Exception:
        return []
    return []

def _country_outline_rings(country: str, claimed_view: bool=False) -> list[list[tuple[float,float]]]:
    global _LAST_OUTLINE_SRC
    rings = _natural_earth_country_outline_rings(country)
    _LAST_OUTLINE_SRC = "ne" if rings else "osm"
    if not rings:
        rings = _nominatim_country_outline_rings(country)

    # Optional claimed-boundary augmentation (example: Ukraine + Crimea)
    if claimed_view and country.lower() == "ukraine":
        crim = _area_rings_osm("Crimea")
        if crim:
            rings = rings + crim
    return rings

# ---- Natural Earth Admin-0 (country outlines) -------------------------------
NE_URLS = [
    # higher detail first (≈1:50m). Falls back to 110m if needed.
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson",
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
]
_NE_CACHE = {}  # in-memory
_NE_PATH = os.path.join(tempfile.gettempdir(), "ne_admin0_countries.json")

def _load_ne_geojson() -> Optional[dict]:
    # memory cache
    if "_data" in _NE_CACHE:
        return _NE_CACHE["_data"]
    # disk cache
    if os.path.isfile(_NE_PATH) and os.path.getsize(_NE_PATH) > 0:
        try:
            with open(_NE_PATH, "r", encoding="utf-8") as f:
                js = json.load(f)
                _NE_CACHE["_data"] = js
                return js
        except Exception:
            pass
    # fetch
    for url in NE_URLS:
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
            r.raise_for_status()
            js = r.json()
            with open(_NE_PATH, "w", encoding="utf-8") as f:
                json.dump(js, f)
            _NE_CACHE["_data"] = js
            return js
        except Exception:
            continue
    return None

def _ne_match_country_name(props: dict, wanted: str) -> bool:
    wanted = (wanted or "").strip().lower()
    if not wanted:
        return False
    keys = [
        "ADMIN", "SOVEREIGNT", "NAME", "NAME_LONG", "ABBREV", "BRK_NAME",
        "ADMIN_NAME", "FORMAL_EN"
    ]
    vals = []
    for k in keys:
        v = props.get(k) or props.get(k.lower())
        if isinstance(v, str):
            vals.append(v.lower())
    # a few light aliases
    aliases = {
        "drc": "democratic republic of the congo",
        "congo": "republic of the congo",
        "u.k.": "united kingdom", "uk": "united kingdom",
        "u.s.": "united states", "usa": "united states", "us": "united states",
        "bolivia": "bolivia",
    }
    wanted_norm = aliases.get(wanted, wanted)
    return any(wanted_norm == v for v in vals)

_LAST_OUTLINE_SRC = "unknown"

def _country_outline_rings(country: str) -> list[list[tuple[float, float]]]:
    """Prefer Natural Earth; fall back to Nominatim. Returns rings only."""
    rings = _natural_earth_country_outline_rings(country)
    return rings if rings else _nominatim_country_outline_rings(country)

def _natural_earth_country_outline_rings(country: str) -> list[list[tuple[float, float]]]:
    """
    Return rings as [(lon,lat), ...] from Natural Earth Admin-0.
    """
    js = _load_ne_geojson()
    if not js:
        return []
    feats = js.get("features") or []
    for ft in feats:
        props = ft.get("properties") or {}
        if not _ne_match_country_name(props, country):
            continue
        geom = ft.get("geometry") or {}
        gtype = (geom.get("type") or "").lower()
        coords = geom.get("coordinates") or []
        rings: list[list[tuple[float, float]]] = []

        def add_ring(seq):
            ring = []
            for c in seq or []:
                if isinstance(c, (list, tuple)) and len(c) >= 2:
                    # GeoJSON is lon,lat already
                    ring.append((float(c[0]), float(c[1])))
            if len(ring) >= 2:
                rings.append(ring)

        if gtype == "polygon":
            for ring in coords:           # first = outer, rest = holes (we just outline)
                add_ring(ring)
        elif gtype == "multipolygon":
            for poly in coords:
                for ring in (poly or []):
                    add_ring(ring)
        return rings
    return []

# --- Satellite tile proc ---------------------------------------------------
import math, tempfile

_TILE_SIZE = 256

def _lonlat_to_tile_xy(lon, lat, z):
    lat = max(min(lat, 85.05112878), -85.05112878)  # Web Mercator clamp
    n = 2.0 ** z
    xt = (lon + 180.0) / 360.0 * n
    yt = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return xt, yt  # fractional tile coords

def _lonlat_to_pixel_xy(lon, lat, z):
    xt, yt = _lonlat_to_tile_xy(lon, lat, z)
    return xt * _TILE_SIZE, yt * _TILE_SIZE

def _pick_zoom_for_bbox(bbox, target_px_width, min_z=3, max_z=11):
    lon_min, lon_max, lat_min, lat_max = bbox
    # choose the smallest z that gives us ≳ target width in pixels (avoid fetching too many tiles)
    for z in range(min_z, max_z + 1):
        x0, _ = _lonlat_to_pixel_xy(lon_min, lat_max, z)
        x1, _ = _lonlat_to_pixel_xy(lon_max, lat_min, z)
        if (x1 - x0) >= (target_px_width * 1.2):
            return z
    return max_z

def _tile_url(provider, z, x, y):
    # Default: Esri World Imagery (JPG); alternatives supported if you have a key.
    if provider == "esri":
        return f"https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    elif provider == "maptiler":
        key = os.getenv("MAPTILER_KEY", "")
        return f"https://api.maptiler.com/maps/satellite/256/{z}/{x}/{y}.jpg?key={key}"
    else:
        # Fallback (not satellite): OSM standard tiles
        return f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"

def _fetch_tile(provider, z, x, y):
    # simple disk cache
    cache_dir = os.path.join(tempfile.gettempdir(), "tile_cache", provider, str(z), str(x))
    os.makedirs(cache_dir, exist_ok=True)
    ext = ".jpg" if provider in ("esri", "maptiler") else ".png"
    fpath = os.path.join(cache_dir, f"{y}{ext}")
    if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
        try:
            return Image.open(fpath).convert("RGB")
        except Exception:
            pass
    try:
        url = _tile_url(provider, z, x, y)
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        r.raise_for_status()
        with open(fpath, "wb") as f:
            f.write(r.content)
        return Image.open(fpath).convert("RGB")
    except Exception:
        # blank tile fallback
        return Image.new("RGB", (_TILE_SIZE, _TILE_SIZE), (235, 235, 235))

def _satellite_mosaic_for_bbox(bbox, size, provider="esri"):
    """
    Return a PIL.Image of satellite tiles covering bbox, resized to `size` (W,H).
    """
    from PIL import Image
    W, H = size
    z = _pick_zoom_for_bbox(bbox, W)
    lon_min, lon_max, lat_min, lat_max = bbox

    xpix_min, ypix_min = _lonlat_to_pixel_xy(lon_min, lat_max, z)  # top-left
    xpix_max, ypix_max = _lonlat_to_pixel_xy(lon_max, lat_min, z)  # bottom-right

    x_tile_min = int(math.floor(xpix_min / _TILE_SIZE))
    y_tile_min = int(math.floor(ypix_min / _TILE_SIZE))
    x_tile_max = int(math.floor(xpix_max / _TILE_SIZE))
    y_tile_max = int(math.floor(ypix_max / _TILE_SIZE))

    tiles_x = x_tile_max - x_tile_min + 1
    tiles_y = y_tile_max - y_tile_min + 1
    if tiles_x <= 0 or tiles_y <= 0:
        return None

    # mosaic
    mosaic = Image.new("RGB", (tiles_x * _TILE_SIZE, tiles_y * _TILE_SIZE), (220, 220, 220))
    for tx in range(x_tile_min, x_tile_max + 1):
        for ty in range(y_tile_min, y_tile_max + 1):
            tile = _fetch_tile(provider, z, tx, ty)
            mosaic.paste(tile, ((tx - x_tile_min) * _TILE_SIZE, (ty - y_tile_min) * _TILE_SIZE))

    # crop to exact bbox in pixel-space
    crop_left   = int(round(xpix_min - x_tile_min * _TILE_SIZE))
    crop_top    = int(round(ypix_min - y_tile_min * _TILE_SIZE))
    crop_right  = int(round(xpix_max - x_tile_min * _TILE_SIZE))
    crop_bottom = int(round(ypix_max - y_tile_min * _TILE_SIZE))
    crop_right  = max(crop_left + 1, crop_right)
    crop_bottom = max(crop_top + 1, crop_bottom)

    mosaic = mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))
    if mosaic.size != (W, H):
        mosaic = mosaic.resize((W, H), Image.BILINEAR)
    return mosaic

def _mindat_llm_extract_multi(query: str) -> dict:
    """
    Returns:
      {"country": "Peru",
       "targets": [{"kind":"commodity","name":"Copper","elements":["Cu"]},
                   {"kind":"commodity","name":"Gold","elements":["Au"]}]}
    """
    sys = (
        "From the user's text, extract exactly ONE present-day COUNTRY and "
        "a LIST of targets (commodities or mineral species). "
        "Rules:\n"
        "- Commodities: generic metals or symbols (copper, Cu, Pb-Zn, gold...).\n"
        "- Minerals: species names (chalcopyrite, bornite, native copper...).\n"
        "- Prefer ISO 3166 country names.\n"
        "Respond STRICT JSON ONLY with keys: country (string), "
        "targets (array of objects with keys kind ('commodity'|'mineral'), name, elements (array; empty if kind=mineral))."
    )
    shots = [
        {"role":"user","content":"Copper and Gold of Peru"},
        {"role":"assistant","content":'{"country":"Peru","targets":[{"kind":"commodity","name":"Copper","elements":["Cu"]},{"kind":"commodity","name":"Gold","elements":["Au"]}]}'},
        {"role":"user","content":"Pb-Zn and Silver in Morocco"},
        {"role":"assistant","content":'{"country":"Morocco","targets":[{"kind":"commodity","name":"Lead-Zinc","elements":["Pb","Zn"]},{"kind":"commodity","name":"Silver","elements":["Ag"]}]}'},
        {"role":"user","content":"Native Copper and Chalcopyrite, Chile"},
        {"role":"assistant","content":'{"country":"Chile","targets":[{"kind":"mineral","name":"Native Copper","elements":[]},{"kind":"mineral","name":"Chalcopyrite","elements":[]}]}'},
    ]
    try:
        resp = _client().chat.completions.create(
            model=HF_MODEL_ID, temperature=0.0, max_tokens=200,
            messages=[{"role":"system","content":sys}] + shots + [{"role":"user","content":query}]
        )
        txt = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", txt, flags=re.S)
        js = json.loads(m.group(0) if m else txt)
        # normalize
        country = (js.get("country") or "").strip()
        targets = []
        for t in (js.get("targets") or []):
            kind = (t.get("kind") or "").strip().lower()
            name = (t.get("name") or "").strip()
            elements = [e.strip() for e in (t.get("elements") or []) if isinstance(e, str) and e.strip()]
            if kind in ("commodity","mineral") and (name or elements):
                targets.append({"kind":kind, "name":name, "elements":elements})
        if country and targets:
            return {"country": country, "targets": targets}
    except Exception:
        pass

    # ---------- Heuristic fallback ----------
    text = (query or "").lower()
    country = _extract_country(query) or _mindat_detect_country(query) or ""

    COMMODITY = {
        "copper":["Cu"], "gold":["Au"], "silver":["Ag"], "lead":["Pb"], "zinc":["Zn"],
        "nickel":["Ni"], "cobalt":["Co"], "tin":["Sn"], "tungsten":["W"], "molybdenum":["Mo"],
        "uranium":["U"], "lithium":["Li"], "iron":["Fe"], "manganese":["Mn"],
    }
    # split like: "copper, gold and silver" or "Pb-Zn, Cu"
    words = re.split(r"[,\;/]| and |\s+\&\s+", text)
    found = []
    # symbols like Pb-Zn, Cu, Au
    for tok in re.findall(r"\b(?:[A-Z][a-z]?)?(?:-[A-Z][a-z]?)+\b|\b[A-Z][a-z]?\b", query):
        els = [e for e in re.split(r"-", tok) if re.fullmatch(r"[A-Z][a-z]?", e)]
        if els:
            nm = "-".join(els)
            found.append({"kind":"commodity","name":nm,"elements":els})

    for w in words:
        w = w.strip()
        if w in COMMODITY:
            found.append({"kind":"commodity","name":w.title(),"elements":COMMODITY[w]})
        elif re.fullmatch(r"(native copper|chalcopyrite|bornite|chalcocite|tennantite|covellite|enargite|malachite|azurite)", w):
            found.append({"kind":"mineral","name":w.title(),"elements":[]})

    # dedupe by label
    uniq = []
    seen = set()
    for t in found:
        key = (t["kind"], t["name"] or ",".join(t["elements"]))
        if key not in seen:
            seen.add(key); uniq.append(t)

    return {"country": country, "targets": uniq}

def _mindat_points_for_target(country: str, kind: str, name: str, elements: list[str], token: str):
    pts, label = _mindat_points(country, kind, name, elements, token)  # reuse your existing single-target fetcher
    # normalize to dicts for the drawer
    norm = []
    for p in pts:
        if isinstance(p, dict):
            lon, lat = p.get("longitude") or p.get("lon"), p.get("latitude") or p.get("lat")
            nm = p.get("name","")
        else:
            lon, lat = (p[0], p[1]) if len(p) >= 2 else (None, None)
            nm = p[2] if len(p) >= 3 else ""
        if lon is None or lat is None: 
            continue
        norm.append({"longitude": float(lon), "latitude": float(lat), "name": nm})
    return {"label": label or name or "target", "points": norm, "count": len(norm)}

def _mindat_points_multi(country: str, targets: list[dict], token: str):
    groups = []
    for t in targets:
        groups.append(_mindat_points_for_target(country, t.get("kind",""), t.get("name",""), t.get("elements") or [], token))
    # drop empty groups
    return [g for g in groups if g["points"]]

def _projector_mercator(W: int, H: int, bbox: tuple[float, float, float, float]):
    """
    lon/lat -> (x,y) in pixels on a canvas of size W×H using Web Mercator,
    matched to the same bbox used for the tile mosaic.
    """
    import math

    def merc_xy(lon: float, lat: float):
        # clamp to Web Mercator limit
        lat = max(min(lat, 85.05112878), -85.05112878)
        x = (lon + 180.0) / 360.0
        y = (1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0
        return x, y  # normalized [0,1]

    lon_min, lon_max, lat_min, lat_max = bbox
    x0, y0 = merc_xy(lon_min, lat_max)  # top-left
    x1, y1 = merc_xy(lon_max, lat_min)  # bottom-right

    sx = W / max(1e-12, (x1 - x0))
    sy = H / max(1e-12, (y1 - y0))

    def to_px(lon: float, lat: float) -> tuple[int, int]:
        x, y = merc_xy(lon, lat)
        return int(round((x - x0) * sx)), int(round((y - y0) * sy))

    return to_px
        
### KNOWLEDGE-GRAPH GEOKG EXTRACTION PROC #####################################################################
def _kg_pdf_dir_candidates() -> list[str]:
    p = _downloads_dir()
    print("[KG] Scanning:", p)  # tiny diagnostic
    return [p]

def _kg_list_pdfs(max_pdfs: int = KG_MAX_PDFS) -> list[str]:
    out = []
    for base in _kg_pdf_dir_candidates():
        for name in os.listdir(base):
            if name.lower().endswith(".pdf"):
                out.append(os.path.join(base, name))
    out.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    return out[:max(1, min(len(out), max_pdfs))]

def _pdf_extract_text_all(pdf_path: str, max_pages: int = KG_MAX_PAGES) -> str:
    """Read up to max_pages pages. pdfminer → PyPDF2 fallback."""
    try:
        from pdfminer.high_level import extract_text  # type: ignore
        return extract_text(pdf_path, maxpages=max_pages) or ""
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(open(pdf_path, "rb"))
        pages = reader.pages[:max_pages]
        txt = []
        for p in pages:
            try:
                txt.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(txt)
    except Exception:
        return ""

def _kg_split_sentences(text: str, cap: int = KG_SENTENCES_PER_PDF) -> list[str]:
    """Lightweight sentence splitter; trims ultra-short/ultra-long."""
    raw = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9“"(])', text.strip().replace("\r", " "))
    out = []
    for s in raw:
        s = re.sub(r"\s+", " ", s).strip()
        if 25 <= len(s) <= 600:
            out.append(s)
        if len(out) >= cap:
            break
    return out

_KG_CATEGORIES = [
    "Chronostratigraphy","Lithology",
    "Tectonic","Ore Deposit Feature","Mineral",
    "Geographical Location"
]
_KG_REL_TYPES = [
    "associated_with","occurs_in","located_in","deposited_in","overlies","underlies",
    "equivalent_to","part_of","controls","sourced_from","sealed_by","contains","produces","intersects"
]

def _kg_norm_name(name: str) -> str:
    """Canonical key (casefold + compact spaces + strip punctuation at ends)."""
    n = (name or "").strip().strip(".,;:()[]{}").lower()
    n = re.sub(r"\s+", " ", n)
    return n

def _kg_llm_extract_batch(sentences: list[str], client: InferenceClient) -> list[dict]:
    if not sentences:
        return []
    payload = [{"i": i, "text": (s or "")[:600]} for i, s in enumerate(sentences)]

    sys = (
      "You are an information extraction model for geoscience. "
      "For EACH input sentence, return STRICT JSON ONLY.\n"
      'Schema: {"sentences":[{"i":<int>,"text":"<verbatim>",'
      '"entities":[{"name":"<exact substring>","category":"<one>"}],"relations":[]}]}.\n'
      "Allowed categories (exact spelling ONLY): "
      "Chronostratigraphy, Lithology, "
      "Tectonic, Minerals, Geographical Location.\n"
      "NEVER invent other categories. If something does not fit, either choose the closest allowed one or SKIP it.\n"
      "Rules:\n"
      "- Use the EXACT substring from the sentence as 'name'.\n"
      "- Drop standalone numbers, percentages, measurements, depths, VRo values, codes, units (e.g. '15', '3%', '150 m', '0.8%VRo').\n"
      "- Drop titles of documents/databases/software/organizations (e.g., 'Atlas', 'Journal', 'Report', 'Database', 'Software').\n"
      "- Geographical Location: countries/regions/areas (e.g., 'North Queensland', 'Libya', 'North Sea').\n"
      "- Tectonics: fault, anticline, island arc, subduction, syncline, basin, horst, graben, trough, ridge.\n"
      "- Lithology: shale, sandstone, limestone, dolomite, siltstone, conglomerate, chalk, evaporite, etc.\n"
      "- Chronostratigraphy MUST be extracted if present (named ages or ranges like '201–145 Ma')."
    )

    # few-shot to anchor ages + lithostrat + env + locations
    shots = [
        {
            "role":"user",
            "content":json.dumps([{"i":0,"text":"Deposition occurred between 201–145 Ma in the North Sea."}], ensure_ascii=False)
        },
        {
            "role":"assistant",
            "content":'{"sentences":[{"i":0,"text":"Deposition occurred between 201–145 Ma in the North Sea.","entities":[{"name":"201–145 Ma","category":"Chronostratigraphy"},{"name":"North Sea","category":"Geographical Location"}],"relations":[]}]}'
        },
        {
            "role":"user",
            "content":json.dumps([{"i":0,"text":"Higher copper grades are associated with garnet skarn, in addition to quartz vein stockworks."}], ensure_ascii=False)
        },
        {
            "role":"assistant",
            "content":'{"sentences":[{"i":0,"text":"Higher copper grades are associated with garnet skarn, in addition to quartz vein stockworks.","entities":[{"name":"copper","category":"Mineral"},{"name":"garnet skarn","category":"Lithology"}, {"name":"quartz vein stockworks","category":"Ore Deposit Feature"}],"relations":[]}]}'
        },

        {
            "role":"user",
            "content":json.dumps([{"i":0,"text":"Organic-rich shale of the Late Jurassic Kimmeridge Clay Formation was deposited on a marine shelf."}], ensure_ascii=False)
        },
        {
            "role":"assistant",
            "content":'{"sentences":[{"i":0,"text":"Organic-rich shale of the Late Jurassic Kimmeridge Clay Formation was deposited in the Viking Graben.","entities":[{"name":"shale","category":"Lithology"},{"name":"Late Jurassic","category":"Chronostratigraphy"},{"name":"Viking Graben","category":"Tectonic"}],"relations":[]}]}'
        },
        {
            "role":"user",
            "content":json.dumps([{"i":0,"text":"Cu-Au deposits can be interpreted as transtensional features, formed during northwest compression and north-striking left lateral faulting."}], ensure_ascii=False)
        },
        {
            "role":"assistant",
            "content":'{"sentences":[{"i":0,"text":"Cu-Au deposits can be interpreted as transtensional features, formed during northwest compression and north-striking left lateral faulting.","entities":[{"name":"Cu-Au deposits","category":"Mineral"},{"name":"transtensional","category":"Tectonic"},{"name":"lateral faulting","category":"Tectonic"}],"relations":[]}]}'
        },
    ]

    # --- call LLM ---
    raw = ""
    arr = None
    try:
        resp = client.chat.completions.create(
            model=HF_MODEL_ID,
            messages=[{"role":"system","content":sys}] + shots + [{"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=min(1200, 240 + 160*len(payload)),
        )
        raw = (resp.choices[0].message.content or "").strip()
        js = _safe_json_extract(raw) or {}
        arr = js.get("sentences") if isinstance(js, dict) else None
    except Exception as e:
        if KG_DEBUG:
            print(f"[KG][LLM] Exception: {type(e).__name__}: {e}", flush=True)

    # --- parse into dict by i, with debug ---
    results_by_i: dict[int, dict] = {}
    if not isinstance(arr, list):
        if KG_DEBUG:
            print(f"[KG][LLM] Malformed/bad JSON. Raw (first 400 chars): {raw[:400]!r}", flush=True)
    else:
        if KG_DEBUG:
            print(f"[KG][LLM] Batch inputs:{len(payload)} · JSON items:{len(arr)}", flush=True)
        for item in arr:
            if not isinstance(item, dict): 
                continue
            i = item.get("i")
            txt = item.get("text","")
            ents_raw = item.get("entities") or []
            ents = []
            for e in ents_raw:               
                nm = str(e.get("name","")).strip()
                cat = str(e.get("category","")).strip()
                if not nm:
                    continue
                cat = _kg_clean_category(nm, cat)
                if not cat:
                    continue  # drop out-of-scope things like "Depth", "VRo", "Software"
                ents.append({"name": nm, "category": cat})                
                
                
            if isinstance(i, int):
                results_by_i[i] = {"i": i, "text": txt, "entities": ents, "relations": []}

        # per-sentence debug (first few only)
        if KG_DEBUG:
            for i in range(min(KG_DEBUG_SENTS, len(payload))):
                it = results_by_i.get(i)
                snip = (payload[i]["text"][:160] + ("..." if len(payload[i]["text"]) > 160 else ""))
                if it:
                    names = [e["name"] for e in it.get("entities", [])][:6]
                    print(f"[KG][LLM] i={i} READ: {snip!r}  → entities:{len(names)} {names}", flush=True)
                else:
                    print(f"[KG][LLM] i={i} READ: {snip!r}  → entities:0 (missing item)", flush=True)

    # Fill output in input order; if an index is missing, return empty item (LLM-only, no rules)
    out: list[dict] = []
    for i, s in enumerate(sentences):
        item = results_by_i.get(i)
        if not item:
            item = {"i": i, "text": (s or "")[:600], "entities": [], "relations": []}
        out.append(item)
    return out

# --- Save Knowledge Graph to RDF ---------------------------------------------
def _slug_for_uri(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w]+", "_", s).strip("_")
    return (s or "node")[:120]

def _save_rdf_simple(nodes_rows: list[list], edges_rows: list[list], out_path: str) -> None:
    # nodes_rows: [name, type, count]
    # edges_rows: [src, tgt, weight]
    lines = []
    lines.append('@prefix ga:  <urn:geoassist:> .')
    lines.append('@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .')
    lines.append('@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .')
    lines.append('')

    # nodes
    for name, typ, cnt in nodes_rows:
        u = "urn:node:" + _slug_for_uri(str(name))
        t = "urn:cat:"  + _slug_for_uri(str(typ or "Unknown"))
        label = str(name).replace('\\', '\\\\').replace('"', '\\"')
        lines.append(f'<{u}> a <{t}> ; rdfs:label "{label}" ; ga:count "{int(cnt)}"^^xsd:integer .')

    # edges
    for src, tgt, w in edges_rows:
        ua = "urn:node:" + _slug_for_uri(str(src))
        ub = "urn:node:" + _slug_for_uri(str(tgt))
        lines.append(f'<{ua}> ga:linkedTo <{ub}> ; ga:weight "{int(w)}"^^xsd:integer .')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
           
def _kg_build_graph_from_docs(texts: list[tuple[str, str]], client: InferenceClient) -> tuple[dict, dict]:
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str], dict] = {}
    # Wire up the UI counters
    NCOUNTER = globals().setdefault("_KG_NODES_COUNTER", {})
    ECOUNTER = globals().setdefault("_KG_EDGES_COUNTER", {})
    for fname, t in texts:
        sents = _kg_split_sentences(t, cap=KG_SENTENCES_PER_PDF)
        if not sents:
            print(f"[KG] {fname}: 0 sentences → skipped", flush=True)
            continue

        n0_nodes, n0_edges = len(nodes), len(edges)
        file_mentions, file_pairs = 0, 0

        for i in range(0, len(sents), KG_BATCH_SIZE):
            batch = sents[i:i+KG_BATCH_SIZE]
            out = _kg_llm_extract_batch(batch, client)  # we only use 'entities' & 'text'
            if KG_DEBUG:
                print(f"[KG] Batch {i//KG_BATCH_SIZE+1}: {len(batch)} sentences", flush=True)

            for item in out:
                ents = item.get("entities") or []
                sent_text = item.get("text") or ""
                # nodes
                for e in ents:                    
                    nm = (e.get("name") or "").strip()
                    cat_in = (e.get("category") or "").strip()
                    cat = _kg_clean_category(nm, cat_in)
                    if not nm or not cat:
                        continue                    
                    k = _kg_norm_name(nm)
                    if not k:
                        continue
                    nd = nodes.get(k)
                    if nd is None:
                        nd = nodes[k] = {"name": nm, "category": cat, "mentions": 0, "degree": 0}
                    else:
                        if len(nm) > len(nd["name"]):
                            nd["name"] = nm
                        if (nd.get("category") in (None,"","Unknown")) and cat:
                            nd["category"] = cat
                    nd["mentions"] += 1
                    # update UI node counter
                    ck = _kg_norm_name(nm)
                    cm = NCOUNTER.get(ck, {"type": cat, "count": 0})
                    if not cm.get("type"):
                        cm["type"] = cat
                    cm["count"] += 1
                    NCOUNTER[ck] = cm
                    file_mentions += 1
                # edges by co-occurrence
                pairs = _kg_pairs_from_text_entities(sent_text, ents)
                for (a,b) in pairs:
                    ed = edges.get((a,b))
                    if ed is None:
                        ed = edges[(a,b)] = {"a": a, "b": b, "count": 0}
                    ed["count"] += 1
                    # update UI edge counter
                    pair = (a, b) if a < b else (b, a)
                    ECOUNTER[pair] = ECOUNTER.get(pair, 0) + 1
                file_pairs += len(pairs)

        # degrees
        for k in nodes: nodes[k]["degree"] = 0
        for (a,b), _ed in edges.items():
            if a in nodes: nodes[a]["degree"] += 1
            if b in nodes: nodes[b]["degree"] += 1

        print(f"[KG] ✓ {fname} — sentences:{len(sents)} · entity_mentions:{file_mentions} · pairs_added:{file_pairs} "
              f"· unique_nodes:+{len(nodes)-n0_nodes} (total {len(nodes)}) · unique_edges:+{len(edges)-n0_edges} (total {len(edges)})",
              flush=True)
    return nodes, edges

def _kg_pairs_from_text_entities(text: str, entities: list[dict]) -> list[tuple[str, str]]:
    if not text or not entities:
        return []
    clauses = re.split(r"\s*[;:–—]\s*", text)
    ent_names = [(e.get("name") or "").strip() for e in entities if (e.get("name") or "").strip()]
    ent_norms = {nm: _kg_norm_name(nm) for nm in ent_names}
    pairs: set[tuple[str, str]] = set()
    for c in (cl for cl in clauses if cl.strip()):
        clc = c.lower()
        present = sorted({ent_norms[nm] for nm in ent_names if nm and nm.lower() in clc and ent_norms.get(nm)})
        for i in range(len(present)):
            for j in range(i+1, len(present)):
                a, b = present[i], present[j]
                pairs.add((a,b) if a<b else (b,a))
    if not pairs and len(ent_norms) >= 2:
        present = sorted({k for k in ent_norms.values() if k})
        for i in range(len(present)):
            for j in range(i+1, len(present)):
                a, b = present[i], present[j]
                pairs.add((a,b) if a<b else (b,a))
    return list(pairs)

_KG_ALLOWED = {
    "Chronostratigraphy","Lithology",
    "Tectonic","Ore Deposit Feature","Mineral",
    "Geographical Location",
}

# lightweight aliasing → allowed set
_KG_CAT_ALIASES = {
    "location": "Geographical Location",
    "geographical": "Geographical Location",
    "geography": "Geographical Location",
    "minerals": "Mineral",
    "tectonics": "Tectonic",
    "tectonic": "Tectonic",
}

# (optional) skip obvious non-entities
_KG_STOPWORDS = {"depth","vro","v.r.o","software"}

_LITHO_TERMS = {"shale","sandstone","siltstone","limestone","dolomite","marl","mudstone",
                "claystone","carbonate","evaporite","anhydrite","halite","coal","conglomerate"}
_ENV_TERMS   = {"deltaic","fluvial","tidal","lacustrine","reef","shelf","slope","basinal",
                "turbidite","aeolian","alluvial","sabkha","marine"}
_TECT_TERMS  = {"fault","rift","uplift","inversion","thrust","fold","horst","graben","shear","tectonic"}
_GBODY_TERMS = {"basin","field","reservoir","prospect","well","block","licence","license",
                "platform","trough","ridge","high","low","arch","swell","saddle","hinge"}

_CHRONO_RE   = re.compile(
    r"(?i)\b(?:early|middle|late|lower|upper)?\s*"
    r"(quaternary|neogene|paleogene|cretaceous|jurassic|triassic|permian|carboniferous|devonian|silurian|ordovician|cambrian|miocene|pliocene|pliestocene|holocene|oligocene|eocene|paleocene)\b"
    r"|(\b\d{1,3}\s*(?:–|-|to)\s*\d{1,3}\s*ma\b|\b\d{2,3}\s*ma\b)"
)

def _kg_guess_category_from_name(nm: str) -> Optional[str]:
    s = (nm or "").lower()
    if nm.lower() in _KG_STOPWORDS: 
        return None
    if _CHRONO_RE.search(s):
        return "Chronostratigraphy"
    if any(t in s for t in _LITHO_TERMS):
        return "Lithology"
    if any(t in s for t in _TECT_TERMS):
        return "Tectonic"
    if re.search(r"(?i)\b(sea|gulf|ocean|mediterranean|north sea|red sea|sahara|africa|libya|egypt|algeria)\b", s):
        return "Geographical Location"
    return None

def _kg_clean_category(nm: str, cat: str | None) -> Optional[str]:
    c = (cat or "").strip()
    if c:
        c = _KG_CAT_ALIASES.get(c.lower(), c)
        if c in _KG_ALLOWED:
            return c
    # try to infer; if unsure, drop
    return _kg_guess_category_from_name(nm)

def _kg_run_pipeline() -> tuple[Optional[str], list[list], list[list]]:
    """
    Returns (image_path, nodes_rows, edges_rows)
    nodes_rows: [entity, category, mentions]
    edges_rows: [source, target, weight]
    """
    pdfs = _kg_list_pdfs(KG_MAX_PDFS)
    print(f"[KG] PDFs to scan: {len(pdfs)}", flush=True)
    if not pdfs:
        return None, [], []

    # Read text from PDFs
    texts: list[tuple[str, str]] = []
    for p in pdfs:
        try:
            t = _pdf_extract_text_all(p, max_pages=KG_MAX_PAGES)
            if t and len(t) > 200:
                texts.append((os.path.basename(p), t))
        except Exception:
            continue
    if not texts:
        return None, [], []

    # Build graph (nodes/edges) from docs
    client = _client()
    nodes, edges = _kg_build_graph_from_docs(texts, client)
    if not nodes:
        return None, [], []

    # ---- Tables (what the UI shows) ----
    nodes_rows = sorted(
        [[v["name"], v["category"], int(v["mentions"])] for v in nodes.values()],
        key=lambda r: (-r[2], r[0].lower())
    )

    edges_rows: list[list] = []
    for (a, b), ed in edges.items():
        sa = nodes.get(a, {"name": a}).get("name", a)
        sb = nodes.get(b, {"name": b}).get("name", b)
        w = int(ed.get("count", 1))
        if sa and sb and sa != sb and w > 0:
            edges_rows.append([sa, sb, w])
    edges_rows.sort(key=lambda r: (-r[2], r[0].lower(), r[1].lower()))

    # ---- Always draw with NetworkX proc (thicker/darker edges) ----
    img = _save_kg_network_png(nodes_rows, edges_rows)

    return img, nodes_rows, edges_rows

def _save_kg_network_png(nodes_rows: list[list], edges_rows: list[list]) -> Optional[str]:
    """
    nodes_rows: [[node, type, count], ...]
    edges_rows: [[src, tgt, weight], ...]
    Returns filepath of saved PNG (or None).
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except Exception:
        return None

    G = nx.Graph()
    # add nodes
    for name, typ, cnt in nodes_rows:
        try:
            G.add_node(str(name), typ=str(typ or ""), count=int(cnt))
        except Exception:
            continue

    # add edges
    for src, tgt, w in edges_rows:
        if not src or not tgt or src == tgt:
            continue
        try:
            G.add_edge(str(src), str(tgt), weight=int(w))
        except Exception:
            continue

    if G.number_of_nodes() == 0:
        # draw a small placeholder image so Gradio shows *something*
        fig = plt.figure(figsize=(4, 2.4), dpi=160)
        plt.axis("off")
        plt.text(0.5, 0.5, "No nodes/edges", ha="center", va="center")
        out = os.path.join(tempfile.gettempdir(), f"kg_empty_{int(time.time())}.png")
        plt.savefig(out, bbox_inches="tight", dpi=160)
        plt.close(fig)
        return out

    # layout
    try:
        pos = nx.spring_layout(G, k=0.22, seed=42, weight="weight")
    except Exception:
        pos = nx.random_layout(G, seed=42)

    # sizes & widths
    sizes = [300 + 35 * max(1, int(G.nodes[n].get("count", 1))) for n in G.nodes()]
    weights = [max(0.8, (edata.get("weight", 1)) ** 0.7) for _, _, edata in G.edges(data=True)]

    # color by type (just a few buckets for legibility)
    type_palette = {
        "Chronostratigraphy": "#1f77b4",
        "Lithology":          "#2ca02c",
        "Tectonic":  "#9467bd",
        "Ore Deposit Feature":         "#e377c2",
        "Mineral":           "#7f7f7f",
        "Geographical Location": "#bcbd22",   
    }
    node_colors = []
    for n in G.nodes():
        typ = (G.nodes[n].get("typ") or G.nodes[n].get("type") or "").strip()
        node_colors.append(type_palette.get(typ, "#4c78a8"))
    
    # draw
    fig = plt.figure(figsize=(10.5, 8.0), dpi=150)
    ax = plt.gca()
    ax.axis("off")
    #nx.draw_networkx_edges(G, pos, width=weights, alpha=0.35, ax=ax)
    nx.draw_networkx_edges(
    G, pos,
    width=[max(1.5, w) for w in weights],   # thicker
    alpha=0.85,                              # less transparent
    edge_color="#444444",                    # darker
    ax=ax
    )
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, linewidths=0.5, edgecolors="#222", ax=ax)
    
    # label ALL nodes
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=KG_LABEL_FONT_SIZE, font_color="#111", ax=ax
    )
    
    ax.set_title("Automated Geoscience Knowledge Graph from PDFs", fontsize=12)
    
    # legend / key for categories present
    cats_present = sorted({
        (G.nodes[n].get("typ") or G.nodes[n].get("type") or "").strip()
        for n in G.nodes()
        if (G.nodes[n].get("typ") or G.nodes[n].get("type"))
    })
    handles = [
        plt.Line2D([0],[0], marker='o', linestyle='',
                   markerfacecolor=type_palette.get(c, "#4c78a8"),
                   markeredgecolor="#222", markersize=8, label=c)
        for c in cats_present
    ]
    if handles:
        ax.legend(handles=handles, title="Entity type", loc="upper right", frameon=False, fontsize=8, title_fontsize=9)
    
    out = os.path.join(tempfile.gettempdir(), f"kg_net_{int(time.time())}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return out if os.path.isfile(out) and os.path.getsize(out) > 0 else None

### OVERALL ORCHESTRATING PROCEDURE #####################################################################################

def run_simple(
    question: str,
    ref_target: int,
    enable_gplates: bool,
    anim_ms: int = 800,
    gplates_model: str = DEFAULT_GPLATES_MODEL,
    enable_mindat: bool = True,  # MINDAT
    enable_ddg_agent: bool = DDG_AGENT_ENABLED_DEFAULT,
    ddg_rounds_n: int = DDG_AGENT_ROUNDS_DEFAULT,
    ddg_max_per_round: int = DDG_AGENT_MAX_PER_ROUND,
    enable_kg: bool = True,
    save_rdf: bool = False,
) -> Tuple[str, Optional[str], str, list, Optional[str], str, Optional[str], str, Optional[str], list, list]:

        
    # ---------- placeholders ----------
    out_md: str = "### LLM Generated Snippet\n\n(No summary generated.)\n\n## Sources\n\n(none)"
    gplates_gif_path: Optional[str] = None
    macro_md: str = "Macrostrat panel not run."
    macro_rows: list = []
    mindat_img_path, mindat_md_txt, mindat_group = mindat_panel_from_query(question, return_groups=True)
    mindat_md: str = "Mindat panel not run."
    year_hist_path: Optional[str] = None
    ddg_summary_md: str = "### DuckDuckGo PDF Agent — Summary\n\n_Agent disabled._"
    kg_img_path: Optional[str] = None
    kg_nodes_rows: list[list] = []
    kg_edges_rows: list[list] = []

    has_question = bool((question or "").strip())

    
    # ---------- References + short answer ----------
    if has_question:
        ref_target = int(max(5, min(ref_target, REF_MAX)))
        refs = discover_references(question, ref_target=ref_target)
        sources_index = build_sources_index(refs)
        sources_index_md = "\n".join(f"- {line}" for line in sources_index["numbered"])
        client = _client()
        answer = write_50w_answer(client, question, sources_index_md=sources_index_md).strip()
        sources_md = format_sources_md(refs)
        out_md = f"### LLM Generated Snippet\n\n{answer}\n\n## Sources\n\n{sources_md}"

    # ---------- GPlates (optional; only makes sense with a query) ----------
    if enable_gplates and has_question:
        try:
            start_ma, end_ma = infer_age_range_from_query_llm_first(question, fallback=(200, 0))
            start_ma = int(max(0, min(540, start_ma)))
            end_ma   = int(max(0, min(540, end_ma)))
            if start_ma < end_ma:
                start_ma, end_ma = end_ma, start_ma

            bbox = _infer_bbox_from_query(question)
            if bbox:
                bbox = expand_bbox(
                    bbox, factor=BBOX_EXPAND,
                    min_lon_span=BBOX_MIN_LON_DEG, min_lat_span=BBOX_MIN_LAT_DEG,
                    pad_deg=BBOX_PAD_DEG,
                )
            else:
                lon0, lat0 = _infer_point_from_query(question)
                half_lon = max(BBOX_MIN_LON_DEG / 2.0, 20.0)
                half_lat = max(BBOX_MIN_LAT_DEG / 2.0, 12.0)
                bbox = (
                    max(-180.0, lon0 - half_lon), min(180.0, lon0 + half_lon),
                    max( -90.0, lat0 - half_lat), min( 90.0, lat0 + half_lat),
                )
                bbox = expand_bbox(
                    bbox, factor=BBOX_EXPAND,
                    min_lon_span=BBOX_MIN_LON_DEG, min_lat_span=BBOX_MIN_LAT_DEG,
                    pad_deg=BBOX_PAD_DEG,
                )

            gif = build_gplates_animation_gif_fast(
                user_query=question,
                start_ma=int(start_ma),
                end_ma=int(end_ma),
                max_frames=12,
                model=gplates_model,
                extent=bbox,
                per_request_connect=2.5,
                per_request_read=5.0,
                global_deadline_s=30.0,
                concurrency=4,
                show_point=True,
                frame_duration_ms=int(anim_ms),
            )
            if gif and os.path.isfile(gif) and os.path.getsize(gif) > 0:
                gplates_gif_path = gif
            else:
                try:
                    gif2 = create_local_placeholder_gif(question)
                    if gif2 and os.path.isfile(gif2) and os.path.getsize(gif2) > 0:
                        gplates_gif_path = gif2
                except Exception:
                    pass
        except Exception as e:
            print(f"[GPlates] Error: {e}", flush=True)

    # ---------- Macrostrat (query-dependent) ----------
    if has_question:
        macro_md, macro_rows = query_macrostrat_from_text(question)
    else:
        macro_md, macro_rows = "Enter a query to fetch Macrostrat results.", []

# ---------- Mindat (country + mineral from query) ----------
    try:
        if enable_mindat and has_question:
            mindat_img_path, mindat_md = mindat_panel_from_query(question)
        else:
            mindat_img_path, mindat_md = (
                None,
                "Mindat panel disabled." if not enable_mindat else "Mindat panel idle (no query)."
            )
    except Exception as e:
        mindat_img_path, mindat_md = None, f"Mindat panel error: {e}"
            
    # ---------- DuckDuckGo PDF Agent (downloads PDFs to DOWNLOAD_DIR) ----------
    if enable_ddg_agent and has_question:
        try:
            saved, logs, queries_used, domain_counts, details = ddg_pdf_search_agent(
                user_query=question,
                rounds=int(ddg_rounds_n),
                max_per_round=int(ddg_max_per_round),
                per_site_limit=DDG_AGENT_PER_SITE_LIMIT,
                max_file_mb=DDG_AGENT_MAX_FILE_MB,
                concurrency=DDG_AGENT_CONCURRENCY,
            )
            year_hist_path = _save_pdf_year_histogram(details)

            # compact summary
            L = ["### DuckDuckGo PDF Agent — Summary", ""]
            L.append(f"- Saved **{len(saved)}** PDF(s) to `{_downloads_dir()}`.")
            if queries_used:
                L.append("**Queries used:**")
                for qx in dict.fromkeys(queries_used):  # preserve order, unique
                    L.append(f"- `{qx}`")
            if domain_counts:
                L.append("**Downloads by domain:**")
                for dom, cnt in domain_counts.items():
                    L.append(f"- `{dom}` — {cnt} file(s)")
                    
            if details:
                L.append("**Downloads (title · authors · year · domain · file):**")
                L.append("")
                for d in details:
                    title = d.get("title") or (d.get("filename") or "[untitled].pdf")
                    authors = d.get("authors") or "Unknown"
                    years = d.get("years") or "n.d."
                    dom = d.get("domain") or ""
                    fname = d.get("filename") or ""
                    verdict = d.get("llm_is_geo")
                    conf = d.get("llm_conf")
                    extra = f" · LLM:{verdict} ({conf:.2f})" if verdict is not None and conf is not None else ""
                    L.append(f"- **{title}**  \n  _{authors}_ · _{years}_ · `{dom}` · `{fname}`{extra}")
                    snip = (d.get("snippet") or "").strip()
                    if snip:
                        L.append(f"\n> {snip}\n")
                                        
            ddg_summary_md = "\n".join(L)
        except Exception as e:
            ddg_summary_md = f"### DuckDuckGo PDF Agent — Summary\n\n_Error: {e}_"

    # ---------- Knowledge Graph (can run with or without a query) ----------
    if enable_kg:
        try:
            kg_img_path, kg_nodes_rows, kg_edges_rows = _kg_run_pipeline()
            if not kg_img_path:
                kg_img_path = _save_kg_network_png(kg_nodes_rows, kg_edges_rows)

            # Only save RDF if content exists
            if save_rdf and kg_nodes_rows:
                rdf_path = os.path.join(_downloads_dir(), "GEOAssist KnowledgeGraph.RDF")
                _save_rdf_simple(kg_nodes_rows, kg_edges_rows, rdf_path)
                print(f"[KG][RDF] Saved: {rdf_path}", flush=True)

            print(f"[KG][UI] Rendered: nodes={len(kg_nodes_rows)} edges={len(kg_edges_rows)} → {kg_img_path}", flush=True)
        except Exception as e:
            print(f"[KG][UI] Error preparing KG outputs: {e}", flush=True)

    return (out_md, gplates_gif_path, macro_md, macro_rows,
                mindat_img_path, mindat_md,
                year_hist_path, ddg_summary_md, kg_img_path, kg_nodes_rows, kg_edges_rows, mindat_group)
    
# GRADIO UI #####################################################################################

try:
    if not os.path.exists(ICON_PATH):
        r = requests.get(ICON_URL, timeout=15); r.raise_for_status()
        with open(ICON_PATH, "wb") as f: f.write(r.content)
except Exception as e:
    print(f"[GeoResearch] Warning: could not cache icon: {e}")

with gr.Blocks(title="Geological Deep Research Agent for Literature and Data Searches") as demo:    
    gr.HTML(f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
          <img src="{ICON_URL}" alt="App icon"
               style="width:{ICON_SIZE}px;height:{ICON_SIZE}px;border-radius:12px;object-fit:contain;">
          <div>
            <h1 style="margin:0; font-size:clamp(32px, 4.5vw, 56px); line-height:1.15; font-weight:700;">
              GEOAssist Agent V3.0
            </h1>
            <p style="margin:8px 0 0 0; font-size:clamp(16px, 1.4vw, 35px); line-height:1.6;">
              An open-source LLM driven autonomous research agent for geoscience data and literature.
            </p>
          </div>
        </div>
    """)
    mindat_state = gr.State([])  # holds groups for CSV export
    # Question (left) + Button (right)
    with gr.Row():
        q = gr.Textbox(
            label="Research question",
            placeholder="e.g., Jurassic rifting controls on North Sea basin architecture and source–reservoir juxtaposition",
            lines=3,
            scale=4,
        )
        go = gr.Button("Click to run agents", variant="primary", scale=1)

    with gr.Row():
        refs = gr.Slider(5, REF_MAX, value=REF_TARGET_DEFAULT, step=5, label="Bibliographic References Target")
        
    with gr.Row():
        do_ddg = gr.Checkbox(value=DDG_AGENT_ENABLED_DEFAULT, label="DuckDuckGo Adaptive Search PDF Download Agent")
        ddg_rounds = gr.Slider(1, 100, value=DDG_AGENT_ROUNDS_DEFAULT, step=1, label="PDF Agent Rounds")
        ddg_max_per = gr.Slider(0, 100, value=DDG_AGENT_MAX_PER_ROUND, step=10, label="Max PDFs per round")
    
    with gr.Row():
        do_gplates = gr.Checkbox(value=True, label="Global Plate Tectonics (GPlates)")
        anim_ms = gr.Slider(100, 2000, value=800, step=50, label="Animation speed (ms/frame)")
        gplates_model = gr.Dropdown(choices=GPLATES_MODELS, value=DEFAULT_GPLATES_MODEL, label="GPlates model")
        do_mindat = gr.Checkbox(value=True, label="Mindat mineral occurrences panel")
    
    with gr.Row():
        do_kg  = gr.Checkbox(value=True,  label="Build Geoscience Knowledge Graph (GeoKG) from PDFs")
        save_rdf = gr.Checkbox(value=True, label="Save GeoKG to RDF")

    with gr.Row():
        # LEFT: DuckDuckGo summary
        with gr.Column(scale=1):
            with gr.Accordion("DuckDuckGo PDF Agent — Summary", open=True):
                year_hist_img = gr.Image(type="filepath", label="PDFs by Year (from DDG Agent)")
                ddg_agent_md = gr.Markdown()
    
        # RIGHT: GPlates + Macrostrat + Mindat + Answer
        with gr.Column(scale=1):
            gplates_img = gr.Image(type="filepath", label="GPlates reconstruction (GIF)")
    
            with gr.Accordion("Macrostrat (auto from your query)", open=True):
                macro_md = gr.Markdown(label="Macrostrat query & exports")
                macro_tbl = gr.Dataframe(
                    headers=["unit_or_name", "strat_name", "best_int", "t_age", "b_age", "lith", "source_id", "map_id"],
                    label="Macrostrat units (overlapping age window at detected location)",
                    wrap=True,
                    interactive=False
                )
                
            # Answer + Sources BELOW Macrostrat & Mindat
            out = gr.Markdown(label="Answer + Sources")
    
    # --- NEW: Knowledge Graph panel ---
    # NOTE: If you run on more than a few PDFs, the graph will be so dense the visual will not be readable
    #  But that is ok - use the RDF export file so it can be examined in more detail in a specific Graph Application
    with gr.Accordion("Knowledge Graph (from PDFs in DOWNLOAD_DIR)", open=True):
        kg_img = gr.Image(type="filepath", label="Knowledge Graph (nodes & edges)", show_download_button=True)
        kg_nodes_tbl = gr.Dataframe(
            headers=["node", "type", "count"],
            label="Nodes (entity mention counts)",
            wrap=True,
            interactive=False
        )
        kg_edges_tbl = gr.Dataframe(
            headers=["source", "target", "weight"],
            label="Edges (co-mentions within sentence; weight = frequency)",
            wrap=True,
            interactive=False
        )

    # --- Mindat Panel ---            
    with gr.Accordion("Mindat occurrences (auto from your query)", open=True):
        mindat_img = gr.Image(
            type="filepath",
            label="Mineral, Element and Rock occurrences map (Mindat)",
            show_download_button=True,
            height=820,        # ← taller “window” in the UI
            elem_id="mindat_img"
        )
        mindat_md = gr.Markdown()
        mindat_dl  = gr.DownloadButton("Export occurrences CSV", variant="secondary")  # Export Mindat occurrences to CSV

    go.click(
            fn=run_simple,
            inputs=[q, refs, do_gplates, anim_ms, gplates_model, do_mindat, do_ddg, ddg_rounds, ddg_max_per, do_kg, save_rdf],
            outputs=[out, gplates_img, macro_md, macro_tbl, mindat_img, mindat_md, year_hist_img, ddg_agent_md, kg_img, kg_nodes_tbl, kg_edges_tbl, mindat_state],
        )

    mindat_dl.click(fn=_mindat_groups_to_csv, inputs=mindat_state, outputs=mindat_dl)
    
# MAIN AND LAUNCH #####################################################################################
if __name__ == "__main__":
    demo.queue().launch()
    
# END OF CODE ########################################################################################