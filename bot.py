# travelbuddy_bot.py

import os
import io
import csv
import logging
import sys
import argparse
import math
from typing import Tuple, List, Dict, Any, Optional
from dotenv import load_dotenv
import requests
import pandas as pd
import datetime as dt

# ===== Инициализация/логирование =====
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")  # обязателен
WORLDNEWS_API_KEY = os.getenv("WORLDNEWS_API_KEY")  # для новостей
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY")    # apilayer currency_data
WM_USER_AGENT = os.getenv("WM_USER_AGENT", "TravelBuddyBot/1.0 (+contact: example@example.com)")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# В некоторых средах TeleBot может отсутствовать — ок для офлайн-чека
try:
    import telebot
    from telebot import types
    bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML") if BOT_TOKEN else None
except Exception as e:
    logging.warning("TeleBot не инициализирован: %s", e)
    bot = None

# ===== Роль/дисклеймер =====
ROLE_NAME = "TravelBuddy"
ROLE_PROMPT = (
    "Ты — ассистент путешествий: помоги выбрать страну/город, покажи погоду, новости, курсы валют, достопримечательности и праздники. "
    "Пиши кратко и по делу. Если данных не хватает — попроси уточнение. "
)
DISCLAIMER = "ℹ️ Данные из публичных API. Возможны задержки/неточности."

# ===== Пользовательский контекст (in-memory) =====
# chat_id -> {country, country_code, currency, capital, lat, lon, user_base_currency}
USER_CTX: Dict[int, Dict[str, Any]] = {}

# ===== Меню =====
def main_menu_kb():
    if not bot:
        return None
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.row("🎯 Выбрать страну", "🌍 Что выбрано")
    kb.row("📰 Новости", "💱 Курс валют", "🌦 Погода")
    kb.row("🗺️ Достопримечательности", "🎉 Праздники", "🆘 Помощь")
    return kb

# ===== УТИЛИТЫ: гео/страны/валюта =====
def restcountries_by_name(q: str) -> Optional[dict]:
    url = "https://restcountries.com/v3.1/name/" + requests.utils.quote(q)
    params = {"fields": "name,cca2,capital,currencies,latlng"}
    r = requests.get(url, params=params, timeout=12, headers={"User-Agent": WM_USER_AGENT})
    if r.status_code != 200:
        return None
    data = r.json()
    if not isinstance(data, list) or not data:
        return None
    return data[0]

def geocode_city_openmeteo(name: str, country_code: Optional[str] = None) -> Optional[dict]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": name, "count": 1, "language": "en", "format": "json"}
    if country_code:
        params["filter"] = "countrycode"
        params["country"] = country_code.upper()
    r = requests.get(url, params=params, timeout=10, headers={"User-Agent": WM_USER_AGENT})
    if r.status_code != 200:
        return None
    data = r.json() or {}
    results = data.get("results") or []
    if not results:
        return None
    it = results[0]
    return {
        "name": it.get("name"),
        "lat": it.get("latitude"),
        "lon": it.get("longitude"),
        "country_code": it.get("country_code"),
        "admin1": it.get("admin1")
    }

def get_weather_open_meteo(lat: float, lon: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,weather_code",
        "hourly": "temperature_2m,precipitation_probability,weather_code",
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=12, headers={"User-Agent": WM_USER_AGENT})
    r.raise_for_status()
    return r.json()

# ===== Валюты =====
def get_fx_quotes_usd(symbols: str) -> dict[str, float]:
    if not CURRENCY_API_KEY:
        raise RuntimeError("CURRENCY_API_KEY не задан. Добавьте ключ в .env")
    url = "https://api.apilayer.com/currency_data/live"
    params = {"source": "USD", "currencies": symbols}
    headers = {"apikey": CURRENCY_API_KEY, "User-Agent": WM_USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        err = data.get("error") or {}
        raise RuntimeError(f"API error: {err.get('type') or err.get('info') or 'unknown'}")
    quotes = data.get("quotes") or {}
    out: dict[str, float] = {}
    for sym in [s.strip().upper() for s in symbols.split(",") if s.strip()]:
        key = "USD" + sym
        if key in quotes:
            out[sym] = float(quotes[key])
    if not out:
        raise RuntimeError("Пустой ответ по котировкам.")
    return out

def cross_rate(base: str, target: str, usd_quotes: dict[str, float]) -> Optional[float]:
    base = base.upper()
    target = target.upper()
    if base == target:
        return 1.0
    u2t = usd_quotes.get(target)
    u2b = usd_quotes.get(base)
    if not (u2t and u2b and u2b > 0):
        return None
    return u2t / u2b

# ===== Новости (РУ по умолчанию, фолбэк без языка) =====
def get_worldnews(country_code: str, limit: int = 5, language: Optional[str] = "ru") -> list[dict]:
    if not WORLDNEWS_API_KEY:
        raise RuntimeError("WORLDNEWS_API_KEY не задан. Добавь ключ в .env")

    def _fetch(lang: Optional[str]) -> list[dict]:
        url = "https://api.worldnewsapi.com/search-news"
        params = {
            "api-key": WORLDNEWS_API_KEY,
            "source-countries": country_code.lower(),
            "sort": "publish-time",
            "sort-direction": "desc",
            "number": limit,
            "offset": 0,
        }
        if lang:
            params["language"] = lang
        today = dt.date.today()
        week_ago = today - dt.timedelta(days=7)
        params["earliest-publish-date"] = week_ago.isoformat()
        params["latest-publish-date"] = today.isoformat()

        r = requests.get(url, params=params, timeout=12, headers={"User-Agent": WM_USER_AGENT})
        r.raise_for_status()
        data = r.json() or {}
        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(data.get("message") or "WorldNewsAPI вернул ошибку")
        items = data.get("news", []) or []
        out: list[dict] = []
        for a in items[:limit]:
            out.append({
                "title": a.get("title") or "Без заголовка",
                "url": a.get("url"),
                "publish_date": a.get("publish_date"),
                "source": a.get("author") or a.get("source_country"),
                "summary": a.get("summary") or "",
            })
        return out

    items = _fetch(language)
    if not items and language:
        items = _fetch(None)
    return items

# ===== Праздники (Nager.Date, без ключа) =====
def get_next_public_holidays(country_code: str, limit: int = 5) -> list[dict]:
    if not country_code:
        return []
    url = f"https://date.nager.at/api/v3/NextPublicHolidays/{country_code.upper()}"
    r = requests.get(url, timeout=10, headers={"User-Agent": WM_USER_AGENT, "Accept": "application/json"})
    if r.status_code != 200:
        return []
    data = r.json() or []
    out = []
    today = dt.date.today()
    for it in data[:limit]:
        try:
            d = dt.date.fromisoformat(it.get("date"))
        except Exception:
            continue
        days_left = (d - today).days
        out.append({
            "date": d.isoformat(),
            "name": it.get("name") or it.get("localName") or "Holiday",
            "local_name": it.get("localName"),
            "global": bool(it.get("global")),
            "counties": it.get("counties"),
            "days_left": days_left
        })
    return out[:limit]

def fmt_holidays(items: list[dict], country_code: str) -> str:
    if not items:
        return f"🎉 Праздники ({country_code.upper()}): ближайших не найдено.\n{DISCLAIMER}"
    lines = [f"🎉 <b>Ближайшие праздники ({country_code.upper()})</b>"]
    for i, h in enumerate(items, 1):
        date = h.get("date")
        name = h.get("name") or "Holiday"
        loc = h.get("local_name")
        days = h.get("days_left")
        badge = f" — {loc}" if loc and loc != name else ""
        when = f" (через {days} дн.)" if isinstance(days, int) and days >= 0 else ""
        lines.append(f"{i}. {date}: {name}{badge}{when}")
    lines.append("\n" + DISCLAIMER)
    return "\n".join(lines)

# ===== Вспомогательное: дистанция (haversine) =====
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0  # м
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def _dist_km(lat1, lon1, lat2, lon2) -> float:
    return _haversine_m(float(lat1), float(lon1), float(lat2), float(lon2)) / 1000.0

# ===== Достопримечательности: ТОЛЬКО Wikipedia =====
def _wiki_geosearch(lat: float, lon: float, radius_m: int, limit: int, lang: str) -> List[dict]:
    base = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": min(radius_m, 10000),
        "gslimit": min(limit, 50),
        "format": "json",
        "formatversion": "2",
    }
    headers = {"User-Agent": WM_USER_AGENT, "Accept": "application/json"}
    r = requests.get(base, params=params, headers=headers, timeout=12)
    r.raise_for_status()
    data = r.json() or {}
    return ((data.get("query") or {}).get("geosearch")) or []

def _wiki_pages_details(page_ids: List[int], lang: str) -> Dict[int, dict]:
    if not page_ids:
        return {}
    base = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts|pageimages|info",
        "pageids": "|".join(str(p) for p in page_ids[:50]),
        "exsentences": "2",
        "explaintext": "1",
        "pithumbsize": "320",
        "inprop": "url",
        "format": "json",
        "formatversion": "2",
    }
    headers = {"User-Agent": WM_USER_AGENT, "Accept": "application/json"}
    r = requests.get(base, params=params, headers=headers, timeout=12)
    r.raise_for_status()
    data = r.json() or {}
    pages = (data.get("query") or {}).get("pages") or []
    out: Dict[int, dict] = {}
    for p in pages:
        out[p.get("pageid")] = {
            "fullurl": p.get("fullurl"),
            "extract": p.get("extract"),
            "thumbnail": ((p.get("thumbnail") or {}).get("source")),
        }
    return out

def get_attractions_wikipedia(lat: float, lon: float, radius_m: int = 5000, limit: int = 10, lang: str = "ru") -> list[dict]:
    items = _wiki_geosearch(lat, lon, radius_m, limit, lang)
    if not items:
        return []
    page_ids = [it.get("pageid") for it in items if it.get("pageid")]
    details = _wiki_pages_details(page_ids, lang)
    out = []
    for it in items[:limit]:
        pid = it.get("pageid")
        det = details.get(pid, {})
        out.append({
            "name": it.get("title") or "Unnamed",
            "kinds": "Wikipedia",
            "dist_m": it.get("dist"),
            "website": det.get("fullurl"),
            "address": None,
            "summary": det.get("extract"),
            "thumb": det.get("thumbnail"),
        })
    return out

def get_attractions(lat: float, lon: float, radius_m: int = 5000, limit: int = 10) -> list[dict]:
    try:
        res = get_attractions_wikipedia(lat, lon, radius_m, limit, lang="ru")
        if res:
            return res
    except Exception as e:
        logging.warning("Wikipedia RU failed: %s", e)
    try:
        return get_attractions_wikipedia(lat, lon, radius_m, limit, lang="en")
    except Exception as e:
        logging.error("Wikipedia EN failed: %s", e)
        return []

# ===== Планировщик из CSV =====
CSV_EXPECTED_COLS = ["name", "lat", "lon", "duration_min"]

def _normalize_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим колонки к ожидаемым: name, lat, lon, duration_min, priority(optional)
    duration_min — длительность визита в минутах.
    priority — целое 1..5 (5 = must-see). Если нет — 3.
    """
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_name = pick("name", "title", "place", "место", "название")
    c_lat  = pick("lat", "latitude", "широта")
    c_lon  = pick("lon", "lng", "long", "longitude", "долгота")
    c_dur  = pick("duration_min", "duration", "visit_min", "время", "длительность", "минуты")
    c_pr   = pick("priority", "prio", "rating", "рейтинг", "приоритет")

    need = [c_name, c_lat, c_lon, c_dur]
    if any(x is None for x in need):
        raise ValueError("CSV должен содержать колонки: name, lat, lon, duration_min (возможны синонимы).")

    out = pd.DataFrame({
        "name": df[c_name].astype(str).str.strip(),
        "lat": pd.to_numeric(df[c_lat], errors="coerce"),
        "lon": pd.to_numeric(df[c_lon], errors="coerce"),
        "duration_min": pd.to_numeric(df[c_dur], errors="coerce"),
    })
    out["priority"] = 3
    if c_pr:
        pr = pd.to_numeric(df[c_pr], errors="coerce")
        out.loc[~pr.isna(), "priority"] = pr.astype(int).clip(1, 5)

    out = out.dropna(subset=["lat", "lon", "duration_min"])
    out = out[out["duration_min"] > 0]
    out = out.reset_index(drop=True)
    if out.empty:
        raise ValueError("После нормализации не осталось валидных строк.")
    return out

def build_itinerary_from_csv(df: pd.DataFrame, ctx: Dict[str, Any], day_budget_min: int = 420, walk_min_per_km: int = 12):
    """
    Простая эвристика:
      1) Сортируем по priority (desc), затем «ближайший сосед».
      2) Раскладываем по дням в рамках дневного бюджета.
    Старт: координаты выбранной столицы (если есть) либо первая точка.
    """
    places = _normalize_csv(df).copy()

    start_lat = ctx.get("lat")
    start_lon = ctx.get("lon")
    if start_lat is None or start_lon is None:
        start_lat, start_lon = places.loc[0, "lat"], places.loc[0, "lon"]

    places = places.sort_values(["priority"], ascending=False).reset_index(drop=True)
    used = set()
    order = []
    cur_lat, cur_lon = start_lat, start_lon

    # Жадный ближайший сосед (с небольшим штрафом за низкий приоритет)
    while len(used) < len(places):
        best_idx, best_score = None, None
        for idx, row in places.iterrows():
            if idx in used:
                continue
            d = _dist_km(cur_lat, cur_lon, row.lat, row.lon)
            score = d + (5 - int(row.priority)) * 0.3
            if best_score is None or score < best_score:
                best_idx, best_score = idx, score
        used.add(best_idx)
        order.append(best_idx)
        cur_lat, cur_lon = places.loc[best_idx, "lat"], places.loc[best_idx, "lon"]

    # Раскладываем по дням
    days: List[List[dict]] = []
    day: List[dict] = []
    budget_left = day_budget_min
    cur_lat, cur_lon = start_lat, start_lon

    for idx in order:
        row = places.loc[idx]
        travel_min = int(round(_dist_km(cur_lat, cur_lon, row.lat, row.lon) * walk_min_per_km))
        block_min = travel_min + int(row.duration_min)

        if block_min <= budget_left or not day:
            day.append({
                "name": row.name,
                "lat": float(row.lat),
                "lon": float(row.lon),
                "visit_min": int(row.duration_min),
                "travel_min": travel_min,
                "priority": int(row.priority),
            })
            budget_left -= block_min
            cur_lat, cur_lon = row.lat, row.lon
        else:
            days.append(day)
            day = [{
                "name": row.name,
                "lat": float(row.lat),
                "lon": float(row.lon),
                "visit_min": int(row.duration_min),
                "travel_min": int(round(_dist_km(start_lat, start_lon, row.lat, row.lon) * walk_min_per_km)),
                "priority": int(row.priority),
            }]
            budget_left = day_budget_min - (day[0]["visit_min"] + day[0]["travel_min"])
            cur_lat, cur_lon = row.lat, row.lon

    if day:
        days.append(day)

    return {
        "days": days,
        "config": {
            "day_budget_min": day_budget_min,
            "walk_min_per_km": walk_min_per_km,
            "start_lat": start_lat,
            "start_lon": start_lon,
        }
    }

def fmt_itinerary(plan: dict, ctx: Dict[str, Any]) -> str:
    country = ctx.get("country")
    capital = ctx.get("capital")
    header = f"🗺️ <b>План поездки</b>"
    if country or capital:
        extra = " — "
        if country: extra += f"{country}"
        if capital: extra += f" / база: {capital}"
        header += extra
    lines = [header]
    cfg = plan.get("config", {})
    lines.append(f"• Бюджет на день: ~{cfg.get('day_budget_min', 420)} мин, скорость: ~{cfg.get('walk_min_per_km', 12)} мин/км.")

    for di, day in enumerate(plan.get("days", []), 1):
        lines.append(f"\n<b>День {di}</b>")
        total_walk = 0
        total_visit = 0
        for i, p in enumerate(day, 1):
            total_walk += p["travel_min"]
            total_visit += p["visit_min"]
            tleg = f"→ пешком ~{p['travel_min']} мин • " if p["travel_min"] > 0 else ""
            lines.append(f"{i}. {tleg}{p['name']} (⏱ {p['visit_min']} мин, приор. {p['priority']})")
        lines.append(f"Итого за день: ходьба ~{total_walk} мин, посещения ~{total_visit} мин, всего ~{total_walk + total_visit} мин.")
    lines.append("\n" + DISCLAIMER)
    return "\n".join(lines)

# ===== Старый CSV-отчёт (оставил на месте — вдруг пригодится в CLI) =====
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV не найден: {path}")
    return pd.read_csv(path)

# ===== ТЕКСТОВЫЕ ФОРМАТТЕРЫ =====
def fmt_country_summary(meta: dict) -> str:
    name = (meta.get("name") or {}).get("common") or "—"
    code = meta.get("cca2") or "—"
    capital = ", ".join(meta.get("capital") or []) or "—"
    latlng = meta.get("latlng") or [None, None]
    lat, lon = (latlng[0], latlng[1]) if len(latlng) >= 2 else (None, None)
    currencies = meta.get("currencies") or {}
    cur_codes = ", ".join(sorted(currencies.keys())) or "—"
    return (
        f"🎯 <b>Страна выбрана:</b> {name} ({code})\n"
        f"🏛️ Столица: {capital}\n"
        f"💱 Валюта(ы): {cur_codes}\n"
        f"📍 Координаты столицы: {lat}, {lon}\n"
    )

def fmt_weather(data: dict) -> str:
    cur = data.get("current", {})
    tz = data.get("timezone", "local")
    t = cur.get("temperature_2m")
    wind = cur.get("wind_speed_10m")
    rh = cur.get("relative_humidity_2m")
    code = cur.get("weather_code")
    return (
        "🌦 <b>Погода (Open-Meteo)</b>\n"
        f"• Температура: {t} °C\n"
        f"• Ветер: {wind} m/s\n"
        f"• Влажность: {rh} %\n"
        f"• Погодный код: {code}\n"
        f"🕒 Часовой пояс: {tz}\n"
        f"\n" + DISCLAIMER
    )

def fmt_news(items: list[dict], country_code: str) -> str:
    if not items:
        return f"📰 Новости по стране {country_code}: ничего не найдено за неделю.\n{DISCLAIMER}"
    lines = [f"📰 <b>Новости ({country_code.upper()})</b>"]  # RU default
    for i, a in enumerate(items, 1):
        date = (a.get("publish_date") or "")[:10]
        src = a.get("source") or ""
        url = a.get("url") or ""
        title = a.get("title") or "Без заголовка"
        line = f"{i}. <a href=\"{url}\">{title}</a>"
        if date:
            line += f" — {date}"
        if src:
            line += f" ({src})"
        lines.append(line)
    lines.append("\n" + DISCLAIMER)
    return "\n".join(lines)

def fmt_attractions(items: list[dict]) -> str:
    if not items:
        return "🗺️ Достопримечательности: ничего не нашли рядом.\n" + DISCLAIMER
    lines = ["🗺️ <b>Топ-места рядом (Wikipedia)</b>"]
    for i, it in enumerate(items, 1):
        name = it.get("name") or "Unnamed"
        dist = it.get("dist_m")
        web = it.get("website")
        summary = it.get("summary") or ""
        summary = (summary[:220] + "…") if len(summary) > 220 else summary
        suffix = f" — ~{int(dist)} м" if isinstance(dist, (int, float)) else ""
        line = f"{i}. {name}{suffix}"
        if web:
            line += f"\n   🔗 <a href=\"{web}\">Статья</a>"
        if summary:
            line += f"\n   {summary}"
        lines.append(line)
    lines.append("\n" + DISCLAIMER)
    return "\n".join(lines)

# ===== ХЕНДЛЕРЫ TELEGRAM =====
if bot:
    @bot.message_handler(commands=["start"])
    def handle_start(message):
        user = message.from_user
        text = (
            f"Привет, <b>{user.first_name or 'друг'}</b>! 👋\n"
            f"Я — {ROLE_NAME}. Помогу подготовиться к поездке: страна, новости, курсы, погода, места и праздники.\n"
            f"Команды: /help\n"
        )
        bot.send_message(message.chat.id, text, reply_markup=main_menu_kb())

    @bot.message_handler(commands=["help"])
    def handle_help(message):
        help_text = (
            "<b>Команды</b>:\n"
            "/start — приветствие\n"
            "/help — список команд\n"
            "/set_destination &lt;страна&gt; — выбрать страну поездки (например: /set_destination Japan)\n"
            "/country — показать текущую выбранную страну\n"
            "/news — новости по выбранной стране (на русском, если доступны)\n"
            "/fx [&lt;base=EUR&gt;] — курс базовой валюты к валюте страны (крест через USD)\n"
            "/weather [город] — погода по столице страны или по городу\n"
            "/sights [город] — достопримечательности (Wikipedia) рядом со столицей или указанным городом\n"
            "/holidays — ближайшие национальные праздники выбранной страны\n"
            "/csv_template — прислать шаблон CSV для маршрута\n"
            "💡 Пришли свой CSV с местами — соберу черновой план по дням.\n"
            "\n" + DISCLAIMER
        )
        bot.reply_to(message, help_text)

    @bot.message_handler(commands=["csv_template"])
    def handle_csv_template(message):
        """
        Шаблон CSV: name,lat,lon,duration_min,priority
        """
        sample = io.StringIO()
        writer = csv.writer(sample)
        writer.writerow(["name","lat","lon","duration_min","priority"])
        writer.writerow(["Музей искусства","35.6895","139.6917","90","5"])
        writer.writerow(["Синтоистский храм","35.6733","139.7104","60","4"])
        writer.writerow(["Смотровая площадка","35.6586","139.7454","45","3"])
        data = sample.getvalue().encode("utf-8")
        bot.send_document(
            message.chat.id,
            io.BytesIO(data),
            visible_file_name="travel_template.csv",
            caption="Шаблон CSV. Обязательные поля: name, lat, lon, duration_min. priority — опционально (1..5)."
        )

    @bot.message_handler(commands=["set_destination"])
    def handle_set_destination(message):
        chat_id = message.chat.id
        parts = (message.text or "").split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            bot.reply_to(message, "Использование: <code>/set_destination Japan</code>")
            return
        query = parts[1].strip()
        try:
            meta = restcountries_by_name(query)
            if not meta:
                bot.reply_to(message, "Не нашёл такую страну. Попробуй точнее (English name).")
                return
            USER_CTX.setdefault(chat_id, {})
            USER_CTX[chat_id]["country"] = (meta.get("name") or {}).get("common")
            USER_CTX[chat_id]["country_code"] = meta.get("cca2")
            currencies = meta.get("currencies") or {}
            cur_code = next(iter(currencies.keys()), None)
            USER_CTX[chat_id]["currency"] = cur_code
            capital = (meta.get("capital") or [None])[0]
            USER_CTX[chat_id]["capital"] = capital
            latlng = meta.get("latlng") or [None, None]
            if len(latlng) >= 2:
                USER_CTX[chat_id]["lat"] = latlng[0]
                USER_CTX[chat_id]["lon"] = latlng[1]
            USER_CTX[chat_id].setdefault("user_base_currency", "EUR")
            bot.reply_to(message, "Готово.\n" + fmt_country_summary(meta))
        except Exception as e:
            bot.reply_to(message, f"Не удалось выбрать страну: {e}")

    @bot.message_handler(commands=["country"])
    def handle_country(message):
        chat_id = message.chat.id
        ctx = USER_CTX.get(chat_id)
        if not ctx or not ctx.get("country_code"):
            bot.reply_to(message, "Страна ещё не выбрана. Используй: <code>/set_destination Italy</code>")
            return
        meta = {
            "name": {"common": ctx.get("country")},
            "cca2": ctx.get("country_code"),
            "capital": [ctx.get("capital")] if ctx.get("capital") else [],
            "currencies": {ctx.get("currency"): {}} if ctx.get("currency") else {},
            "latlng": [ctx.get("lat"), ctx.get("lon")]
        }
        bot.reply_to(message, fmt_country_summary(meta))

    @bot.message_handler(commands=["news"])
    def handle_news(message):
        chat_id = message.chat.id
        ctx = USER_CTX.get(chat_id) or {}
        code = (ctx.get("country_code") or "").upper()
        if not code:
            bot.reply_to(message, "Сначала выбери страну: <code>/set_destination Spain</code>")
            return
        try:
            items = get_worldnews(code, limit=5, language="ru")
            bot.reply_to(message, fmt_news(items, code))
        except Exception as e:
            bot.reply_to(message, f"Не удалось загрузить новости: {e}")

    @bot.message_handler(commands=["fx"])
    def handle_fx(message):
        chat_id = message.chat.id
        ctx = USER_CTX.get(chat_id) or {}
        if not ctx.get("country_code") or not ctx.get("currency"):
            bot.reply_to(message, "Выбери страну, чтобы узнать местную валюту: <code>/set_destination Thailand</code>")
            return
        base = ctx.get("user_base_currency", "EUR")
        for part in (message.text or "").split():
            if part.lower().startswith("base="):
                base = part.split("=", 1)[1].strip().upper() or base
        ctx["user_base_currency"] = base

        target = ctx["currency"].upper()
        try:
            syms = ",".join(sorted({base, target}))
            quotes = get_fx_quotes_usd(syms)
            rate_bt = cross_rate(base, target, quotes)
            rate_tb = cross_rate(target, base, quotes)
            lines = ["💱 <b>Курс валют</b> (через USD)"]
            lines.append(f"База: {base} → Местная: {target}")
            if isinstance(rate_bt, float):
                lines.append(f"1 {base} = {rate_bt:.4f} {target}")
            if isinstance(rate_tb, float):
                lines.append(f"1 {target} = {rate_tb:.4f} {base}")
            if base in quotes:
                lines.append(f"1 USD = {quotes[base]:.4f} {base}")
            if target in quotes:
                lines.append(f"1 USD = {quotes[target]:.4f} {target}")
            lines.append("\n" + DISCLAIMER)
            bot.reply_to(message, "\n".join(lines))
        except Exception as e:
            bot.reply_to(message, f"Не удалось посчитать курс: {e}")

    @bot.message_handler(commands=["weather"])
    def handle_weather(message):
        chat_id = message.chat.id
        ctx = USER_CTX.get(chat_id) or {}
        parts = (message.text or "").split(maxsplit=1)
        custom_city = parts[1].strip() if len(parts) > 1 else None

        if custom_city:
            cc = (ctx.get("country_code") or "").upper() or None
            geo = geocode_city_openmeteo(custom_city, country_code=cc)
        else:
            if not ctx.get("capital") or not ctx.get("lat") or not ctx.get("lon"):
                bot.reply_to(message, "Нужно выбрать страну: <code>/set_destination Portugal</code>\n"
                                      "Либо укажи город: <code>/weather Porto</code>")
                return
            geo = {"name": ctx.get("capital"), "lat": ctx.get("lat"), "lon": ctx.get("lon")}

        if not geo:
            bot.reply_to(message, "Не удалось найти такой город. Попробуй иначе.")
            return

        try:
            data = get_weather_open_meteo(geo["lat"], geo["lon"])
            header = f"📍 Локация: {geo.get('name')}"
            bot.reply_to(message, header + "\n" + fmt_weather(data))
        except Exception as e:
            bot.reply_to(message, f"Не удалось получить погоду: {e}")

    @bot.message_handler(commands=["sights"])
    def handle_sights(message):
        chat_id = message.chat.id
        ctx = USER_CTX.get(chat_id) or {}
        parts = (message.text or "").split(maxsplit=1)
        custom_city = parts[1].strip() if len(parts) > 1 else None

        geo = None
        if custom_city:
            cc = (ctx.get("country_code") or "").upper() or None
            geo = geocode_city_openmeteo(custom_city, country_code=cc)
        else:
            if ctx.get("lat") and ctx.get("lon") and ctx.get("capital"):
                geo = {"name": ctx.get("capital"), "lat": ctx.get("lat"), "lon": ctx.get("lon")}

        if not geo:
            bot.reply_to(message, "Укажи город: <code>/sights Kyoto</code> или выбери страну с известной столицей.")
            return

        try:
            items = get_attractions(geo["lat"], geo["lon"], radius_m=5000, limit=10)
            title = f"📍 Район: {geo.get('name')} (радиус ~5 км)\n"
            bot.reply_to(message, title + fmt_attractions(items))
        except Exception as e:
            bot.reply_to(message, f"Не удалось загрузить места: {e}")

    @bot.message_handler(commands=["holidays"])
    def handle_holidays(message):
        chat_id = message.chat.id
        ctx = USER_CTX.get(chat_id) or {}
        code = (ctx.get("country_code") or "").upper()
        if not code:
            bot.reply_to(message, "Сначала выбери страну: <code>/set_destination Spain</code>")
            return
        try:
            items = get_next_public_holidays(code, limit=5)
            bot.reply_to(message, fmt_holidays(items, code))
        except Exception as e:
            bot.reply_to(message, f"Не удалось загрузить праздники: {e}")

    # ==== Загрузка CSV с местами → авто-план ====
    @bot.message_handler(content_types=["document"])
    def handle_csv_upload(message):
        doc = message.document
        if not doc or not (doc.file_name or "").lower().endswith(".csv"):
            return
        try:
            file_info = bot.get_file(doc.file_id)
            content = bot.download_file(file_info.file_path)
            df = pd.read_csv(io.BytesIO(content))

            chat_id = message.chat.id
            ctx = USER_CTX.get(chat_id) or {}
            plan = build_itinerary_from_csv(df, ctx, day_budget_min=420, walk_min_per_km=12)
            bot.reply_to(message, fmt_itinerary(plan, ctx))
        except Exception as e:
            bot.reply_to(
                message,
                "Не удалось обработать CSV. Убедись, что есть колонки "
                "<code>name, lat, lon, duration_min</code> (и, опционально, <code>priority</code> 1..5).\n"
                f"Ошибка: {e}"
            )

    # ==== Кнопки меню ====
    @bot.message_handler(func=lambda m: (m.text or "").strip() in [
        "🎯 Выбрать страну", "🌍 Что выбрано", "📰 Новости", "💱 Курс валют",
        "🌦 Погода", "🗺️ Достопримечательности", "🎉 Праздники", "🆘 Помощь"
    ])
    def handle_buttons(message):
        t = (message.text or "").strip()
        if t == "🎯 Выбрать страну":
            bot.reply_to(message, "Напиши: <code>/set_destination CountryName</code>\nНапр.: <code>/set_destination Japan</code>")
        elif t == "🌍 Что выбрано":
            handle_country(message)
        elif t == "📰 Новости":
            handle_news(message)
        elif t == "💱 Курс валют":
            handle_fx(message)
        elif t == "🌦 Погода":
            handle_weather(message)
        elif t == "🗺️ Достопримечательности":
            handle_sights(message)
        elif t == "🎉 Праздники":
            handle_holidays(message)
        elif t == "🆘 Помощь":
            handle_help(message)

    # ==== Текстовые триггеры (минимум магии) ====
    @bot.message_handler(content_types=["text"])
    def handle_text(message):
        text = (message.text or "").lower()
        if "страна" in text or "поездк" in text:
            bot.reply_to(message, "Используй: <code>/set_destination Greece</code> — и поехали 😎")
            return
        if any(k in text for k in ["погод", "weather"]):
            return handle_weather(message)
        if any(k in text for k in ["новост", "news"]):
            return handle_news(message)
        if any(k in text for k in ["валют", "курс", "fx"]):
            return handle_fx(message)
        if any(k in text for k in ["сайтс", "места", "достопр", "sights"]):
            return handle_sights(message)
        if any(k in text for k in ["праздн", "holiday", "holidays"]):
            return handle_holidays(message)
        bot.reply_to(message, ROLE_PROMPT + "\n\nПодскажи страну или город, с которых начнём?")

# ===== CLI режим (опционально — оставить для теста) =====
def main_cli():
    parser = argparse.ArgumentParser(description="TravelBuddy CLI demo")
    parser.add_argument("--country", default="Japan", help="Страна назначения")
    args = parser.parse_args()
    meta = restcountries_by_name(args.country) or {}
    print(fmt_country_summary(meta or {}))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        main_cli()
    else:
        if bot and BOT_TOKEN:
            logging.info("Запуск бота TravelBuddy…")
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        else:
            logging.error("BOT_TOKEN не задан или TeleBot недоступен. Для CLI: python travelbuddy_bot.py --country Japan")
